"""
Volatility Surface and VRP Engine

Extracts signals from options implied volatility surfaces.
Computes Variance Risk Premium (VRP), volatility skew, and term structure signals.
"""

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from portfolio_execution.signals.base import AlphaModel, SignalDirection, SignalNorm
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VolatilitySurfaceConfig:
    """Configuration for volatility surface calculations."""

    skew_delta: float = 0.25  # 25-delta for skew calculation
    near_term_days: int = 30
    far_term_days: int = 90
    vrp_lookback_days: int = 21  # ~1 trading month


class VolRegime(Enum):
    LOW_VOL = "LOW_VOL"
    NORMAL = "NORMAL"
    HIGH_VOL = "HIGH_VOL"
    CRISIS = "CRISIS"


class VolatilitySurfaceAlpha(AlphaModel):
    """
    Options implied volatility signals.
    """

    def __init__(self, config: VolatilitySurfaceConfig = VolatilitySurfaceConfig(), **kwargs):
        super().__init__(
            name="volatility_surface",
            lookback=kwargs.pop("lookback", 20),
            direction=SignalDirection.LONG_SHORT,
            norm=SignalNorm.ZSCORE,
            **kwargs,
        )
        self.config = config

    def _compute_raw_signal(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Expects pre-computed 'implied_vol', 'realized_vol', 'skew_25d', 'term_structure_ratio'
        """
        required_cols = ["implied_vol", "realized_vol", "skew_25d", "term_structure_ratio"]
        for col in required_cols:
            if col not in data.columns:
                return pd.Series(index=data.index, dtype=float)

        iv = data["implied_vol"]
        rv = data["realized_vol"]
        skew = data["skew_25d"]
        term = data["term_structure_ratio"]

        # 1. Variance Risk Premium (VRP) = IV - RV
        # High VRP -> options are expensive, market pricing in risk
        # Often mean-reverting (short VRP = short premium strategy)
        vrp = iv - rv

        # 2. Skew = Put IV (25d) - Call IV (25d)
        # Steep skew -> high demand for downside protection

        # 3. Term structure = Near month IV / Far month IV
        # > 1 means backwardation (panic)

        # Simple composite vol signal (higher means more bearish/expensive protection)
        # For equities, high vol usually correlates with negative returns
        raw_signal = -(vrp + skew + term)
        return raw_signal


import numpy as np

class VolatilityRegimeDetector:
    """
    Detects the current market volatility regime based on India VIX.
    Uses a 3-State Hidden Markov Model (HMM) to capture state transitions 
    and persistence, rather than brittle hardcoded thresholds.
    """

    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        # Typical means for India VIX: Low (13), Normal (18), Crisis (28)
        self.emission_means = np.array([[13.0], [18.0], [28.0]])
        
        try:
            from hmmlearn.hmm import GaussianHMM
            self.model = GaussianHMM(n_components=self.n_components, covariance_type="diag")
            self.model.means_ = self.emission_means
            self.model.covars_ = np.array([[2.0], [3.0], [10.0]])
            # Simple transition matrix favoring persistence
            self.model.transmat_ = np.array([
                [0.95, 0.04, 0.01],
                [0.10, 0.85, 0.05],
                [0.05, 0.15, 0.80]
            ])
            self.model.startprob_ = np.array([0.6, 0.3, 0.1])
        except ImportError:
            self.model = None

    def fit(self, historical_vix: np.ndarray):
        """Fit the HMM on historical VIX data."""
        if self.model is not None and len(historical_vix) >= 30:
            self.model.fit(historical_vix.reshape(-1, 1))

    def detect_regime(self, current_vix: float, recent_history: list[float] | None = None) -> VolRegime:
        """Categorize the current VIX level using the HMM or fallback."""
        if self.model is not None and recent_history is not None and len(recent_history) >= 5:
            seq = np.array(recent_history + [current_vix]).reshape(-1, 1)
            try:
                states = self.model.predict(seq)
                state = states[-1]
                
                # Map state to VolRegime based on sorted means
                means = self.model.means_.flatten()
                sorted_indices = np.argsort(means)
                
                if state == sorted_indices[0]:
                    return VolRegime.LOW_VOL
                elif state == sorted_indices[1]:
                    return VolRegime.NORMAL
                else:
                    return VolRegime.CRISIS
            except Exception as e:
                logger.warning(f"HMM prediction failed: {e}. Falling back to thresholds.")

        # Fallback to thresholds if hmmlearn is missing, not enough history, or error
        if current_vix < 15.0:
            return VolRegime.LOW_VOL
        elif current_vix < 22.0:
            return VolRegime.NORMAL
        else:
            return VolRegime.CRISIS
