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


class VolatilityRegimeDetector:
    """
    Detects the current market volatility regime based on India VIX.
    """

    def __init__(self):
        # Typical historical percentiles/levels for India VIX
        self.threshold_low = 12.0
        self.threshold_high = 20.0
        self.threshold_crisis = 30.0

    def detect_regime(self, current_vix: float) -> VolRegime:
        """Categorize the current VIX level."""
        if current_vix < self.threshold_low:
            return VolRegime.LOW_VOL
        elif current_vix < self.threshold_high:
            return VolRegime.NORMAL
        elif current_vix < self.threshold_crisis:
            return VolRegime.HIGH_VOL
        else:
            return VolRegime.CRISIS
