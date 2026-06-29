"""
India Regime Filter

3-state regime classifier for NSE. Stop trading when signal doesn't work.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("regime_filter")

try:
    from hmmlearn import hmm

    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn not available. Regime filter will use simple filter only.")


class RegimeType(str, Enum):
    """Regime types."""

    BULL = "bull"
    BEAR = "bear"
    CHOP = "chop"


@dataclass
class RegimeFilterResults:
    """Results from regime filter analysis."""

    regime_series: pd.Series
    regime_distribution: dict[str, float]
    ic_by_regime: dict[str, float]
    position_multipliers: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "regime_series": self.regime_series.to_dict(),
            "regime_distribution": self.regime_distribution,
            "ic_by_regime": self.ic_by_regime,
            "position_multipliers": self.position_multipliers,
        }


class IndiaRegimeFilter:
    """
    3-state regime classifier for NSE.

    State 0: trending up   — full signal, full size
    State 1: trending down — reduce size or go flat
    State 2: choppy/range  — turn off ML signal entirely
    """

    def __init__(self, n_states: int = 3):
        """
        Initialize regime filter.

        Args:
            n_states: Number of regime states (default 3)
        """
        self.n_states = n_states
        self.logger = logger

        if HMM_AVAILABLE:
            self.model = hmm.GaussianHMM(
                n_components=n_states, covariance_type="diag", n_iter=200, random_state=42
            )
        else:
            self.model = None

        self.X_index = None
        self.label_map = {}

        self.logger.info(f"IndiaRegimeFilter initialized with {n_states} states")

    def fit(self, nifty_close: pd.Series) -> "IndiaRegimeFilter":
        """
        Fit HMM model on NIFTY data.

        Args:
            nifty_close: NIFTY closing prices

        Returns:
            Self for method chaining
        """
        if not HMM_AVAILABLE:
            self.logger.warning("HMM not available, skipping fit")
            return self

        returns = nifty_close.pct_change().dropna()
        vol_20 = returns.rolling(20).std()
        trend_20 = nifty_close.pct_change(20)
        adv_dec = returns.rolling(5).mean()  # proxy for breadth

        X = np.column_stack(
            [returns.fillna(0), vol_20.fillna(vol_20.mean()), trend_20.fillna(0), adv_dec.fillna(0)]
        )

        self.model.fit(X)
        self.X_index = returns.index

        self.logger.info("HMM model fitted")

        return self

    def predict_regime(self, nifty_close: pd.Series) -> pd.Series:
        """
        Predict regime for given NIFTY data.

        Args:
            nifty_close: NIFTY closing prices

        Returns:
            Regime series
        """
        if not HMM_AVAILABLE:
            self.logger.warning("HMM not available, using simple filter")
            return self._simple_regime_filter(nifty_close)

        returns = nifty_close.pct_change().fillna(0)
        vol_20 = returns.rolling(20).std().fillna(returns.std())
        trend_20 = nifty_close.pct_change(20).fillna(0)
        adv_dec = returns.rolling(5).mean().fillna(0)

        X = np.column_stack([returns, vol_20, trend_20, adv_dec])
        states = self.model.predict(X)

        # Label states by mean return (highest = bull, lowest = bear)
        state_returns = {s: returns[states == s].mean() for s in range(self.n_states)}
        sorted_states = sorted(state_returns, key=state_returns.get, reverse=True)

        if self.n_states == 3:
            self.label_map = {
                sorted_states[0]: RegimeType.BULL,
                sorted_states[1]: RegimeType.CHOP,
                sorted_states[2]: RegimeType.BEAR,
            }
        else:
            # Generic labeling for other state counts
            self.label_map = {sorted_states[i]: f"state_{i}" for i in range(self.n_states)}

        regime = pd.Series([self.label_map[s] for s in states], index=nifty_close.index)

        return regime

    def _simple_regime_filter(self, nifty_close: pd.Series) -> pd.Series:
        """
        Simple regime filter based on 20-day return.

        If 20-day NIFTY return is between -1% and +1% → chop.
        Otherwise → bull if positive, bear if negative.

        Args:
            nifty_close: NIFTY closing prices

        Returns:
            Regime series
        """
        trend_20 = nifty_close.pct_change(20).fillna(0)

        regime = pd.Series(index=nifty_close.index, dtype=object)

        # Chop: between -1% and +1%
        chop_mask = (trend_20 >= -0.01) & (trend_20 <= 0.01)
        regime[chop_mask] = RegimeType.CHOP

        # Bull: above +1%
        bull_mask = trend_20 > 0.01
        regime[bull_mask] = RegimeType.BULL

        # Bear: below -1%
        bear_mask = trend_20 < -0.01
        regime[bear_mask] = RegimeType.BEAR

        # Fill any NaN with chop
        regime = regime.fillna(RegimeType.CHOP)

        return regime

    def position_multiplier(self, regime: str) -> float:
        """
        Scale position size by regime.

        Args:
            regime: Regime type

        Returns:
            Position multiplier
        """
        multipliers = {RegimeType.BULL: 1.0, RegimeType.CHOP: 0.0, RegimeType.BEAR: 0.3}

        return multipliers.get(regime, 0.5)

    def get_regime_distribution(self, regime_series: pd.Series) -> dict[str, float]:
        """
        Get distribution of regimes.

        Args:
            regime_series: Regime series

        Returns:
            Dictionary with regime percentages
        """
        distribution = {}
        total = len(regime_series)

        for regime in [RegimeType.BULL, RegimeType.BEAR, RegimeType.CHOP]:
            count = (regime_series == regime).sum()
            distribution[regime] = count / total if total > 0 else 0.0

        return distribution


def validate_regime_ic(
    predictions: pd.DataFrame, returns: pd.DataFrame, regimes: pd.Series
) -> dict[str, float]:
    """
    Validate IC per regime.

    Args:
        predictions: Predictions DataFrame
        returns: Returns DataFrame
        regimes: Regime series

    Returns:
        Dictionary with IC per regime
    """

    ic_by_regime = {}

    for regime in [RegimeType.BULL, RegimeType.CHOP, RegimeType.BEAR]:
        dates = regimes[regimes == regime].index

        if len(dates) < 10:
            ic_by_regime[regime] = 0.0
            continue

        p = predictions.loc[predictions.index.isin(dates)]
        r = returns.loc[returns.index.isin(dates)]

        ic = compute_ic_single(p, r)
        pct = (regimes == regime).mean() * 100

        ic_by_regime[regime] = ic

        print(f"  {regime.value:5}: IC={ic:+.4f}  ({pct:.0f}% of time)")

    return ic_by_regime


def compute_ic_single(pred_df: pd.DataFrame, ret_df: pd.DataFrame) -> float:
    """
    Compute IC for a single subset of data.

    Args:
        pred_df: Predictions DataFrame
        ret_df: Returns DataFrame

    Returns:
        Mean IC
    """
    from scipy.stats import spearmanr

    ics = []

    for date in pred_df.index:
        if date not in ret_df.index:
            continue

        p = pred_df.loc[date].dropna()
        r = ret_df.loc[date].dropna()
        c = p.index.intersection(r.index)

        if len(c) < 5:
            continue

        ic, _ = spearmanr(p[c], r[c])
        if not np.isnan(ic):
            ics.append(ic)

    return np.mean(ics) if ics else 0.0


def apply_regime_filter(
    positions: pd.Series, regime: str, filter_instance: IndiaRegimeFilter | None = None
) -> pd.Series:
    """
    Apply regime filter to positions.

    Args:
        positions: Position series
        regime: Current regime
        filter_instance: Regime filter instance (optional)

    Returns:
        Filtered positions
    """
    if filter_instance:
        multiplier = filter_instance.position_multiplier(regime)
    else:
        # Default multipliers
        multipliers = {RegimeType.BULL: 1.0, RegimeType.CHOP: 0.0, RegimeType.BEAR: 0.3}
        multiplier = multipliers.get(regime, 0.5)

    return positions * multiplier


def get_regime_summary(regime_series: pd.Series, ic_by_regime: dict[str, float]) -> str:
    """
    Get regime summary.

    Args:
        regime_series: Regime series
        ic_by_regime: IC by regime

    Returns:
        Summary string
    """
    summary = []

    distribution = {}
    total = len(regime_series)

    for regime in [RegimeType.BULL, RegimeType.BEAR, RegimeType.CHOP]:
        count = (regime_series == regime).sum()
        distribution[regime] = count / total if total > 0 else 0.0

    summary.append("=== REGIME SUMMARY ===")
    summary.append(
        f"Bull: {distribution[RegimeType.BULL]:.1%} of time, IC={ic_by_regime.get(RegimeType.BULL, 0):.4f}"
    )
    summary.append(
        f"Chop: {distribution[RegimeType.CHOP]:.1%} of time, IC={ic_by_regime.get(RegimeType.CHOP, 0):.4f}"
    )
    summary.append(
        f"Bear: {distribution[RegimeType.BEAR]:.1%} of time, IC={ic_by_regime.get(RegimeType.BEAR, 0):.4f}"
    )

    # Expected result check
    if ic_by_regime.get(RegimeType.BULL, 0) > 0.04 and ic_by_regime.get(RegimeType.CHOP, 0) < 0.02:
        summary.append("\n✓ Regime filter effective: High IC in bull, low IC in chop")
    else:
        summary.append("\n✗ Regime filter may need adjustment")

    return "\n".join(summary)
