"""
Point-in-Time Fundamental Data Engine

Generates signals based on fundamental financial data (earnings, revenue, debt)
using strictly point-in-time (PIT) information to prevent look-ahead bias.
Supports dynamic fundamental delay based on SEBI reporting rules.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_execution.signals.base import AlphaModel, SignalDirection, SignalNorm
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FundamentalPITConfig:
    """Configuration for Point-in-Time fundamental models."""

    use_strict_as_of_date: bool = True
    earnings_momentum_lookback: int = 4  # Quarters
    dynamic_delay: bool = True  # Use SEBI reporting rules


class FundamentalPITAlpha(AlphaModel):
    """
    Point-in-time fundamental signals.
    Evaluates Earnings Yield, ROE, and Debt/Equity momentum.
    Incorporates Dynamic Fundamental Delay to reflect reality of Indian reporting timelines.
    """

    def __init__(self, config: FundamentalPITConfig = FundamentalPITConfig(), **kwargs):
        super().__init__(
            name="fundamental_pit",
            lookback=kwargs.pop("lookback", 20),
            direction=SignalDirection.LONG_SHORT,
            norm=SignalNorm.ZSCORE,
            **kwargs,
        )
        self.config = config

    def get_dynamic_delay(self, quarter: int) -> int:
        """
        SEBI LODR regulations for Indian equities:
        Q1, Q2, Q3 (June, Sept, Dec): 45 days from quarter end
        Q4 / Annual (March): 60 days from financial year end
        """
        if pd.isna(quarter):
            return 45  # Default fallback
        if int(quarter) == 4:
            return 60
        return 45

    def _compute_raw_signal(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Expects PIT fundamental columns:
        'earnings_yield', 'roe', 'debt_to_equity_change'
        Data must be aligned as of the signal generation date.
        """
        required_cols = ["earnings_yield", "roe", "debt_to_equity_change"]
        for col in required_cols:
            if col not in data.columns:
                logger.debug(f"Missing fundamental column: {col}. Skipping signal.")
                return pd.Series(index=data.index, dtype=float)

        # Handle dynamic delay if reporting info is present
        # If 'quarter' and 'quarter_end_date' and 'current_date' are provided, we can simulate the delay
        if (
            self.config.dynamic_delay
            and "quarter" in data.columns
            and "quarter_end_date" in data.columns
            and "current_date" in data.columns
        ):
            # Mask out data that hasn't been reported yet based on dynamic delay
            delays = data["quarter"].apply(self.get_dynamic_delay)
            days_since_end = (
                pd.to_datetime(data["current_date"]) - pd.to_datetime(data["quarter_end_date"])
            ).dt.days

            # If days_since_end < delay, data hasn't been officially reported yet!
            mask = days_since_end >= delays

            # Use previous values if not reported yet
            ey = data["earnings_yield"].where(mask, np.nan).ffill().fillna(0)
            roe = data["roe"].where(mask, np.nan).ffill().fillna(0)
            de_change = data["debt_to_equity_change"].where(mask, np.nan).ffill().fillna(0)
        else:
            ey = data["earnings_yield"].fillna(0)
            roe = data["roe"].fillna(0)
            de_change = data["debt_to_equity_change"].fillna(0)

        # We like high Earnings Yield, high ROE, and decreasing Debt/Equity
        raw_signal = ey + roe - de_change
        return raw_signal


class EarningsSurpriseAlpha(AlphaModel):
    """
    Post-Earnings Announcement Drift (PEAD) based on Standardized Unexpected Earnings (SUE).
    """

    def __init__(self, drift_decay_days: int = 20, **kwargs):
        super().__init__(
            name="earnings_surprise",
            lookback=kwargs.pop("lookback", 20),
            direction=SignalDirection.LONG_SHORT,
            norm=SignalNorm.ZSCORE,
            **kwargs,
        )
        self.drift_decay_days = drift_decay_days
        self._lambda = np.log(2) / self.drift_decay_days if self.drift_decay_days > 0 else 1.0

    def _compute_raw_signal(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Expects 'sue_score' and 'days_since_earnings' in data.
        SUE = (Actual EPS - Consensus EPS) / StdDev(Consensus EPS)
        """
        if "sue_score" not in data.columns or "days_since_earnings" not in data.columns:
            return pd.Series(index=data.index, dtype=float)

        sue = data["sue_score"]
        days_since = data["days_since_earnings"]

        # Exponential decay of the PEAD effect
        decay_factor = np.exp(-self._lambda * days_since)
        raw_signal = sue * decay_factor

        return raw_signal
