"""
Alternative Data Signal Engine

Generates alpha signals from non-traditional data sources:
- NLP-based news sentiment
- Options unusual activity / Smart Money flow
- Institutional (FII/DII) cash flows
"""

import numpy as np
import pandas as pd

from portfolio_execution.signals.base import AlphaModel, SignalDirection, SignalNorm
from utils.logger import get_logger

logger = get_logger(__name__)


class NewsSentimentAlpha(AlphaModel):
    """
    NLP-based news sentiment scoring.
    Decays sentiment exponentially over time.
    """

    def __init__(self, decay_halflife_days: float = 3.0, **kwargs):
        super().__init__(
            name="news_sentiment",
            lookback=kwargs.pop("lookback", 20),
            direction=SignalDirection.LONG_SHORT,
            norm=SignalNorm.ZSCORE,
            **kwargs,
        )
        self.decay_halflife_days = decay_halflife_days
        self._lambda = np.log(2) / self.decay_halflife_days if self.decay_halflife_days > 0 else 1.0

    def _compute_raw_signal(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Expects 'sentiment_score' and 'days_since_news' in data/kwargs.
        Signal = sentiment_score * exp(-lambda * days_since_news)
        """
        if "sentiment_score" not in data.columns or "days_since_news" not in data.columns:
            logger.warning("Missing sentiment data columns. Returning empty signal.")
            return pd.Series(index=data.index, dtype=float)

        sentiment = data["sentiment_score"]
        days_since = data["days_since_news"]

        # Exponential decay of sentiment
        decay_factor = np.exp(-self._lambda * days_since)
        raw_signal = sentiment * decay_factor

        return raw_signal


class OptionFlowAlpha(AlphaModel):
    """
    Options unusual activity detection.
    Looks for put/call ratio divergences and unusual volume.
    """

    def __init__(self, volume_zscore_threshold: float = 2.0, **kwargs):
        super().__init__(
            name="option_flow",
            lookback=kwargs.pop("lookback", 20),
            direction=SignalDirection.LONG_SHORT,
            norm=SignalNorm.ZSCORE,
            **kwargs,
        )
        self.volume_zscore_threshold = volume_zscore_threshold

    def _compute_raw_signal(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Expects 'put_call_ratio' and 'option_volume_zscore' in data.
        """
        if "put_call_ratio" not in data.columns or "option_volume_zscore" not in data.columns:
            return pd.Series(index=data.index, dtype=float)

        pcr = data["put_call_ratio"]
        vol_zscore = data["option_volume_zscore"]

        # Smart money often sells puts when bullish (PCR increases but implied vol drops)
        # However, naive PCR (buying puts) means bearish. We use a simple contrarian interpretation
        # if volume is unusually high.

        # Base signal: high PCR = bearish (-1), low PCR = bullish (+1)
        # Normalized PCR around 1.0
        pcr_signal = 1.0 - pcr

        # Amplify signal if there's unusual option volume
        volume_multiplier = np.where(vol_zscore > self.volume_zscore_threshold, vol_zscore, 1.0)

        raw_signal = pcr_signal * volume_multiplier
        return raw_signal


class FIIDIIFlowAlpha(AlphaModel):
    """
    Institutional (FII/DII) cash flow momentum signals.
    Tracks net buying/selling to detect sector rotation.
    """

    def __init__(self, momentum_window: int = 5, **kwargs):
        super().__init__(
            name="fii_dii_flow",
            lookback=kwargs.pop("lookback", 20),
            direction=SignalDirection.LONG_SHORT,
            norm=SignalNorm.ZSCORE,
            **kwargs,
        )
        self.momentum_window = momentum_window

    def _compute_raw_signal(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Expects 'fii_net_flow' and 'dii_net_flow' in data.
        Calculates cumulative flow momentum.
        """
        if "fii_net_flow" not in data.columns or "dii_net_flow" not in data.columns:
            return pd.Series(index=data.index, dtype=float)

        fii_flow = data["fii_net_flow"]
        dii_flow = data["dii_net_flow"]

        # Total institutional flow
        total_flow = fii_flow + dii_flow

        # In a real implementation, this would be a rolling sum over momentum_window
        # assuming 'total_flow' is already the rolling sum for this day's snapshot
        raw_signal = total_flow

        return raw_signal
