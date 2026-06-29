"""
Intraday Trading Playbook Setups — Indian Markets (NSE/BSE)

This module implements rule-based alpha models for:
1. OpeningRangeBreakout (ORB): Trades breakouts of the 15-minute or 30-minute range.
2. VwapPullback: Trades pullback entries towards the Volume-Weighted Average Price.
3. PdhPdlBreakout: Trades breakouts of the Previous Day High (PDH) or Previous Day Low (PDL).
"""

import pandas as pd

from portfolio_execution.signals.base import AlphaModel, SignalDirection, SignalNorm
from utils.logger import get_logger

logger = get_logger(__name__)


class OpeningRangeBreakout(AlphaModel):
    """
    Opening Range Breakout (ORB) Alpha Model.
    Generates signals when current price breaks above the opening range high (long)
    or below the opening range low (short).

    Parameters
    ----------
    orb_minutes : int
        Duration of the opening range (default 15 minutes).
    atr_filter_mult : float
        ATR multiple required as a minimum breakout distance to filter noise.
    """

    def __init__(
        self,
        orb_minutes: int = 15,
        atr_filter_mult: float = 0.5,
        norm: SignalNorm = SignalNorm.ZSCORE,
        **kwargs,
    ):
        super().__init__(
            name="orb_breakout",
            lookback=75,  # Needs intraday bars
            norm=norm,
            direction=SignalDirection.LONG_SHORT,
            **kwargs,
        )
        self.orb_minutes = orb_minutes
        self.atr_filter_mult = atr_filter_mult

    def _compute_raw_signal(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Expects `data` containing multi-index columns:
        symbols -> 'open', 'high', 'low', 'close', 'volume', 'atr'
        or wide DataFrames passed via kwargs.
        """
        # In a real environment, the orchestrator passes the session state
        # from state_manager which contains ORH/ORL.
        # Fallback to local calculation if not provided.
        state_mgr = kwargs.get("state_manager")
        if state_mgr is not None:
            signals = {}
            for symbol in state_mgr.symbols:
                orh = state_mgr.get_opening_range_high(symbol)
                orl = state_mgr.get_opening_range_low(symbol)
                curr_price = state_mgr.get_current_price(symbol)
                atr = state_mgr.get_atr(symbol) or 0.0

                if orh is None or orl is None or curr_price is None:
                    continue

                filter_dist = self.atr_filter_mult * atr
                if curr_price > orh + filter_dist:
                    signals[symbol] = 1.0
                elif curr_price < orl - filter_dist:
                    signals[symbol] = -1.0
                else:
                    signals[symbol] = 0.0

            return pd.Series(signals, dtype=float)

        # Fallback raw data calculation:
        # Expects 'close' prices
        if data.empty:
            return pd.Series(dtype=float)

        # Assuming data is a wide close price DataFrame
        # Simple cross-sectional momentum as fallback breakout
        ret = data.pct_change(self.orb_minutes).iloc[-1]
        return ret


class VwapPullback(AlphaModel):
    """
    VWAP Pullback Alpha Model.
    Generates long signals when a stock is in an uptrend (price > EMA) and pulls back
    towards VWAP (without breaking below it). Generates short signals for the inverse.

    Parameters
    ----------
    ema_trend_window : int
        Lookback for trend determination.
    pullback_threshold_pct : float
        Divergence limit from VWAP to trigger entry.
    """

    def __init__(
        self,
        ema_trend_window: int = 50,
        pullback_threshold_pct: float = 0.005,  # 0.5%
        norm: SignalNorm = SignalNorm.ZSCORE,
        **kwargs,
    ):
        super().__init__(
            name="vwap_pullback",
            lookback=ema_trend_window,
            norm=norm,
            direction=SignalDirection.LONG_SHORT,
            **kwargs,
        )
        self.ema_trend_window = ema_trend_window
        self.pullback_threshold_pct = pullback_threshold_pct

    def _compute_raw_signal(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        state_mgr = kwargs.get("state_manager")
        if state_mgr is not None:
            signals = {}
            for symbol in state_mgr.symbols:
                vwap = state_mgr.get_vwap(symbol)
                ema = state_mgr.get_ema(symbol, self.ema_trend_window)
                curr_price = state_mgr.get_current_price(symbol)

                if vwap is None or ema is None or curr_price is None:
                    continue

                # Up-trend pullback
                if curr_price > ema and curr_price > vwap:
                    dist = (curr_price - vwap) / vwap
                    if dist <= self.pullback_threshold_pct:
                        # Near VWAP
                        signals[symbol] = 1.0
                    else:
                        signals[symbol] = 0.0
                # Down-trend pullback
                elif curr_price < ema and curr_price < vwap:
                    dist = (vwap - curr_price) / vwap
                    if dist <= self.pullback_threshold_pct:
                        # Near VWAP
                        signals[symbol] = -1.0
                    else:
                        signals[symbol] = 0.0
                else:
                    signals[symbol] = 0.0

            return pd.Series(signals, dtype=float)

        return pd.Series(dtype=float)


class PdhPdlBreakout(AlphaModel):
    """
    Previous Day High/Low Breakout (PDH/PDL) Alpha Model.
    Triggers buy orders when the price breaks above the previous day's high,
    and sell/short orders when it breaks below the previous day's low.
    """

    def __init__(
        self,
        norm: SignalNorm = SignalNorm.ZSCORE,
        **kwargs,
    ):
        super().__init__(
            name="pdh_pdl_breakout",
            lookback=2,
            norm=norm,
            direction=SignalDirection.LONG_SHORT,
            **kwargs,
        )

    def _compute_raw_signal(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        state_mgr = kwargs.get("state_manager")
        if state_mgr is not None:
            signals = {}
            for symbol in state_mgr.symbols:
                pdh = state_mgr.get_pdh(symbol)
                pdl = state_mgr.get_pdl(symbol)
                curr_price = state_mgr.get_current_price(symbol)

                if pdh is None or pdl is None or curr_price is None:
                    continue

                if curr_price > pdh:
                    signals[symbol] = 1.0
                elif curr_price < pdl:
                    signals[symbol] = -1.0
                else:
                    signals[symbol] = 0.0

            return pd.Series(signals, dtype=float)

        return pd.Series(dtype=float)
