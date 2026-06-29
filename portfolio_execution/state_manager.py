"""
Session State Manager

Tracks all intraday session state:
- Candle accumulation (1m → 5m → 15m)
- VWAP calculation
- EMA computation
- Opening Range (ORH/ORL)
- Previous Day High/Low (PDH/PDL)
- Market state classification

This is the single source of truth for "what is the market doing right now?"
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np

from utils.logger import get_logger

logger = get_logger("state_manager")


class MarketState(str, Enum):
    """Classified market state for strategy selection."""

    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"
    BREAKOUT_UP = "breakout_up"
    BREAKOUT_DOWN = "breakout_down"
    CHOP = "chop"
    PRE_MARKET = "pre_market"
    CLOSED = "closed"


@dataclass
class Candle:
    """OHLCV candle."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body_ratio(self) -> float:
        """Body as percentage of range. High = decisive candle."""
        return self.body / self.range if self.range > 0 else 0.0

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open


@dataclass
class SessionLevels:
    """Key price levels for the current session."""

    pdh: float = 0.0  # Previous Day High
    pdl: float = float("inf")  # Previous Day Low
    pdc: float = 0.0  # Previous Day Close
    orh: float = 0.0  # Opening Range High (first 15m)
    orl: float = float("inf")  # Opening Range Low (first 15m)
    day_high: float = 0.0  # Current day high
    day_low: float = float("inf")  # Current day low
    day_open: float = 0.0  # Day open price
    vwap: float = 0.0  # Volume Weighted Average Price
    ema_20_5m: float = 0.0  # 20-period EMA on 5-minute chart


class SessionStateManager:
    """
    Manages all intraday session state for a single instrument.

    Accumulates 1-minute bars into 5-minute and 15-minute bars,
    computes VWAP, EMAs, and classifies the market state.
    """

    def __init__(self, symbol: str, pdh: float, pdl: float, pdc: float):
        self.symbol = symbol
        self.levels = SessionLevels(pdh=pdh, pdl=pdl, pdc=pdc)

        # Candle storage (bounded to prevent memory leaks)
        self._candles_1m: deque = deque(maxlen=500)  # 500 bars = ~8 hours
        self._candles_5m: deque = deque(maxlen=200)
        self._candles_15m: deque = deque(maxlen=100)

        self._total_minutes: int = 0

        # VWAP accumulator
        self._cumulative_tp_volume: float = 0.0  # Σ(typical_price × volume)
        self._cumulative_volume: float = 0.0  # Σ(volume)

        # EMA state
        self._ema_20_values: deque = deque(maxlen=100)  # Track EMA history
        self._ema_multiplier: float = 2.0 / (20 + 1)  # EMA(20) smoothing

        # Opening range tracking
        self._opening_range_set: bool = False
        self._session_start: datetime | None = None

        # State
        self._current_state: MarketState = MarketState.PRE_MARKET
        self._state_history: list[tuple[datetime, MarketState]] = []

        logger.info(f"[{symbol}] Session initialized | PDH={pdh:.2f} PDL={pdl:.2f} PDC={pdc:.2f}")

    @property
    def state(self) -> MarketState:
        return self._current_state

    @property
    def candles_1m(self) -> deque:
        return self._candles_1m

    @property
    def candles_5m(self) -> deque:
        return self._candles_5m

    @property
    def candles_15m(self) -> deque:
        return self._candles_15m

    def on_candle_1m(self, candle: Candle) -> None:
        """
        Process a new 1-minute candle. This is the main entry point.

        Updates: day high/low, VWAP, 5m/15m aggregation, opening range, EMA, market state.
        """
        self._candles_1m.append(candle)
        self._total_minutes += 1

        # Set session start
        if self._session_start is None:
            self._session_start = candle.timestamp
            self.levels.day_open = candle.open

        # Update day high/low
        self.levels.day_high = max(self.levels.day_high, candle.high)
        self.levels.day_low = min(self.levels.day_low, candle.low)

        # Update VWAP
        self._update_vwap(candle)

        # Aggregate into 5m and 15m
        self._aggregate_candles()

        # Check if opening range should be set (first 15 minutes: 9:15-9:30)
        self._update_opening_range(candle)

        # Update EMA on 5m candles
        if len(self._candles_5m) > 0:
            self._update_ema_20()

        # Classify market state
        self._classify_state()

    def _update_vwap(self, candle: Candle) -> None:
        """Update VWAP with new candle data."""
        typical_price = (candle.high + candle.low + candle.close) / 3.0
        self._cumulative_tp_volume += typical_price * candle.volume
        self._cumulative_volume += candle.volume

        if self._cumulative_volume > 0:
            self.levels.vwap = self._cumulative_tp_volume / self._cumulative_volume

    def _aggregate_candles(self) -> None:
        """Aggregate 1m candles into 5m and 15m bars."""
        # 5-minute aggregation
        if self._total_minutes >= 5 and self._total_minutes % 5 == 0:
            batch = list(self._candles_1m)[-5:]
            agg = Candle(
                timestamp=batch[0].timestamp,
                open=batch[0].open,
                high=max(c.high for c in batch),
                low=min(c.low for c in batch),
                close=batch[-1].close,
                volume=sum(c.volume for c in batch),
            )
            self._candles_5m.append(agg)

        # 15-minute aggregation
        if self._total_minutes >= 15 and self._total_minutes % 15 == 0:
            batch = list(self._candles_1m)[-15:]
            agg = Candle(
                timestamp=batch[0].timestamp,
                open=batch[0].open,
                high=max(c.high for c in batch),
                low=min(c.low for c in batch),
                close=batch[-1].close,
                volume=sum(c.volume for c in batch),
            )
            self._candles_15m.append(agg)

    def _update_opening_range(self, candle: Candle) -> None:
        """
        Set opening range from the first 15 minutes of trading (9:15-9:30).
        """
        if self._opening_range_set:
            return

        # Count minutes since session start
        minutes_elapsed = self._total_minutes

        # Update ORH/ORL with each candle in the first 15 minutes
        if minutes_elapsed <= 15:
            self.levels.orh = max(self.levels.orh, candle.high)
            self.levels.orl = min(self.levels.orl, candle.low)

        # Lock the opening range after 15 minutes
        if minutes_elapsed == 15:
            self._opening_range_set = True
            logger.info(
                f"[{self.symbol}] Opening Range SET | ORH={self.levels.orh:.2f} ORL={self.levels.orl:.2f}"
            )

    def _update_ema_20(self) -> None:
        """Compute 20-period EMA on 5-minute closes."""
        closes = [c.close for c in self._candles_5m]

        if len(closes) < 20:
            # Not enough data — use SMA as seed
            self.levels.ema_20_5m = np.mean(closes)
        else:
            if not self._ema_20_values:
                # First EMA calculation — seed with SMA
                sma = np.mean(closes[:20])
                self._ema_20_values.append(sma)
                # Calculate EMA for remaining periods
                for price in closes[20:]:
                    ema = (
                        price - self._ema_20_values[-1]
                    ) * self._ema_multiplier + self._ema_20_values[-1]
                    self._ema_20_values.append(ema)
            else:
                # Incremental update with latest close
                latest_close = closes[-1]
                ema = (
                    latest_close - self._ema_20_values[-1]
                ) * self._ema_multiplier + self._ema_20_values[-1]
                self._ema_20_values.append(ema)

            self.levels.ema_20_5m = self._ema_20_values[-1]

    def _classify_state(self) -> None:
        """
        Classify the current market state based on price action relative to key levels.

        Uses:
        - Price vs VWAP (above = bullish bias, below = bearish bias)
        - Price vs ORH/ORL (breakout detection)
        - Price vs PDH/PDL (key level interaction)
        - EMA slope (trend direction)
        - Candle structure (chop detection)
        """
        if len(self._candles_1m) < 15:
            self._current_state = MarketState.PRE_MARKET
            return

        if not self._opening_range_set:
            self._current_state = MarketState.PRE_MARKET
            return

        last_price = self._candles_1m[-1].close
        vwap = self.levels.vwap
        orh = self.levels.orh
        orl = self.levels.orl

        # Check for breakout states
        if last_price > orh and last_price > vwap:
            # Above opening range AND VWAP — bullish breakout candidate
            if last_price > self.levels.pdh:
                self._current_state = MarketState.BREAKOUT_UP
            else:
                self._current_state = MarketState.TREND_UP

        elif last_price < orl and last_price < vwap:
            # Below opening range AND VWAP — bearish breakout candidate
            if last_price < self.levels.pdl:
                self._current_state = MarketState.BREAKOUT_DOWN
            else:
                self._current_state = MarketState.TREND_DOWN

        else:
            # Inside opening range or mixed signals
            # Check for chop: price crossing VWAP repeatedly
            if len(self._candles_5m) >= 4:
                recent = list(self._candles_5m)[-4:]
                vwap_crosses = 0
                for i in range(1, len(recent)):
                    prev_above = recent[i - 1].close > vwap
                    curr_above = recent[i].close > vwap
                    if prev_above != curr_above:
                        vwap_crosses += 1

                if vwap_crosses >= 2:
                    self._current_state = MarketState.CHOP
                else:
                    self._current_state = MarketState.RANGE
            else:
                self._current_state = MarketState.RANGE

        self._state_history.append((self._candles_1m[-1].timestamp, self._current_state))

    def is_above_vwap(self) -> bool:
        """Check if current price is above VWAP."""
        if not self._candles_1m:
            return False
        return self._candles_1m[-1].close > self.levels.vwap

    def is_ema_rising(self) -> bool:
        """Check if 5m 20-EMA is rising (slope > 0)."""
        if len(self._ema_20_values) < 3:
            return False
        return self._ema_20_values[-1] > self._ema_20_values[-3]

    def is_ema_falling(self) -> bool:
        """Check if 5m 20-EMA is falling (slope < 0)."""
        if len(self._ema_20_values) < 3:
            return False
        return self._ema_20_values[-1] < self._ema_20_values[-3]

    def get_atr(self, period: int = 14) -> float:
        """Calculate ATR on 5-minute candles for stop-loss sizing."""
        if len(self._candles_5m) < period + 1:
            # Fallback: use 1m candles
            candles = self._candles_1m
            if len(candles) < period + 1:
                return 0.0
        else:
            candles = self._candles_5m

        true_ranges = []
        for i in range(1, min(len(candles), period + 1)):
            high = candles[-i].high
            low = candles[-i].low
            prev_close = candles[-i - 1].close
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)

        return np.mean(true_ranges) if true_ranges else 0.0

    def get_snapshot(self) -> dict:
        """Return a full snapshot of current session state for logging/alerts."""
        last_price = self._candles_1m[-1].close if self._candles_1m else 0.0
        return {
            "symbol": self.symbol,
            "state": self._current_state.value,
            "last_price": last_price,
            "vwap": self.levels.vwap,
            "ema_20_5m": self.levels.ema_20_5m,
            "orh": self.levels.orh,
            "orl": self.levels.orl,
            "pdh": self.levels.pdh,
            "pdl": self.levels.pdl,
            "day_high": self.levels.day_high,
            "day_low": self.levels.day_low,
            "atr": self.get_atr(),
            "above_vwap": self.is_above_vwap(),
            "ema_rising": self.is_ema_rising(),
            "candles_1m_count": len(self._candles_1m),
            "candles_5m_count": len(self._candles_5m),
            "candles_15m_count": len(self._candles_15m),
            "opening_range_set": self._opening_range_set,
        }

    def reset_for_new_session(self, pdh: float, pdl: float, pdc: float) -> None:
        """Reset state for a new trading day."""
        self.levels = SessionLevels(pdh=pdh, pdl=pdl, pdc=pdc)
        self._candles_1m.clear()
        self._candles_5m.clear()
        self._candles_15m.clear()
        self._cumulative_tp_volume = 0.0
        self._cumulative_volume = 0.0
        self._ema_20_values.clear()
        self._opening_range_set = False
        self._session_start = None
        self._current_state = MarketState.PRE_MARKET
        self._state_history.clear()
        logger.info(f"[{self.symbol}] Session RESET for new day | PDH={pdh:.2f} PDL={pdl:.2f}")
