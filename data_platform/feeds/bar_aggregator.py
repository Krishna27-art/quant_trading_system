"""
Bar Aggregator — In-Memory Sliding Windows & Redis OHLCV Cache

Transforms raw tick streams into real-time OHLCV bars (1m, 5m, 15m, 60m, 1d)
in memory and syncs them to Redis/local cache so inference loops can evaluate
without synchronous HTTP REST calls.
"""
from __future__ import annotations

import json
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any
import pandas as pd

from utils.logger import get_logger
from data_platform.feeds.feed_manager import TickData

logger = get_logger("bar_aggregator")


@dataclass
class Bar:
    symbol: str
    timeframe: str
    start_epoch: float
    open: float
    high: float
    low: float
    close: float
    volume: int
    tick_count: int = 1

    def update(self, price: float, vol: int) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += vol
        self.tick_count += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_epoch": self.start_epoch,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "tick_count": self.tick_count,
        }


class BarAggregator:
    """
    In-memory OHLCV bar builder with sliding window history.
    """
    # Timeframe to duration in seconds
    TF_SECONDS: dict[str, int] = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "60m": 3600,
        "1d": 86400,
        "1wk": 604800,
    }

    def __init__(self, max_bars: int = 500, redis_client: Any | None = None) -> None:
        self.max_bars = max_bars
        self.redis = redis_client
        self._lock = threading.RLock()
        # symbol -> timeframe -> current active Bar
        self._current_bars: dict[str, dict[str, Bar]] = defaultdict(dict)
        # symbol -> timeframe -> deque of completed Bars
        self._history: dict[str, dict[str, deque[Bar]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self.max_bars))
        )
        # Cache readiness flag
        self.cache_ready: bool = False


    def on_tick(self, tick: TickData) -> list[Bar]:
        """
        Process incoming tick, update active bars, roll over completed bars,
        and return a list of any bars that just completed.
        """
        completed_bars: list[Bar] = []
        with self._lock:
            for tf_name, tf_sec in self.TF_SECONDS.items():
                bar_start = (int(tick.timestamp) // tf_sec) * tf_sec
                current_bar = self._current_bars[tick.symbol].get(tf_name)

                if current_bar is None:
                    current_bar = Bar(
                        symbol=tick.symbol,
                        timeframe=tf_name,
                        start_epoch=bar_start,
                        open=tick.ltp,
                        high=tick.ltp,
                        low=tick.ltp,
                        close=tick.ltp,
                        volume=tick.volume,
                    )
                    self._current_bars[tick.symbol][tf_name] = current_bar
                elif bar_start > current_bar.start_epoch:
                    self._history[tick.symbol][tf_name].append(current_bar)
                    completed_bars.append(current_bar)
                    self._publish_to_redis(current_bar)

                    current_bar = Bar(
                        symbol=tick.symbol,
                        timeframe=tf_name,
                        start_epoch=bar_start,
                        open=tick.ltp,
                        high=tick.ltp,
                        low=tick.ltp,
                        close=tick.ltp,
                        volume=tick.volume,
                    )
                    self._current_bars[tick.symbol][tf_name] = current_bar
                else:
                    current_bar.update(tick.ltp, tick.volume)

                self._cache_latest(current_bar)

        return completed_bars

    def _publish_to_redis(self, bar: Bar) -> None:
        if not self.redis:
            return
        try:
            key = f"ohlcv:completed:{bar.symbol}:{bar.timeframe}"
            self.redis.lpush(key, json.dumps(bar.to_dict()))
            self.redis.ltrim(key, 0, self.max_bars - 1)
        except Exception as exc:
            logger.debug(f"Redis publish failed for {bar.symbol}: {exc}")

    def _cache_latest(self, bar: Bar) -> None:
        if not self.redis:
            return
        try:
            key = f"ohlcv:latest:{bar.symbol}:{bar.timeframe}"
            self.redis.set(key, json.dumps(bar.to_dict()))
        except Exception as exc:
            logger.debug(f"Redis cache latest failed: {exc}")

    def get_bars_df(self, symbol: str, timeframe: str, include_current: bool = True) -> pd.DataFrame:
        """
        Return a pandas DataFrame of historical bars + current active bar,
        ready for CanonicalFeatureBuilder.
        """
        with self._lock:
            history = list(self._history[symbol][timeframe])
            if include_current and timeframe in self._current_bars[symbol]:
                history.append(self._current_bars[symbol][timeframe])

        if not history:
            return pd.DataFrame()

        data = [b.to_dict() for b in history]
        df = pd.DataFrame(data)
        # Sort by start_epoch to ensure monotonic time series
        df.sort_values("start_epoch", inplace=True)
        df["timestamp"] = pd.to_datetime(df["start_epoch"], unit="s", utc=True)
        try:
            df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")
        except TypeError:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")
        df.set_index("timestamp", inplace=True)
        return df[["open", "high", "low", "close", "volume"]]

    def get_cached_ohlcv(self, symbol: str, timeframe: str, min_bars: int = 50) -> pd.DataFrame:
        """
        Helper for live prediction loop: gets bars from in-memory cache,
        or falls back to Redis if in-memory history is sparse.
        """
        df = self.get_bars_df(symbol, timeframe)
        if len(df) >= min_bars:
            return df
        
        if self.redis:
            try:
                key = f"ohlcv:completed:{symbol}:{timeframe}"
                raw_bars = self.redis.lrange(key, 0, self.max_bars - 1)
                if raw_bars:
                    data = [json.loads(b) for b in reversed(raw_bars)]
                    df_redis = pd.DataFrame(data)
                    df_redis.sort_values("start_epoch", inplace=True)
                    df_redis["timestamp"] = pd.to_datetime(df_redis["start_epoch"], unit="s", utc=True)
                    try:
                        df_redis["timestamp"] = df_redis["timestamp"].dt.tz_convert("Asia/Kolkata")
                    except TypeError:
                        df_redis["timestamp"] = df_redis["timestamp"].dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")
                    df_redis.set_index("timestamp", inplace=True)
                    return df_redis[["open", "high", "low", "close", "volume"]]
            except Exception as exc:
                logger.debug(f"Redis lrange failed for {symbol}: {exc}")

        return df


_default_aggregator: BarAggregator | None = None
_aggregator_lock = threading.Lock()


def get_default_aggregator(redis_client: Any | None = None) -> BarAggregator:
    global _default_aggregator
    with _aggregator_lock:
        if _default_aggregator is None:
            _default_aggregator = BarAggregator(redis_client=redis_client)
        elif redis_client is not None and _default_aggregator.redis is None:
            _default_aggregator.redis = redis_client
    return _default_aggregator


def fetch_cached_bars(symbol: str, timeframe: str, min_bars: int = 50) -> pd.DataFrame:
    """Module-level helper to fetch cached bars from the default aggregator."""
    return get_default_aggregator().get_cached_ohlcv(symbol, timeframe, min_bars=min_bars)

