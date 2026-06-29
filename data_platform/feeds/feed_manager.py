"""
Feed Manager — Redundant Data Feed with Automatic Failover

Feed hierarchy:
  1. Primary   → WebSocket (lowest latency)
  2. Fallback  → REST polling (reliable but slower)
  3. Emergency → Last-known cached ticks (stale data, logged)

Each feed is health-monitored.  When the active feed fails consecutively
or its staleness exceeds a threshold, the manager promotes the next tier
and fires an alert.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from utils.logger import get_logger

logger = get_logger("feed_manager")


class FeedTier(Enum):
    PRIMARY = auto()  # WebSocket
    FALLBACK = auto()  # REST polling
    EMERGENCY = auto()  # Cached last-known


class FeedState(str, Enum):
    CONNECTED = "connected"
    DEGRADED = "degraded"
    DISCONNECTED = "disconnected"


@dataclass
class TickData:
    """Normalised tick coming out of any feed."""

    symbol: str
    ltp: float
    bid: float
    ask: float
    volume: int
    timestamp: float  # exchange epoch
    received_at: float = 0.0  # local epoch when we received it
    sequence_num: int = 0  # deterministic monotonic sequence ID
    feed_tier: FeedTier = FeedTier.PRIMARY

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "ltp": self.ltp,
            "bid": self.bid,
            "ask": self.ask,
            "volume": self.volume,
            "timestamp": self.timestamp,
            "received_at": self.received_at,
            "feed_tier": self.feed_tier.name,
        }


@dataclass
class FeedHealthStats:
    """Rolling stats for one feed tier."""

    consecutive_failures: int = 0
    total_ticks: int = 0
    total_errors: int = 0
    last_tick_epoch: float = 0.0
    last_error_epoch: float = 0.0
    last_error_msg: str = ""

    @property
    def seconds_since_last_tick(self) -> float:
        if self.last_tick_epoch <= 0:
            return float("inf")
        return time.time() - self.last_tick_epoch


class FeedManager:
    """
    Manages redundant market data feeds with automatic failover.

    Parameters
    ----------
    ws_connect_fn
        Async callable that opens a WebSocket and yields TickData via callback.
    rest_poll_fn
        Sync/async callable(symbol) -> TickData  for REST fallback.
    staleness_threshold_s
        Seconds after which a feed is considered stale.
    max_consecutive_failures
        Failures before promoting next tier.
    on_tick
        User callback invoked for every validated tick.
    on_failover
        Callback(old_tier, new_tier) fired on tier change.
    """

    def __init__(
        self,
        ws_connect_fn: Callable[..., Any] | None = None,
        rest_poll_fn: Callable[..., Any] | None = None,
        staleness_threshold_s: float = 10.0,
        max_consecutive_failures: int = 5,
        rest_poll_interval_s: float = 1.0,
        on_tick: Callable[[TickData], None] | None = None,
        on_failover: Callable[[FeedTier, FeedTier], None] | None = None,
    ) -> None:
        self._ws_connect = ws_connect_fn
        self._rest_poll = rest_poll_fn
        self.staleness_threshold_s = staleness_threshold_s
        self.max_consecutive_failures = max_consecutive_failures
        self.rest_poll_interval_s = rest_poll_interval_s
        self.on_tick = on_tick
        self.on_failover = on_failover

        self._active_tier: FeedTier = FeedTier.PRIMARY
        self._health: dict[FeedTier, FeedHealthStats] = {
            tier: FeedHealthStats() for tier in FeedTier
        }
        self._cache: dict[str, TickData] = {}  # symbol → last tick
        self._lock = threading.Lock()
        self._running = False
        self._symbols: list[str] = []

        # Add non-blocking tick queue
        self._tick_queue = queue.Queue(maxsize=10000)
        self._dropped_ticks = 0
        self._consumer_running = True
        self._consumer_thread = threading.Thread(
            target=self._consume_ticks, daemon=True, name="tick_consumer"
        )
        self._consumer_thread.start()

    # ── public API ──────────────────────────────────────────

    @property
    def active_tier(self) -> FeedTier:
        return self._active_tier

    @property
    def health(self) -> dict[str, Any]:
        with self._lock:
            return {
                tier.name: {
                    "consecutive_failures": h.consecutive_failures,
                    "total_ticks": h.total_ticks,
                    "total_errors": h.total_errors,
                    "seconds_since_last_tick": round(h.seconds_since_last_tick, 2),
                    "last_error": h.last_error_msg,
                }
                for tier, h in self._health.items()
            }

    def get_cached_tick(self, symbol: str) -> TickData | None:
        """Return the last-known tick for *symbol* from the cache."""
        with self._lock:
            return self._cache.get(symbol)

    def subscribe(self, symbols: list[str]) -> None:
        """Set the list of symbols to subscribe to."""
        self._symbols = list(symbols)
        logger.info("Subscribed to %d symbols", len(self._symbols))

    async def start(self) -> None:
        """Start the feed manager event loop (runs until stop())."""
        self._running = True
        logger.info("FeedManager starting on tier %s", self._active_tier.name)
        while self._running:
            try:
                if self._active_tier == FeedTier.PRIMARY:
                    await self._run_websocket()
                elif self._active_tier == FeedTier.FALLBACK:
                    await self._run_rest_polling()
                else:
                    await self._run_emergency_cache()
            except Exception as exc:
                self._record_error(self._active_tier, str(exc))
                logger.error("Feed tier %s error: %s", self._active_tier.name, exc)
                self._maybe_failover()
                await asyncio.sleep(1)

    def stop(self) -> None:
        self._running = False
        self._consumer_running = False
        logger.info("FeedManager stopped")

    # ── WebSocket tier ──────────────────────────────────────

    async def _run_websocket(self) -> None:
        if self._ws_connect is None:
            logger.warning("No WebSocket connector configured, falling back")
            self._promote(FeedTier.FALLBACK)
            return
        try:
            await self._ws_connect(
                symbols=self._symbols,
                on_tick=self._on_ws_tick,
            )
        except Exception as exc:
            raise RuntimeError(f"WebSocket disconnected: {exc}") from exc

    def _on_ws_tick(self, tick: TickData) -> None:
        """
        WebSocket callback — must be NON-BLOCKING.
        Just puts on queue and returns immediately.
        """
        tick.received_at = time.time()
        tick.feed_tier = FeedTier.PRIMARY
        try:
            self._tick_queue.put_nowait(tick)
        except queue.Full:
            self._dropped_ticks += 1

    def _consume_ticks(self) -> None:
        """
        Separate thread consumes ticks from queue.
        WebSocket thread is never blocked.
        """
        while self._consumer_running:
            try:
                tick = self._tick_queue.get(timeout=0.1)
                self._accept_tick(tick, tick.feed_tier)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Tick processing error: {e}")

    # ── REST polling tier ───────────────────────────────────

    async def _run_rest_polling(self) -> None:
        if self._rest_poll is None:
            logger.warning("No REST poller configured, falling back to cache")
            self._promote(FeedTier.EMERGENCY)
            return
        while self._running and self._active_tier == FeedTier.FALLBACK:
            for symbol in self._symbols:
                try:
                    tick = self._rest_poll(symbol)
                    if tick is not None:
                        tick.received_at = time.time()
                        tick.feed_tier = FeedTier.FALLBACK
                        self._accept_tick(tick, FeedTier.FALLBACK)
                except Exception as exc:
                    self._record_error(FeedTier.FALLBACK, str(exc))
            self._check_staleness(FeedTier.FALLBACK)
            await asyncio.sleep(self.rest_poll_interval_s)

    # ── Emergency cache tier ────────────────────────────────

    async def _run_emergency_cache(self) -> None:
        logger.warning("Running on EMERGENCY cached ticks — data is stale!")
        while self._running and self._active_tier == FeedTier.EMERGENCY:
            # Redesigned: Do NOT call self.on_tick(tick) under any circumstances.
            # Only publish FEED_STALE to Redis to make sure strategy halts.
            if getattr(self, "redis_client", None):
                try:
                    self.redis_client.publish(
                        "channel:feed_stale",
                        json.dumps(
                            {
                                "reason": "emergency_cache_active",
                                "msg": "Running on emergency cache. Signal generation halted.",
                            }
                        ),
                    )
                except Exception as e:
                    logger.error(f"Failed to publish FEED_STALE from emergency cache: {e}")
            await asyncio.sleep(self.rest_poll_interval_s * 5)
            # Periodically try to recover upper tiers
            self._attempt_recovery()

    # ── internal helpers ────────────────────────────────────

    def _accept_tick(self, tick: TickData, tier: FeedTier) -> None:
        with self._lock:
            stats = self._health[tier]
            stats.total_ticks += 1
            stats.consecutive_failures = 0
            stats.last_tick_epoch = time.time()
            self._cache[tick.symbol] = tick

        # Enforce exchange timestamp staleness gate (>30s)
        exchange_epoch = tick.timestamp
        if exchange_epoch > 1e11:  # milliseconds
            exchange_epoch = exchange_epoch / 1000.0

        staleness = time.time() - exchange_epoch
        if staleness > 30.0:
            logger.error(
                f"Tick staleness gate triggered for {tick.symbol}: staleness={staleness:.2f}s (>30s). Tick dropped."
            )
            if getattr(self, "redis_client", None):
                try:
                    self.redis_client.publish(
                        "channel:feed_stale",
                        json.dumps(
                            {
                                "symbol": tick.symbol,
                                "timestamp": tick.timestamp,
                                "staleness": staleness,
                                "reason": f"Tick staleness {staleness:.2f}s exceeds 30s",
                            }
                        ),
                    )
                except Exception as e:
                    logger.error(f"Failed to publish FEED_STALE to Redis: {e}")
            return

        if self.on_tick:
            self.on_tick(tick)

    def _record_error(self, tier: FeedTier, msg: str) -> None:
        with self._lock:
            stats = self._health[tier]
            stats.consecutive_failures += 1
            stats.total_errors += 1
            stats.last_error_epoch = time.time()
            stats.last_error_msg = msg

    def _check_staleness(self, tier: FeedTier) -> None:
        with self._lock:
            stats = self._health[tier]
        if stats.seconds_since_last_tick > self.staleness_threshold_s:
            logger.warning(
                "Tier %s stale (%.1fs since last tick)",
                tier.name,
                stats.seconds_since_last_tick,
            )
            self._maybe_failover()

    def _maybe_failover(self) -> None:
        with self._lock:
            stats = self._health[self._active_tier]
            needs_failover = (
                stats.consecutive_failures >= self.max_consecutive_failures
                or stats.seconds_since_last_tick > self.staleness_threshold_s
            )
        if needs_failover:
            next_tier = self._next_tier(self._active_tier)
            if next_tier != self._active_tier:
                self._promote(next_tier)

    def _promote(self, new_tier: FeedTier) -> None:
        old = self._active_tier
        self._active_tier = new_tier
        logger.warning("FAILOVER: %s → %s", old.name, new_tier.name)
        if self.on_failover:
            self.on_failover(old, new_tier)

    def _attempt_recovery(self) -> None:
        """Try to recover the primary or fallback tier."""
        if self._active_tier == FeedTier.EMERGENCY:
            with self._lock:
                self._health[FeedTier.PRIMARY].consecutive_failures = 0
            self._promote(FeedTier.PRIMARY)

    @staticmethod
    def _next_tier(current: FeedTier) -> FeedTier:
        order = [FeedTier.PRIMARY, FeedTier.FALLBACK, FeedTier.EMERGENCY]
        idx = order.index(current)
        return order[min(idx + 1, len(order) - 1)]
