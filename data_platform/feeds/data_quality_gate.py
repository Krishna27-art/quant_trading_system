"""
Data Quality Gate — Real-Time Tick Validation

Validates every incoming tick before it enters the strategy pipeline:
- Staleness rejection (tick timestamp too far in the past)
- Price outlier rejection (price deviates > N × ATR from last known)
- Bid/ask sanity (bid <= ltp <= ask, non-negative spread)
- Volume sanity (non-negative, not absurdly large)
- Gap detection and logging for post-market audit
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from utils.logger import get_logger

logger = get_logger("data_quality_gate")


class RejectionReason(str, Enum):
    STALE_TICK = "stale_tick"
    PRICE_OUTLIER = "price_outlier"
    NEGATIVE_PRICE = "negative_price"
    BID_ASK_INVALID = "bid_ask_invalid"
    VOLUME_INVALID = "volume_invalid"
    ZERO_PRICE = "zero_price"
    DUPLICATE_TICK = "duplicate_tick"


@dataclass
class QualityVerdict:
    """Result of validating a single tick."""

    accepted: bool
    symbol: str
    reasons: list[RejectionReason] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class GapEvent:
    """Recorded when a price gap exceeds a configurable threshold."""

    symbol: str
    gap_pct: float
    prev_price: float
    new_price: float
    detected_at: float  # epoch


class DataQualityGate:
    """
    Real-time tick validator sitting between the feed layer and strategies.

    Parameters
    ----------
    max_staleness_s
        Ticks older than this many seconds are rejected.
    atr_outlier_multiplier
        A tick whose absolute price change exceeds this × ATR is rejected.
    atr_lookback
        Number of recent price deltas used to compute a rolling ATR proxy.
    gap_alert_pct
        Price gap (%) above which a GapEvent is recorded.
    max_daily_volume
        Hard cap on a single tick's volume field.
    on_reject
        Optional callback(QualityVerdict) for rejected ticks.
    """

    def __init__(
        self,
        max_staleness_s: float = 5.0,
        atr_outlier_multiplier: float = 8.0,
        atr_lookback: int = 100,
        gap_alert_pct: float = 3.0,
        max_daily_volume: int = 500_000_000,
        on_reject: Callable[[QualityVerdict], None] | None = None,
    ) -> None:
        self.max_staleness_s = max_staleness_s
        self.atr_outlier_multiplier = atr_outlier_multiplier
        self.atr_lookback = atr_lookback
        self.gap_alert_pct = gap_alert_pct
        self.max_daily_volume = max_daily_volume
        self.on_reject = on_reject

        # per-symbol state
        self._last_price: dict[str, float] = {}
        self._last_timestamp: dict[str, float] = {}
        self._price_deltas: dict[str, deque[float]] = {}
        self._global_clock: float = 0.0

        # audit trail
        self._gaps: list[GapEvent] = []
        self._rejection_counts: dict[RejectionReason, int] = dict.fromkeys(RejectionReason, 0)
        self._total_checked: int = 0
        self._total_rejected: int = 0

        self._lock = threading.Lock()

    # ── public API ──────────────────────────────────────────

    def validate(
        self,
        symbol: str,
        ltp: float,
        bid: float,
        ask: float,
        volume: int,
        tick_timestamp: float,
    ) -> QualityVerdict:
        """
        Validate a single tick.

        Returns a QualityVerdict. Ticks that fail any check are rejected.
        """

        def parse_exchange_timestamp(raw_ts: float) -> float:
            """
            NSE sends nanosecond timestamps.
            Must divide by 1e9 to get seconds, not 1e3.
            """
            if raw_ts > 1e18:
                return raw_ts / 1e9  # nanoseconds -> seconds
            elif raw_ts > 1e15:
                return raw_ts / 1e6  # microseconds -> seconds
            elif raw_ts > 1e12:
                return raw_ts / 1e3  # milliseconds -> seconds
            elif raw_ts > 1e9:
                return float(raw_ts)  # already seconds
            else:
                return float(raw_ts)  # epoch seconds

        tick_timestamp = parse_exchange_timestamp(tick_timestamp)

        reasons: list[RejectionReason] = []
        details: dict[str, Any] = {}

        with self._lock:
            self._total_checked += 1

            # 1) staleness ────────────────────────────────────
            self._global_clock = max(self._global_clock, tick_timestamp)
            age = self._global_clock - tick_timestamp
            if age > self.max_staleness_s:
                reasons.append(RejectionReason.STALE_TICK)
                details["age_seconds"] = round(age, 3)

            # 2) basic price sanity ───────────────────────────
            if ltp < 0 or bid < 0 or ask < 0:
                reasons.append(RejectionReason.NEGATIVE_PRICE)

            if ltp == 0:
                reasons.append(RejectionReason.ZERO_PRICE)

            # 3) bid/ask validity ─────────────────────────────
            if ltp > 0 and bid > 0 and ask > 0 and bid > ask:
                reasons.append(RejectionReason.BID_ASK_INVALID)
                details["bid"] = bid
                details["ask"] = ask

            # 4) volume ───────────────────────────────────────
            if volume < 0 or volume > self.max_daily_volume:
                reasons.append(RejectionReason.VOLUME_INVALID)
                details["volume"] = volume

            # 5) duplicate detection ──────────────────────────
            last_ts = self._last_timestamp.get(symbol, 0.0)
            if tick_timestamp > 0 and tick_timestamp == last_ts:
                reasons.append(RejectionReason.DUPLICATE_TICK)

            # 6) price outlier (ATR-based) ────────────────────
            prev_price = self._last_price.get(symbol)
            if prev_price is not None and prev_price > 0 and ltp > 0:
                abs_delta = abs(ltp - prev_price)

                # F1.6 Fix: Detect Corporate Action / Split before ATR outlier check
                pct_move = abs_delta / prev_price
                if pct_move > 0.18:
                    ratio = ltp / prev_price
                    ca_type = "UNKNOWN_CA"
                    if abs(ratio - 0.5) < 0.02:
                        ca_type = "SPLIT_2FOR1"
                    elif abs(ratio - 2.0) < 0.05:
                        ca_type = "BONUS_1FOR1"

                    if ca_type != "UNKNOWN_CA":
                        logger.warning(f"Corp action detected: {symbol} {ca_type}")
                        self._last_price[symbol] = ltp
                        self._last_timestamp[symbol] = tick_timestamp
                        if symbol in self._price_deltas:
                            self._price_deltas[symbol].clear()
                        return QualityVerdict(
                            accepted=True,
                            symbol=symbol,
                            reasons=[],
                            details={"corp_action": ca_type},
                        )

                atr = self._compute_atr(symbol)
                if atr > 0 and abs_delta > self.atr_outlier_multiplier * atr:
                    reasons.append(RejectionReason.PRICE_OUTLIER)
                    details["price_delta"] = round(abs_delta, 4)
                    details["atr"] = round(atr, 4)
                    details["multiplier_used"] = round(abs_delta / atr, 2)

                # 7) gap detection (logged, not rejected) ─────
                gap_pct = (abs_delta / prev_price) * 100
                if gap_pct >= self.gap_alert_pct:
                    gap_evt = GapEvent(
                        symbol=symbol,
                        gap_pct=round(gap_pct, 2),
                        prev_price=prev_price,
                        new_price=ltp,
                        detected_at=self._global_clock,
                    )
                    self._gaps.append(gap_evt)
                    logger.warning(
                        "Price gap detected: %s %.2f%% (%.2f → %.2f)",
                        symbol,
                        gap_pct,
                        prev_price,
                        ltp,
                    )

            # ── update state if accepted ─────────────────────
            accepted = len(reasons) == 0
            if accepted and ltp > 0:
                if prev_price is not None and prev_price > 0:
                    delta = abs(ltp - prev_price)
                    deltas = self._price_deltas.setdefault(symbol, deque(maxlen=self.atr_lookback))
                    deltas.append(delta)
                self._last_price[symbol] = ltp
                self._last_timestamp[symbol] = tick_timestamp

            if not accepted:
                self._total_rejected += 1
                for r in reasons:
                    self._rejection_counts[r] += 1

        verdict = QualityVerdict(
            accepted=accepted,
            symbol=symbol,
            reasons=reasons,
            details=details,
        )

        if not accepted:
            logger.debug(
                "Tick REJECTED %s: %s | %s",
                symbol,
                [r.value for r in reasons],
                details,
            )
            if self.on_reject:
                self.on_reject(verdict)

        return verdict

    # ── statistics ──────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_checked": self._total_checked,
                "total_rejected": self._total_rejected,
                "acceptance_rate": (
                    round(
                        (self._total_checked - self._total_rejected) / max(self._total_checked, 1),
                        4,
                    )
                ),
                "rejection_breakdown": {
                    r.value: c for r, c in self._rejection_counts.items() if c > 0
                },
                "gap_events": len(self._gaps),
            }

    @property
    def recent_gaps(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "symbol": g.symbol,
                    "gap_pct": g.gap_pct,
                    "prev_price": g.prev_price,
                    "new_price": g.new_price,
                    "detected_at": g.detected_at,
                }
                for g in self._gaps[-50:]  # last 50
            ]

    def reset_stats(self) -> None:
        """Reset counters (e.g. at start of new trading day)."""
        with self._lock:
            self._total_checked = 0
            self._total_rejected = 0
            self._rejection_counts = dict.fromkeys(RejectionReason, 0)
            self._gaps.clear()

    # ── internals ───────────────────────────────────────────

    def _compute_atr(self, symbol: str) -> float:
        """Rolling ATR proxy = mean of recent absolute price deltas."""
        deltas = self._price_deltas.get(symbol)
        if not deltas or len(deltas) < 5:
            return 0.0
        return sum(deltas) / len(deltas)
