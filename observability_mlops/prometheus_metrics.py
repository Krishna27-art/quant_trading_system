"""
Prometheus Metrics for Quant Trading Platform

Collects and exports production metrics:
- Order and fill counters with exchange/strategy labels
- Latency histograms for order lifecycle
- Portfolio PnL and drawdown gauges
- Position count per strategy
- Data feed lag detection
"""

from __future__ import annotations

import time

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from utils.logger import get_logger

logger = get_logger("prometheus_metrics")

# Shared registry so tests can create isolated collectors
_default_registry = CollectorRegistry()


class MetricsCollector:
    """Collects and manages Prometheus metrics for the trading platform."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or _default_registry

        # ── counters ────────────────────────────────────────────
        self.orders_total = Counter(
            "orders_total",
            "Total orders submitted",
            labelnames=["exchange", "strategy", "side", "order_type"],
            registry=self.registry,
        )

        self.fills_total = Counter(
            "fills_total",
            "Total order fills received",
            labelnames=["exchange", "strategy", "side"],
            registry=self.registry,
        )

        # ── histograms ─────────────────────────────────────────
        self.order_latency_seconds = Histogram(
            "order_latency_seconds",
            "Order round-trip latency from submission to fill/ack",
            labelnames=["exchange", "strategy"],
            buckets=(
                0.001,
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1.0,
                2.5,
                5.0,
                10.0,
            ),
            registry=self.registry,
        )

        # ── gauges ──────────────────────────────────────────────
        self.portfolio_pnl_gauge = Gauge(
            "portfolio_pnl_rupees",
            "Current portfolio PnL in INR",
            labelnames=["strategy"],
            registry=self.registry,
        )

        self.drawdown_gauge = Gauge(
            "drawdown_ratio",
            "Current drawdown as a ratio (0-1) from peak equity",
            labelnames=["strategy"],
            registry=self.registry,
        )

        self.position_count = Gauge(
            "position_count",
            "Number of open positions",
            labelnames=["strategy", "side"],
            registry=self.registry,
        )

        self.data_feed_lag_seconds = Gauge(
            "data_feed_lag_seconds",
            "Seconds since last tick received from data feed",
            labelnames=["feed", "symbol"],
            registry=self.registry,
        )

        logger.info("MetricsCollector initialised")

    # ── recording helpers ───────────────────────────────────────

    def record_order(
        self,
        exchange: str,
        strategy: str,
        side: str,
        order_type: str,
    ) -> None:
        """Increment the order counter."""
        self.orders_total.labels(
            exchange=exchange,
            strategy=strategy,
            side=side,
            order_type=order_type,
        ).inc()

    def record_fill(
        self,
        exchange: str,
        strategy: str,
        side: str,
        quantity: int = 1,
    ) -> None:
        """Increment fills counter by *quantity*."""
        self.fills_total.labels(
            exchange=exchange,
            strategy=strategy,
            side=side,
        ).inc(quantity)

    def record_order_latency(
        self,
        exchange: str,
        strategy: str,
        latency_seconds: float,
    ) -> None:
        """Observe an order-lifecycle latency sample."""
        self.order_latency_seconds.labels(
            exchange=exchange,
            strategy=strategy,
        ).observe(latency_seconds)

    def set_portfolio_pnl(self, strategy: str, pnl_inr: float) -> None:
        """Set the current PnL gauge for *strategy*."""
        self.portfolio_pnl_gauge.labels(strategy=strategy).set(pnl_inr)

    def set_drawdown(self, strategy: str, drawdown_ratio: float) -> None:
        """Set current drawdown ratio (0 = no drawdown, 1 = 100 % loss)."""
        clamped = max(0.0, min(1.0, drawdown_ratio))
        self.drawdown_gauge.labels(strategy=strategy).set(clamped)

    def set_position_count(
        self,
        strategy: str,
        side: str,
        count: int,
    ) -> None:
        """Set the number of open positions."""
        self.position_count.labels(strategy=strategy, side=side).set(count)

    def record_data_feed_lag(
        self,
        feed: str,
        symbol: str,
        last_tick_epoch: float,
    ) -> None:
        """Compute and set lag from *last_tick_epoch* to now."""
        lag = max(0.0, time.time() - last_tick_epoch)
        self.data_feed_lag_seconds.labels(feed=feed, symbol=symbol).set(lag)

    # ── export ──────────────────────────────────────────────────

    def generate_metrics(self) -> bytes:
        """Return Prometheus text-exposition format bytes."""
        return generate_latest(self.registry)

    def content_type(self) -> str:
        """Return the correct Content-Type for the /metrics endpoint."""
        return CONTENT_TYPE_LATEST


# ── FastAPI integration ─────────────────────────────────────────


def create_metrics_endpoint(collector: MetricsCollector):
    """Return a FastAPI-compatible async handler for GET /metrics."""
    from fastapi.responses import Response

    async def metrics_endpoint() -> Response:
        return Response(
            content=collector.generate_metrics(),
            media_type=collector.content_type(),
        )

    return metrics_endpoint


# Module-level singleton for convenience import
_singleton: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the module-level MetricsCollector singleton."""
    global _singleton
    if _singleton is None:
        _singleton = MetricsCollector()
    return _singleton
