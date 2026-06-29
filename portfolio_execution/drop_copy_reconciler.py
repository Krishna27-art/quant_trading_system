"""
Drop Copy Reconciler

Provides out-of-band trade reconciliation by fetching trade updates
directly from broker APIs (simulating FIX Drop Copy) to detect mismatches
between local OMS state and the exchange's match engine.
"""

import asyncio
import contextlib
from dataclasses import dataclass
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DropCopyConfig:
    """Configuration for drop copy reconciler."""

    broker_name: str = "zerodha"
    poll_interval_seconds: int = 10
    alert_on_mismatch: bool = True
    auto_reconcile: bool = False


@dataclass
class TradeRecord:
    """Standardized representation of a filled trade."""

    order_id: str
    symbol: str
    side: str
    qty: int
    price: float
    timestamp: float
    exchange_id: str | None = None

    def __eq__(self, other: "TradeRecord") -> bool:
        if not isinstance(other, TradeRecord):
            return False
        # Tolerate small price differences if floats don't match exactly
        price_match = abs(self.price - other.price) < 0.01
        return (
            self.order_id == other.order_id
            and self.symbol == other.symbol
            and self.side == other.side
            and self.qty == other.qty
            and price_match
        )


class DropCopyReconciler:
    """
    Simulates a FIX Drop Copy session by polling broker trade endpoints
    and reconciling against local OMS state to prevent double-fills and missing executions.
    """

    def __init__(self, config: DropCopyConfig, oms: Any, broker_api: Any):
        self.config = config
        self.oms = oms
        self.broker_api = broker_api  # Abstracted broker API client
        self.running = False
        self._task: asyncio.Task | None = None

        self.broker_trades_cache: dict[str, TradeRecord] = {}
        self.local_trades_cache: dict[str, TradeRecord] = {}

    async def start(self):
        """Start the background drop copy polling task."""
        if self.running:
            return

        self.running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            f"Drop Copy Reconciler started (interval: {self.config.poll_interval_seconds}s)"
        )

    async def stop(self):
        """Stop the background drop copy polling task."""
        self.running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        logger.info("Drop Copy Reconciler stopped")

    async def _poll_loop(self):
        """Background loop polling broker for trades."""
        while self.running:
            try:
                broker_trades = await self.fetch_broker_trades()
                local_trades = self._get_local_trades()

                self.reconcile(local_trades, broker_trades)

            except Exception as e:
                logger.error(f"Error in Drop Copy polling loop: {e}", exc_info=True)

            await asyncio.sleep(self.config.poll_interval_seconds)

    async def fetch_broker_trades(self) -> list[TradeRecord]:
        """Fetch trades from broker API."""
        try:
            # Assuming broker_api has an async get_trades() method
            # returning a list of raw trade dicts
            if hasattr(self.broker_api, "get_trades"):
                raw_trades = await self.broker_api.get_trades()
                return self._parse_broker_trades(raw_trades)
            else:
                logger.warning("Broker API missing get_trades() method")
                return []
        except Exception as e:
            logger.error(f"Failed to fetch trades from broker: {e}")
            return []

    def _parse_broker_trades(self, raw_trades: list[dict]) -> list[TradeRecord]:
        """Convert raw broker trade dicts to TradeRecord instances."""
        parsed = []
        for rt in raw_trades:
            try:
                # Naive generic mapping, assuming standard fields
                record = TradeRecord(
                    order_id=str(rt.get("order_id", rt.get("orderId", ""))),
                    symbol=str(rt.get("tradingsymbol", rt.get("symbol", ""))),
                    side=str(rt.get("transaction_type", rt.get("side", ""))).upper(),
                    qty=int(rt.get("quantity", rt.get("qty", 0))),
                    price=float(rt.get("average_price", rt.get("price", 0.0))),
                    timestamp=float(rt.get("exchange_timestamp", rt.get("timestamp", 0.0))),
                    exchange_id=str(rt.get("exchange_order_id", "")),
                )
                parsed.append(record)
            except Exception as e:
                logger.error(f"Failed to parse broker trade {rt}: {e}")
        return parsed

    def _get_local_trades(self) -> list[TradeRecord]:
        """Extract filled trades from local OMS state."""
        local_trades = []
        if hasattr(self.oms, "get_all_filled_orders"):
            filled_orders = self.oms.get_all_filled_orders()
            for order in filled_orders:
                record = TradeRecord(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side.name,
                    qty=order.filled_quantity,
                    price=order.average_fill_price,
                    timestamp=(
                        order.last_update_time.timestamp()
                        if hasattr(order.last_update_time, "timestamp")
                        else 0.0
                    ),
                    exchange_id=getattr(order, "exchange_order_id", None),
                )
                local_trades.append(record)
        return local_trades

    def reconcile(self, local_trades: list[TradeRecord], broker_trades: list[TradeRecord]):
        """Compare local and broker trades to find mismatches."""
        local_map = {t.order_id: t for t in local_trades}
        broker_map = {t.order_id: t for t in broker_trades}

        missing_in_broker = []
        missing_in_local = []
        mismatched = []

        # Check local vs broker
        for order_id, local_trade in local_map.items():
            if order_id not in broker_map:
                missing_in_broker.append(local_trade)
            else:
                broker_trade = broker_map[order_id]
                if local_trade != broker_trade:
                    mismatched.append((local_trade, broker_trade))

        # Check broker vs local
        for order_id, broker_trade in broker_map.items():
            if order_id not in local_map:
                missing_in_local.append(broker_trade)

        # Handle unknown fills
        if missing_in_local:
            self.handle_unknown_fills(missing_in_local)

        # Logging & Alerting
        if missing_in_broker or missing_in_local or mismatched:
            logger.warning(
                f"Trade Reconciliation Break: "
                f"{len(missing_in_broker)} local-only, "
                f"{len(missing_in_local)} broker-only, "
                f"{len(mismatched)} mismatched."
            )

            if self.config.alert_on_mismatch:
                self.generate_breaks_report(missing_in_broker, missing_in_local, mismatched)

    def handle_unknown_fills(self, unknown_trades: list[TradeRecord]):
        """Handle fills that exist at the broker but not locally."""
        logger.warning(
            f"Detected {len(unknown_trades)} unknown broker fills! Injecting into OMS..."
        )
        if self.config.auto_reconcile and hasattr(self.oms, "inject_external_fill"):
            for trade in unknown_trades:
                try:
                    self.oms.inject_external_fill(trade)
                    logger.info(f"Successfully injected external fill for order {trade.order_id}")
                except Exception as e:
                    logger.error(f"Failed to inject external fill for {trade.order_id}: {e}")

    def generate_breaks_report(
        self,
        missing_broker: list[TradeRecord],
        missing_local: list[TradeRecord],
        mismatched: list[tuple[TradeRecord, TradeRecord]],
    ):
        """Generate a structured report of trade breaks."""
        # In a real system, this would fire an alert to Slack/PagerDuty
        report = "=== TRADE RECONCILIATION BREAKS ===\n"

        if missing_broker:
            report += "--- Missing at Broker (Phantom Local Fills) ---\n"
            for t in missing_broker:
                report += f"  {t.order_id} | {t.symbol} | {t.side} {t.qty} @ {t.price}\n"

        if missing_local:
            report += "--- Missing Locally (Unrecorded Broker Fills) ---\n"
            for t in missing_local:
                report += f"  {t.order_id} | {t.symbol} | {t.side} {t.qty} @ {t.price}\n"

        if mismatched:
            report += "--- Quantity/Price Mismatches ---\n"
            for loc_t, bro_t in mismatched:
                report += f"  {loc_t.order_id}: Local({loc_t.qty}@{loc_t.price}) != Broker({bro_t.qty}@{bro_t.price})\n"

        logger.warning(report)
