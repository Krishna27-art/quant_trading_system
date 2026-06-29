"""
Execution Management System (EMS)

The HOW layer — handles order routing, broker communication, and fill management.

Responsibilities:
- Smart order routing between multiple brokers
- Order submission with timeout and retry logic
- Fill event processing and forwarding to OMS
- Broker heartbeat monitoring
- Order cancellation
- Broker connection failover
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol

from portfolio_execution.oms import ManagedOrder, OrderSide, OrderStatus, OrderType
from utils.logger import get_logger
from utils.time_utils import now_ist

logger = get_logger("ems")


class BrokerStatus(str, Enum):
    """Broker connection status."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    DEGRADED = "degraded"  # High latency or partial failures
    RECONNECTING = "reconnecting"


class RoutingStrategy(str, Enum):
    """Order routing strategies."""

    PRIMARY = "primary"  # Always use primary broker
    BEST_PRICE = "best_price"  # Route to broker with best price
    ROUND_ROBIN = "round_robin"
    FAILOVER = "failover"  # Use backup if primary fails


@dataclass
class BrokerFill:
    """Fill event from a broker."""

    broker_order_id: str
    our_order_id: str
    symbol: str
    side: str
    filled_quantity: int
    fill_price: float
    timestamp: datetime
    exchange: str = "NSE"
    fees: float = 0.0


@dataclass
class BrokerState:
    """Tracked state for a broker connection."""

    name: str
    status: BrokerStatus = BrokerStatus.DISCONNECTED
    last_heartbeat: datetime | None = None
    latency_ms: float = 0.0
    orders_sent: int = 0
    orders_failed: int = 0
    consecutive_failures: int = 0
    max_consecutive_failures: int = 3


class BrokerAdapter(Protocol):
    """Protocol that all broker adapters must implement."""

    def connect(self) -> bool: ...
    def disconnect(self) -> None: ...
    def submit_order(
        self, symbol: str, side: str, quantity: int, price: float, order_type: str, **kwargs
    ) -> str | None: ...
    def cancel_order(self, broker_order_id: str) -> bool: ...
    def get_order_status(self, broker_order_id: str) -> dict | None: ...
    def heartbeat(self) -> bool: ...


class ExecutionManagementSystem:
    """
    Execution Management System.

    Routes orders to brokers, handles fills, manages broker connections.
    Decoupled from OMS — receives validated orders and handles execution.
    """

    def __init__(
        self,
        routing_strategy: RoutingStrategy = RoutingStrategy.FAILOVER,
        order_timeout_seconds: float = 5.0,
        max_retries: int = 3,
    ):
        self._routing_strategy = routing_strategy
        self._order_timeout = order_timeout_seconds
        self._max_retries = max_retries

        # Smart Order Router
        from portfolio_execution.execution.routing.smart_router import SmartOrderRouter

        self.smart_router = SmartOrderRouter()

        # Broker connections
        self._brokers: dict[str, BrokerState] = {}
        self._adapters: dict[str, object] = {}  # BrokerAdapter instances
        self._primary_broker: str = ""

        # Order tracking: our_order_id → broker_order_id
        self._order_map: dict[str, str] = {}
        self._broker_for_order: dict[str, str] = {}  # order_id → broker_name

        # Fill callback (set by orchestrator to forward fills to OMS)
        self._on_fill: Callable | None = None

        # Metrics
        self._total_orders_submitted: int = 0
        self._total_fills_received: int = 0
        self._total_rejections: int = 0

        logger.info(f"EMS initialized | Routing={routing_strategy.value}")

    def register_broker(self, name: str, adapter: object, is_primary: bool = False) -> None:
        """Register a broker adapter."""
        self._brokers[name] = BrokerState(name=name)
        self._adapters[name] = adapter
        if is_primary or not self._primary_broker:
            self._primary_broker = name

        # Register with SmartOrderRouter
        self.smart_router.register_broker(name, "generic")
        logger.info(f"Broker registered: {name} (primary={is_primary})")

    def set_on_fill(self, callback: Callable) -> None:
        """Set callback for fill events."""
        self._on_fill = callback

    async def connect_all(self) -> dict[str, bool]:
        """Connect to all registered brokers."""
        results = {}
        for name, adapter in self._adapters.items():
            try:
                success = await adapter.connect()
                if success:
                    self._brokers[name].status = BrokerStatus.CONNECTED
                    self._brokers[name].last_heartbeat = now_ist()
                else:
                    self._brokers[name].status = BrokerStatus.DISCONNECTED
                results[name] = success
            except Exception as e:
                logger.error(f"Failed to connect to {name}: {e}")
                self._brokers[name].status = BrokerStatus.DISCONNECTED
                results[name] = False
        return results

    async def disconnect_all(self) -> None:
        """Disconnect from all brokers."""
        for name, adapter in self._adapters.items():
            try:
                await adapter.disconnect()
                self._brokers[name].status = BrokerStatus.DISCONNECTED
            except Exception as e:
                logger.error(f"Error disconnecting {name}: {e}")

    async def submit_order(self, order: ManagedOrder) -> bool:
        """
        Submit a validated order to a broker.

        Returns True if the order was accepted by the broker.
        """
        if order.status != OrderStatus.VALIDATED:
            logger.error(f"Cannot submit order {order.order_id}: status is {order.status}")
            return False

        if getattr(self, "redis_client", None) and getattr(self, "use_redis_streams", False):
            # Update status to SENT first, which triggers synchronous WAL write
            order.update_status(OrderStatus.SENT, "About to publish to stream:orders:pending")

            # Publish to stream:orders:pending
            order_data = {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side.name,
                "quantity": str(order.quantity),
                "price": str(order.price or 0.0),
                "order_type": order.order_type.name,
                "stop_price": str(order.stop_price or 0.0),
                "broker_name": "zerodha" if self._adapters.get("zerodha") else "paper",
            }
            try:
                self.redis_client.xadd("stream:orders:pending", order_data)
                logger.info(
                    f"[{order.symbol}] Order {order.order_id} published to Redis Stream pending queue"
                )
                return True
            except Exception as e:
                logger.error(f"Failed to publish order to Redis stream: {e}")
                order.update_status(OrderStatus.REJECTED, f"Stream publish failed: {e}")
                return False

        # Select broker via Smart Order Router
        from portfolio_execution.execution.routing.smart_router import (
            RoutingRequest,
        )
        from portfolio_execution.execution.routing.smart_router import (
            RoutingStrategy as SORStrategy,
        )

        req = RoutingRequest(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            order_type=order.order_type.value,
            price=order.price,
            strategy=SORStrategy.BALANCED,
        )
        decision = self.smart_router.route_order(req)
        broker_name = decision.selected_broker

        if not broker_name:
            order.update_status(OrderStatus.REJECTED, "No connected broker available")
            self._total_rejections += 1
            return False

        adapter = self._adapters[broker_name]
        broker_state = self._brokers[broker_name]

        # Submit with retry logic
        for attempt in range(1, self._max_retries + 1):
            try:
                order.update_status(
                    OrderStatus.SENT, f"Submitting to {broker_name} (attempt {attempt})"
                )
                start_time = time.monotonic()
                # order is already populated, place it
                broker_order_id = await adapter.place_order(order)
                latency = (time.monotonic() - start_time) * 1000
                broker_state.latency_ms = latency

                if broker_order_id:
                    order.broker_order_id = broker_order_id
                    order.update_status(
                        OrderStatus.ACKNOWLEDGED,
                        f"Acknowledged by {broker_name} with broker ID: {broker_order_id}",
                    )
                    self._order_map[order.order_id] = broker_order_id
                    self._broker_for_order[order.order_id] = broker_name
                    broker_state.orders_sent += 1

                    broker_state.consecutive_failures = 0
                    self._total_orders_submitted += 1

                    logger.info(
                        f"[{order.symbol}] Order SENT to {broker_name} | "
                        f"BrokerID={broker_order_id} | Latency={latency:.1f}ms"
                    )

                    # For paper broker, process fill immediately
                    if "PaperBrokerAdapter" in adapter.__class__.__name__:
                        await self._process_paper_fill(order, adapter, broker_order_id)

                    return True
                else:
                    logger.warning(
                        f"[{order.symbol}] Broker returned None (attempt {attempt}/{self._max_retries})"
                    )

            except Exception as e:
                logger.error(
                    f"[{order.symbol}] Submit failed (attempt {attempt}/{self._max_retries}): {e}"
                )
                broker_state.orders_failed += 1
                broker_state.consecutive_failures += 1

                # Check if broker should be marked degraded
                if broker_state.consecutive_failures >= broker_state.max_consecutive_failures:
                    broker_state.status = BrokerStatus.DEGRADED
                    logger.warning(f"Broker {broker_name} marked DEGRADED")

                if attempt < self._max_retries:
                    # Exponential backoff
                    backoff = min(2**attempt * 0.1, 2.0)
                    await asyncio.sleep(backoff)

        # All retries exhausted
        order.update_status(
            OrderStatus.UNKNOWN, f"Broker timeout after {self._max_retries} retries"
        )
        self._total_rejections += 1
        logger.error(f"[{order.symbol}] Order UNKNOWN — all retries exhausted")
        return False

    async def _process_paper_fill(self, order: ManagedOrder, adapter, broker_order_id: str) -> None:
        """Process fill from paper broker immediately."""
        order_status = await adapter.get_order_status(broker_order_id)
        if order_status and order_status.get("status") == "filled":
            fill = BrokerFill(
                broker_order_id=broker_order_id,
                our_order_id=order.order_id,
                symbol=order_status["symbol"],
                side=order_status["side"],
                filled_quantity=order_status["quantity"],
                fill_price=order_status["fill_price"],
                timestamp=order_status["timestamp"],
            )
            if self._on_fill:
                self._on_fill(order.order_id, fill.filled_quantity, fill.fill_price)
                self._total_fills_received += 1

    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an open order by broker order id."""
        order_id = None
        for our_order_id, mapped_broker_id in self._order_map.items():
            if mapped_broker_id == broker_order_id:
                order_id = our_order_id
                break

        if not order_id:
            logger.error(f"Cancel failed: unknown broker order id {broker_order_id}")
            return False

        broker_name = self._broker_for_order.get(order_id)
        if not broker_name:
            return False

        adapter = self._adapters[broker_name]
        try:
            success = await adapter.cancel_order(broker_order_id)
            if success:
                logger.info(f"Broker order {broker_order_id} CANCELLED")
            return success
        except Exception as e:
            logger.error(f"Cancel failed for broker order {broker_order_id}: {e}")
            return False

    async def cancel_order_async(self, our_order_id: str) -> bool:
        """Async version of cancel order using our internal order ID."""
        broker_order_id = self._order_map.get(our_order_id)
        if not broker_order_id:
            logger.error(f"Cancel failed: unknown order id {our_order_id}")
            return False

        broker_name = self._broker_for_order.get(our_order_id)
        if not broker_name:
            return False

        adapter = self._adapters[broker_name]
        try:
            success = await adapter.cancel_order(broker_order_id)
            if success:
                logger.info(f"Broker order {broker_order_id} CANCELLED")
            return success
        except Exception as e:
            logger.error(f"Cancel failed for broker order {broker_order_id}: {e}")
            return False

    async def submit_market_order_async(self, symbol: str, side: str, quantity: int) -> bool:
        """Submit a market order asynchronously (bypassing OMS pre-trade for emergency liquidation)."""
        logger.critical(f"Emergency submitting market order for {symbol} {side} {quantity}")
        order = ManagedOrder(
            symbol=symbol,
            side=OrderSide(side.lower()),
            order_type=OrderType.MARKET,
            quantity=quantity,
            remaining_quantity=quantity,
            price=0.0,
        )
        order.update_status(OrderStatus.VALIDATED, "Emergency liquidation")
        # In a real system we should register this in OMS. For this emergency async bypass:
        return await self.submit_order(order)

    def _select_broker(self) -> str | None:
        """Select the best broker based on routing strategy."""
        connected = {
            name: state
            for name, state in self._brokers.items()
            if state.status in (BrokerStatus.CONNECTED, BrokerStatus.DEGRADED)
        }

        if not connected:
            return None

        if self._routing_strategy == RoutingStrategy.PRIMARY:
            if self._primary_broker in connected:
                return self._primary_broker
            return None

        elif self._routing_strategy == RoutingStrategy.FAILOVER:
            if self._primary_broker in connected:
                state = connected[self._primary_broker]
                if state.status == BrokerStatus.CONNECTED:
                    return self._primary_broker
            # Fallback to any connected broker
            for name, state in connected.items():
                if state.status == BrokerStatus.CONNECTED:
                    return name
            # Last resort: degraded broker
            return next(iter(connected), None)

        elif self._routing_strategy == RoutingStrategy.BEST_PRICE:
            # For now, select lowest latency
            return min(connected, key=lambda n: connected[n].latency_ms)

        return self._primary_broker if self._primary_broker in connected else None

    async def check_heartbeats(self) -> dict[str, bool]:
        """Check heartbeat on all brokers, update status."""
        results = {}
        for name, adapter in self._adapters.items():
            try:
                alive = await adapter.heartbeat()
                if alive:
                    self._brokers[name].last_heartbeat = now_ist()
                    if self._brokers[name].status != BrokerStatus.CONNECTED:
                        self._brokers[name].status = BrokerStatus.CONNECTED
                        self._brokers[name].consecutive_failures = 0
                        logger.info(f"Broker {name} RECONNECTED")
                else:
                    self._brokers[name].status = BrokerStatus.DISCONNECTED
                    logger.warning(f"Broker {name} heartbeat FAILED")
                results[name] = alive
            except Exception as e:
                self._brokers[name].status = BrokerStatus.DISCONNECTED
                results[name] = False
                logger.error(f"Broker {name} heartbeat error: {e}")
        return results

    def get_status(self) -> dict:
        """Get EMS status report."""
        return {
            "routing": self._routing_strategy.value,
            "primary_broker": self._primary_broker,
            "brokers": {
                name: {
                    "status": state.status.value,
                    "latency_ms": state.latency_ms,
                    "orders_sent": state.orders_sent,
                    "orders_failed": state.orders_failed,
                }
                for name, state in self._brokers.items()
            },
            "total_submitted": self._total_orders_submitted,
            "total_fills": self._total_fills_received,
            "total_rejections": self._total_rejections,
        }
