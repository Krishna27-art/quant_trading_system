"""
Execution Simulator

Simulates execution for research parity with production.
Validates execution assumptions before live trading.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from utils.logger import get_logger

logger = get_logger("execution_simulator")


class ExecutionType(str, Enum):
    """Types of execution."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    ALGORITHMIC = "algorithmic"


class FillType(str, Enum):
    """Types of fills."""

    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"


@dataclass
class SimulatedOrder:
    """Simulated order for backtesting."""

    order_id: str
    symbol: str
    side: str  # buy/sell
    quantity: int
    price: float | None
    order_type: ExecutionType
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "order_type": self.order_type.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class SimulatedFill:
    """Simulated fill for backtesting."""

    fill_id: str
    order_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    timestamp: datetime
    fill_type: FillType
    slippage: float = 0.0
    latency_ms: float = 0.0
    queue_position: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fill_id": self.fill_id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
            "fill_type": self.fill_type.value,
            "slippage": self.slippage,
            "latency_ms": self.latency_ms,
            "queue_position": self.queue_position,
            "metadata": self.metadata,
        }


@dataclass
class ExecutionReport:
    """Execution report for research parity."""

    order_id: str
    symbol: str
    total_quantity: int
    filled_quantity: int
    average_price: float
    total_slippage: float
    average_latency_ms: float
    fill_count: int
    start_time: datetime
    end_time: datetime
    fills: list[SimulatedFill] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "total_quantity": self.total_quantity,
            "filled_quantity": self.filled_quantity,
            "average_price": self.average_price,
            "total_slippage": self.total_slippage,
            "average_latency_ms": self.average_latency_ms,
            "fill_count": self.fill_count,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "fills": [f.to_dict() for f in self.fills],
            "metadata": self.metadata,
        }


class ExecutionSimulator:
    """
    Execution simulator for research parity.

    Simulates execution to validate assumptions before production.
    Uses Nautilus for event-driven backtesting.
    """

    def __init__(self):
        """Initialize execution simulator."""
        self.logger = logger

        # Order tracking
        self._orders: dict[str, SimulatedOrder] = {}
        self._fills: dict[str, list[SimulatedFill]] = {}
        self._reports: dict[str, ExecutionReport] = {}

        # Market data (in production, from Nautilus)
        self._market_data: dict[str, dict[str, Any]] = {}

        # Callbacks
        self._fill_callbacks: list[Callable[[SimulatedFill], None]] = []
        self._order_callbacks: list[Callable[[SimulatedOrder], None]] = []

        self.logger.info("ExecutionSimulator initialized")

    def place_order(self, order: SimulatedOrder) -> str:
        """
        Place a simulated order.

        Args:
            order: Simulated order

        Returns:
            Order ID
        """
        self._orders[order.order_id] = order
        self._fills[order.order_id] = []

        # Notify order callbacks
        for callback in self._order_callbacks:
            try:
                callback(order)
            except Exception as e:
                self.logger.error(f"Order callback error: {e}")

        self.logger.info(f"Placed simulated order: {order.order_id}")
        return order.order_id

    def simulate_fill(
        self,
        order_id: str,
        market_price: float,
        slippage: float = 0.0,
        latency_ms: float = 0.0,
        queue_position: int = 0,
    ) -> SimulatedFill | None:
        """
        Simulate a fill for an order.

        Args:
            order_id: Order ID
            market_price: Current market price
            slippage: Slippage amount
            latency_ms: Execution latency in milliseconds
            queue_position: Queue position at fill

        Returns:
            Simulated fill or None
        """
        if order_id not in self._orders:
            self.logger.error(f"Order {order_id} not found")
            return None

        order = self._orders[order_id]

        # Calculate fill price with slippage
        if order.side == "buy":
            fill_price = market_price * (1 + slippage)
        else:
            fill_price = market_price * (1 - slippage)

        # Create fill
        fill = SimulatedFill(
            fill_id=str(uuid.uuid4()),
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,  # Full fill for simplicity
            price=fill_price,
            timestamp=datetime.utcnow() + timedelta(milliseconds=latency_ms),
            fill_type=FillType.FULL,
            slippage=slippage,
            latency_ms=latency_ms,
            queue_position=queue_position,
        )

        self._fills[order_id].append(fill)

        # Notify fill callbacks
        for callback in self._fill_callbacks:
            try:
                callback(fill)
            except Exception as e:
                self.logger.error(f"Fill callback error: {e}")

        self.logger.info(
            f"Simulated fill: {fill.fill_id}, order={order_id}, "
            f"price={fill_price:.2f}, slippage={slippage:.4f}, latency={latency_ms:.2f}ms"
        )

        return fill

    def get_execution_report(self, order_id: str) -> ExecutionReport | None:
        """
        Get execution report for an order.

        Args:
            order_id: Order ID

        Returns:
            Execution report or None
        """
        if order_id not in self._orders:
            return None

        order = self._orders[order_id]
        fills = self._fills.get(order_id, [])

        if not fills:
            return None

        # Calculate metrics
        filled_quantity = sum(f.quantity for f in fills)
        total_value = sum(f.price * f.quantity for f in fills)
        average_price = total_value / filled_quantity if filled_quantity > 0 else 0.0
        total_slippage = sum(f.slippage for f in fills)
        average_latency = sum(f.latency_ms for f in fills) / len(fills) if fills else 0.0

        report = ExecutionReport(
            order_id=order_id,
            symbol=order.symbol,
            total_quantity=order.quantity,
            filled_quantity=filled_quantity,
            average_price=average_price,
            total_slippage=total_slippage,
            average_latency_ms=average_latency,
            fill_count=len(fills),
            start_time=order.timestamp,
            end_time=max(f.timestamp for f in fills) if fills else order.timestamp,
            fills=fills,
        )

        self._reports[order_id] = report
        return report

    def subscribe_fills(self, callback: Callable[[SimulatedFill], None]):
        """
        Subscribe to fill events.

        Args:
            callback: Callback function
        """
        self._fill_callbacks.append(callback)
        self.logger.info(f"Added fill callback, total callbacks: {len(self._fill_callbacks)}")

    def subscribe_orders(self, callback: Callable[[SimulatedOrder], None]):
        """
        Subscribe to order events.

        Args:
            callback: Callback function
        """
        self._order_callbacks.append(callback)
        self.logger.info(f"Added order callback, total callbacks: {len(self._order_callbacks)}")

    def update_market_data(self, symbol: str, data: dict[str, Any]):
        """
        Update market data for symbol.

        Args:
            symbol: Trading symbol
            data: Market data
        """
        self._market_data[symbol] = data

    def get_market_data(self, symbol: str) -> dict[str, Any] | None:
        """
        Get market data for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Market data or None
        """
        return self._market_data.get(symbol)

    def get_all_reports(self) -> list[ExecutionReport]:
        """
        Get all execution reports.

        Returns:
            List of execution reports
        """
        return list(self._reports.values())

    def reset(self):
        """Reset simulator state."""
        self._orders.clear()
        self._fills.clear()
        self._reports.clear()
        self._market_data.clear()
        self.logger.info("ExecutionSimulator reset")

    def get_status(self) -> dict[str, Any]:
        """
        Get simulator status.

        Returns:
            Status dictionary
        """
        return {
            "total_orders": len(self._orders),
            "total_fills": sum(len(fills) for fills in self._fills.values()),
            "total_reports": len(self._reports),
            "fill_callbacks": len(self._fill_callbacks),
            "order_callbacks": len(self._order_callbacks),
            "market_data_symbols": len(self._market_data),
            "timestamp": datetime.utcnow().isoformat(),
        }
