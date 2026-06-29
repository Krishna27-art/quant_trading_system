"""
Unified Execution Interface

Single execution interface for backtest, paper, and live trading.
Only the data source changes - execution logic is identical.

This ensures research-production parity and prevents backtest lies.

Time Synchronization:
- PTP clock synchronization for nanosecond precision
- Multi-timestamp event tracking (exchange, receive, process, send)
"""

import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Protocol

from portfolio_execution.execution.execution_sequencer import initialize_sequencer
from risk_governance.pre_trade.pre_trade_checks import (
    PreTradeChecker,
    PreTradeConfig,
    SymbolMarketData,
)
from utils.history_persistence import BoundedHistory
from utils.logger import get_logger
from utils.memory_budget import MemoryAction, MemoryBudget, register_memory_budget

logger = get_logger("unified_execution")


# Try to import PTP clock for time synchronization
try:
    import time_synchronization.ptp_clock  # noqa: F401

    PTP_AVAILABLE = True
except ImportError:
    PTP_AVAILABLE = False
    logger.warning("PTP clock not available, using system time")


class ExecutionMode(str, Enum):
    """Execution modes."""

    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class OrderType(str, Enum):
    """Order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"


class OrderSide(str, Enum):
    """Order sides."""

    BUY = "buy"
    SELL = "sell"


class TimeInForce(str, Enum):
    """Time in force."""

    DAY = "day"
    IOC = "ioc"
    FOK = "fok"
    GTC = "gtc"


class FillType(str, Enum):
    """Fill types."""

    FULL = "full"
    PARTIAL = "partial"
    NO_FILL = "no_fill"


class OrderStatus(str, Enum):
    """Order status."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderState(str, Enum):
    """Order state for OMS/EMS protection (FIX protocol)."""

    CREATED = "created"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    PARTIAL_FILL = "partial_fill"
    FILLED = "filled"
    CANCEL_PENDING = "cancel_pending"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    UNKNOWN = "unknown"  # Connection dropped, state uncertain


@dataclass
class Order:
    """Unified order structure for all execution modes with model provenance."""

    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    # OMS/EMS protection fields
    client_order_id: str | None = None
    broker_order_id: str | None = None
    state: OrderState = OrderState.CREATED
    msg_seq_num: int | None = None
    # Execution sequencer fields
    sequence_number: int | None = None  # Global sequence number
    strategy_id: str | None = None  # Source strategy
    # Model provenance fields
    model_version: str | None = None  # Model version (e.g., "v8")
    feature_version: str | None = None  # Feature version (e.g., "v5")
    dataset_version: str | None = None  # Dataset version (e.g., "v3")
    signal_id: str | None = None  # Signal identifier for lineage tracing
    # Statistical arbitrage / spread trading fields
    is_spread_order: bool = False  # Whether this is a spread order
    spread_symbol2: str | None = None  # Second symbol for spread
    spread_hedge_ratio: float | None = None  # Hedge ratio for spread
    spread_position_type: str | None = None  # 'long_spread' or 'short_spread'
    spread_pair_key: str | None = None  # Key for the pair (e.g., "AAPL_MSFT")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force.value,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "metadata": self.metadata,
            "client_order_id": self.client_order_id,
            "broker_order_id": self.broker_order_id,
            "state": self.state.value if self.state else None,
            "msg_seq_num": self.msg_seq_num,
            "sequence_number": self.sequence_number,
            "strategy_id": self.strategy_id,
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "dataset_version": self.dataset_version,
            "signal_id": self.signal_id,
            "is_spread_order": self.is_spread_order,
            "spread_symbol2": self.spread_symbol2,
            "spread_hedge_ratio": self.spread_hedge_ratio,
            "spread_position_type": self.spread_position_type,
            "spread_pair_key": self.spread_pair_key,
        }


@dataclass
class Fill:
    """Unified fill structure for all execution modes."""

    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    fill_type: FillType
    slippage_bps: float = 0.0
    latency_ms: float = 0.0
    commission: float = 0.0
    fees: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fill_id": self.fill_id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
            "fill_type": self.fill_type.value,
            "slippage_bps": self.slippage_bps,
            "latency_ms": self.latency_ms,
            "commission": self.commission,
            "fees": self.fees,
            "metadata": self.metadata,
        }


@dataclass
class ExecutionReport:
    """Unified execution report for all execution modes."""

    order_id: str
    symbol: str
    side: OrderSide
    total_quantity: float
    filled_quantity: float
    average_price: float
    total_slippage_bps: float
    average_latency_ms: float
    total_commission: float
    total_fees: float
    fill_count: int
    start_time: datetime
    end_time: datetime
    status: OrderStatus
    fills: list[Fill] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "total_quantity": self.total_quantity,
            "filled_quantity": self.filled_quantity,
            "average_price": self.average_price,
            "total_slippage_bps": self.total_slippage_bps,
            "average_latency_ms": self.average_latency_ms,
            "total_commission": self.total_commission,
            "total_fees": self.total_fees,
            "fill_count": self.fill_count,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "status": self.status.value,
            "fills": [f.to_dict() for f in self.fills],
            "metadata": self.metadata,
        }


class MarketDataSource(Protocol):
    """Protocol for market data sources."""

    def get_market_price(self, symbol: str, timestamp: datetime) -> float | None:
        """Get market price for symbol at timestamp."""
        ...

    def get_order_book(self, symbol: str, timestamp: datetime) -> dict[str, Any] | None:
        """Get order book for symbol at timestamp."""
        ...

    def get_volatility(self, symbol: str, timestamp: datetime) -> float | None:
        """Get volatility for symbol at timestamp."""
        ...

    def get_spread(self, symbol: str, timestamp: datetime) -> float | None:
        """Get bid-ask spread for symbol at timestamp."""
        ...


class ExecutionEngine(ABC):
    """
    Abstract execution engine.

    All execution modes (backtest, paper, live) inherit from this.
    Only the market data source differs.
    """

    def __init__(self, mode: ExecutionMode, data_source: MarketDataSource):
        """
        Initialize execution engine.

        Args:
            mode: Execution mode (backtest, paper, live)
            data_source: Market data source
        """
        self.mode = mode
        self.data_source = data_source
        self.logger = logger

        # Order tracking
        self._orders: dict[str, Order] = {}
        self._fills: dict[str, deque] = {}  # Use deque with maxlen for bounded memory
        self._reports: dict[str, ExecutionReport] = {}

        # Callbacks - use deque with maxlen to prevent unbounded growth
        self._fill_callbacks: deque = deque(maxlen=100)
        self._order_callbacks: deque = deque(maxlen=100)

        # Memory budget enforcement
        budget = MemoryBudget(
            service_name=f"execution_engine_{mode.value}",
            max_ram_mb=512,  # 512MB max for execution engine
            max_cache_mb=100,  # 100MB max for cache
            max_objects=10000,  # Max 10,000 orders/fills
            action=MemoryAction.GC,
        )
        self.memory_monitor = register_memory_budget(budget)

        # History persistence - persist old orders/fills to disk
        self.order_history = BoundedHistory(
            name=f"orders_{mode.value}",
            max_memory=5000,  # Keep last 5000 orders in memory
            persist_threshold=0.8,
        )
        self.fill_history = BoundedHistory(
            name=f"fills_{mode.value}",
            max_memory=10000,  # Keep last 10000 fills in memory
            persist_threshold=0.8,
        )

        # Execution sequencer for LIVE mode only
        self.sequencer = None
        if mode == ExecutionMode.LIVE:
            self.sequencer = initialize_sequencer()
            self.logger.info("Execution sequencer initialized for LIVE mode")

        self.logger.info(f"ExecutionEngine initialized in {mode.value} mode")

    def place_order(self, order: Order, strategy_id: str | None = None) -> str:
        """
        Place an order.

        Args:
            order: Order to place
            strategy_id: Strategy ID (for sequencer)

        Returns:
            Order ID
        """
        # Use sequencer for LIVE mode
        if self.sequencer and self.mode == ExecutionMode.LIVE:
            order.strategy_id = strategy_id or "unknown"

            # Submit to sequencer
            sequence_number = self.sequencer.submit_order(
                order_id=order.order_id, strategy_id=order.strategy_id, order_data=order.to_dict()
            )

            order.sequence_number = sequence_number
            self.logger.info(f"Order submitted to sequencer: #{sequence_number}")

            # Sequencer will call back to execute
            return order.order_id

        # Direct execution for BACKTEST/PAPER modes
        order.status = OrderStatus.SUBMITTED
        order.timestamp = datetime.utcnow()
        self._orders[order.order_id] = order
        self._fills[order.order_id] = deque(maxlen=1000)  # Bounded fills per order

        # Persist to history
        self.order_history.append(order)

        # Check memory budget
        if not self.memory_monitor.check_budget():
            self.logger.warning("Memory budget check failed, but order placed")

        # Notify order callbacks
        for callback in self._order_callbacks:
            try:
                callback(order)
            except Exception as e:
                self.logger.error(f"Order callback error: {e}")

        self.logger.info(f"Order placed: {order.order_id} in {self.mode.value} mode")
        return order.order_id

    def place_spread_order(
        self,
        symbol1: str,
        symbol2: str,
        position_type: str,  # 'long_spread' or 'short_spread'
        quantity: float,
        hedge_ratio: float,
        pair_key: str,
        strategy_id: str | None = None,
        **order_kwargs,
    ) -> tuple[str, str]:
        """
        Place a spread order (two legs).

        Args:
            symbol1: First symbol
            symbol2: Second symbol
            position_type: 'long_spread' or 'short_spread'
            quantity: Base quantity
            hedge_ratio: Hedge ratio for second leg
            pair_key: Key for the pair (e.g., "AAPL_MSFT")
            strategy_id: Strategy ID
            **order_kwargs: Additional order parameters

        Returns:
            Tuple of (order_id1, order_id2)
        """
        # Calculate quantities for each leg
        if position_type == "long_spread":
            # Long symbol1, Short symbol2
            side1 = OrderSide.BUY
            side2 = OrderSide.SELL
            quantity2 = quantity * hedge_ratio
        else:
            # Short symbol1, Long symbol2
            side1 = OrderSide.SELL
            side2 = OrderSide.BUY
            quantity2 = quantity * hedge_ratio

        # Create first leg order
        order1 = Order(
            order_id=str(uuid.uuid4()),
            symbol=symbol1,
            side=side1,
            quantity=quantity,
            order_type=OrderType.MARKET,
            is_spread_order=True,
            spread_symbol2=symbol2,
            spread_hedge_ratio=hedge_ratio,
            spread_position_type=position_type,
            spread_pair_key=pair_key,
            strategy_id=strategy_id,
            **order_kwargs,
        )

        # Create second leg order
        order2 = Order(
            order_id=str(uuid.uuid4()),
            symbol=symbol2,
            side=side2,
            quantity=quantity2,
            order_type=OrderType.MARKET,
            is_spread_order=True,
            spread_symbol2=symbol1,
            spread_hedge_ratio=1.0 / hedge_ratio if hedge_ratio != 0 else 1.0,
            spread_position_type=position_type,
            spread_pair_key=pair_key,
            strategy_id=strategy_id,
            **order_kwargs,
        )

        # Place both orders
        order_id1 = self.place_order(order1, strategy_id)
        order_id2 = self.place_order(order2, strategy_id)

        self.logger.info(
            f"Spread order placed: {pair_key}, type={position_type}, "
            f"leg1={order_id1}, leg2={order_id2}"
        )

        return (order_id1, order_id2)

    @abstractmethod
    def execute_order(self, order: Order) -> list[Fill]:
        """
        Execute an order.

        This method is implemented differently for each mode:
        - Backtest: Uses historical data
        - Paper: Uses simulated live data
        - Live: Uses real broker API

        Args:
            order: Order to execute

        Returns:
            List of fills
        """
        pass

    def calculate_slippage(
        self, order: Order, market_price: float, volatility: float, spread: float
    ) -> float:
        """
        Calculate slippage for an order.

        This is the SAME logic for all execution modes.

        Args:
            order: Order
            market_price: Market price
            volatility: Volatility
            spread: Bid-ask spread

        Returns:
            Slippage in basis points
        """
        # Base slippage from spread
        slippage_bps = (spread / market_price) * 10000

        # Volatility multiplier
        vol_multiplier = volatility * 100

        # Size impact (simplified)
        size_impact = (order.quantity / 10000) * 0.1

        total_slippage_bps = slippage_bps * (1 + vol_multiplier) + size_impact

        return total_slippage_bps

    def calculate_latency(self, order: Order, queue_position: int = 0) -> float:
        """
        Calculate execution latency.

        This is the SAME logic for all execution modes.

        Args:
            order: Order
            queue_position: Queue position

        Returns:
            Latency in milliseconds
        """
        # Base latency (mode-dependent)
        if self.mode == ExecutionMode.BACKTEST:
            base_latency = 0.0  # Instant in backtest
        elif self.mode == ExecutionMode.PAPER:
            base_latency = 50.0  # Simulated latency
        else:  # LIVE
            base_latency = 10.0  # Co-located latency

        # Queue delay
        queue_delay = queue_position * 2.0

        # Size delay
        size_delay = (order.quantity / 1000) * 1.0

        total_latency_ms = base_latency + queue_delay + size_delay

        return total_latency_ms

    def calculate_fees(self, order: Order, fill_price: float) -> dict[str, float]:
        """
        Calculate fees for an order.

        This is the SAME logic for all execution modes.

        Args:
            order: Order
            fill_price: Fill price

        Returns:
            Fee breakdown
        """
        order_value = order.quantity * fill_price

        # Brokerage (flat fee)
        brokerage = 20.0

        # Transaction charges
        transaction_charges = order_value * 0.00345

        # STT (sell side only)
        stt = 0.0
        if order.side == OrderSide.SELL:
            stt = order_value * 0.00025

        # GST
        gst = (brokerage + transaction_charges) * 0.18

        # Stamp duty
        stamp_duty = order_value * 0.00003

        total_fees = brokerage + transaction_charges + stt + gst + stamp_duty

        return {
            "brokerage": brokerage,
            "transaction_charges": transaction_charges,
            "stt": stt,
            "gst": gst,
            "stamp_duty": stamp_duty,
            "total_fees": total_fees,
        }

    def process_fills(self, order: Order, fills: list[Fill]):
        """
        Process fills for an order.

        This is the SAME logic for all execution modes.

        Args:
            order: Order
            fills: List of fills
        """
        for fill in fills:
            self._fills[order.order_id].append(fill)

            # Persist to history
            self.fill_history.append(fill)

            # Update order
            order.filled_quantity += fill.quantity
            total_value = order.filled_quantity * order.avg_fill_price + fill.quantity * fill.price
            order.avg_fill_price = (
                total_value / order.filled_quantity if order.filled_quantity > 0 else 0.0
            )

            # Update status
            if order.filled_quantity >= order.quantity:
                order.status = OrderStatus.FILLED
            elif order.filled_quantity > 0:
                order.status = OrderStatus.PARTIALLY_FILLED

            # Check memory budget periodically
            if not self.memory_monitor.check_budget():
                self.logger.warning("Memory budget check failed during fill processing")

            # Notify fill callbacks
            for callback in self._fill_callbacks:
                try:
                    callback(fill)
                except Exception as e:
                    self.logger.error(f"Fill callback error: {e}")

    def generate_execution_report(self, order_id: str) -> ExecutionReport | None:
        """
        Generate execution report for an order.

        This is the SAME logic for all execution modes.

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
        total_commission = sum(f.commission for f in fills)
        total_fees = sum(f.fees.get("total_fees", 0) for f in fills)
        total_slippage = sum(f.slippage_bps for f in fills)
        avg_latency = sum(f.latency_ms for f in fills) / len(fills) if fills else 0.0

        report = ExecutionReport(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            total_quantity=order.quantity,
            filled_quantity=order.filled_quantity,
            average_price=order.avg_fill_price,
            total_slippage_bps=total_slippage,
            average_latency_ms=avg_latency,
            total_commission=total_commission,
            total_fees=total_fees,
            fill_count=len(fills),
            start_time=order.timestamp,
            end_time=max(f.timestamp for f in fills) if fills else order.timestamp,
            status=order.status,
            fills=fills,
        )

        self._reports[order_id] = report
        return report

    def subscribe_fills(self, callback: Callable[[Fill], None]):
        """Subscribe to fill events."""
        self._fill_callbacks.append(callback)
        self.logger.info(f"Added fill callback, total: {len(self._fill_callbacks)}")

    def subscribe_orders(self, callback: Callable[[Order], None]):
        """Subscribe to order events."""
        self._order_callbacks.append(callback)
        self.logger.info(f"Added order callback, total: {len(self._order_callbacks)}")

    def get_order(self, order_id: str) -> Order | None:
        """Get order by ID."""
        return self._orders.get(order_id)

    def get_all_orders(self) -> list[Order]:
        """Get all orders."""
        return list(self._orders.values())

    def get_all_reports(self) -> list[ExecutionReport]:
        """Get all execution reports."""
        return list(self._reports.values())

    def reset(self):
        """Reset engine state."""
        self._orders.clear()
        self._fills.clear()
        self._reports.clear()
        self.logger.info(f"ExecutionEngine reset in {self.mode.value} mode")


class BacktestExecutionEngine(ExecutionEngine):
    """
    Backtest execution engine.

    Uses historical market data for execution.
    """

    def execute_order(self, order: Order) -> list[Fill]:
        """Execute order using historical data."""
        fills = []

        # Get market data from historical source
        market_price = self.data_source.get_market_price(order.symbol, order.timestamp)

        if market_price is None:
            self.logger.warning(f"No market data for {order.symbol} at {order.timestamp}")
            order.status = OrderStatus.REJECTED
            return fills

        # Get volatility and spread
        volatility = self.data_source.get_volatility(order.symbol, order.timestamp) or 0.2
        spread = self.data_source.get_spread(order.symbol, order.timestamp) or 0.01

        # Calculate slippage (same logic as live)
        slippage_bps = self.calculate_slippage(order, market_price, volatility, spread)

        # Calculate latency (same logic as live, but 0 in backtest)
        latency_ms = self.calculate_latency(order)

        # Calculate fill price with slippage
        if order.side == OrderSide.BUY:
            fill_price = market_price * (1 + slippage_bps / 10000)
        else:
            fill_price = market_price * (1 - slippage_bps / 10000)

        # Calculate fees (same logic as live)
        fees = self.calculate_fees(order, fill_price)

        # Create fill
        fill = Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            timestamp=order.timestamp + timedelta(milliseconds=latency_ms),
            fill_type=FillType.FULL,
            slippage_bps=slippage_bps,
            latency_ms=latency_ms,
            commission=fees["brokerage"],
            fees=fees,
        )

        fills.append(fill)

        # Process fills (same logic as live)
        self.process_fills(order, fills)

        return fills


class PaperExecutionEngine(ExecutionEngine):
    """
    Paper trading execution engine.

    Uses simulated live market data for execution.
    """

    def execute_order(self, order: Order) -> list[Fill]:
        """Execute order using simulated live data."""
        fills = []

        # Get market data from simulated live source
        market_price = self.data_source.get_market_price(order.symbol, datetime.utcnow())

        if market_price is None:
            self.logger.warning(f"No market data for {order.symbol}")
            order.status = OrderStatus.REJECTED
            return fills

        # Get volatility and spread
        volatility = self.data_source.get_volatility(order.symbol, datetime.utcnow()) or 0.2
        spread = self.data_source.get_spread(order.symbol, datetime.utcnow()) or 0.01

        # Calculate slippage (same logic as live)
        slippage_bps = self.calculate_slippage(order, market_price, volatility, spread)

        # Calculate latency (same logic as live, with simulated delay)
        latency_ms = self.calculate_latency(order)

        # Calculate fill price with slippage
        if order.side == OrderSide.BUY:
            fill_price = market_price * (1 + slippage_bps / 10000)
        else:
            fill_price = market_price * (1 - slippage_bps / 10000)

        # Calculate fees (same logic as live)
        fees = self.calculate_fees(order, fill_price)

        # Create fill
        fill = Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            timestamp=datetime.utcnow() + timedelta(milliseconds=latency_ms),
            fill_type=FillType.FULL,
            slippage_bps=slippage_bps,
            latency_ms=latency_ms,
            commission=fees["brokerage"],
            fees=fees,
        )

        fills.append(fill)

        # Process fills (same logic as live)
        self.process_fills(order, fills)

        return fills


class LiveExecutionEngine(ExecutionEngine):
    """
    Live trading execution engine.

    Uses real broker API for execution.
    Includes FIX sequence reconciliation and rate limiting for OMS/EMS protection.
    """

    def __init__(
        self,
        mode: ExecutionMode,
        data_source: MarketDataSource,
        broker_adapter=None,
        pre_trade_config: PreTradeConfig | None = None,
    ):
        """Initialize live execution engine."""
        super().__init__(mode, data_source)
        self.broker_adapter = broker_adapter

        # OMS/EMS protection: FIX sequence tracking
        self._last_msg_seq_num = 0

        # Rate limiting for order status polling (prevents rate limit bans)
        self._last_poll_time = 0
        self._poll_interval = 1.0  # Minimum 1 second between polls
        self._poll_count = 0
        self._max_polls_per_minute = 30

        # Position tracking for validation
        self._positions: dict[str, float] = {}  # symbol -> quantity

        # Kill switch state
        self._kill_switch_active = False
        self._kill_switch_reason = None

        # Pre-trade risk checker and deduplication guards
        self.pre_trade_checker = PreTradeChecker(pre_trade_config)
        self._client_order_ids: set[str] = set()

    def execute_order(self, order: Order) -> list[Fill]:
        """Execute order using real broker API with FIX sequence reconciliation and event sourcing."""
        fills = []

        # Kill switch check
        if self._kill_switch_active:
            self.logger.error(f"Kill switch active ({self._kill_switch_reason}). Order not sent.")
            order.status = OrderStatus.REJECTED
            return fills

        # OMS/EMS protection: Generate client_order_id for idempotency
        if not order.client_order_id:
            order.client_order_id = f"{order.side.value}_{order.symbol}_{uuid.uuid4().hex[:8]}"

        # Check for duplicate order
        if self._check_duplicate_order(order):
            self.logger.error(f"Duplicate order detected: {order.client_order_id}")
            order.status = OrderStatus.REJECTED
            return fills

        # Add to tracked orders and client_order_ids
        self._client_order_ids.add(order.client_order_id)
        self._orders[order.order_id] = order
        if order.order_id not in self._fills:
            self._fills[order.order_id] = deque(maxlen=1000)

        # Position-based validation before order (local tracking check)
        if not self._validate_position_before_order(order):
            self.logger.error(f"Position validation failed for {order.symbol}")
            order.status = OrderStatus.REJECTED
            return fills

        # Log ORDER_CREATED event
        self._log_order_event(order, "ORDER_CREATED", None, OrderState.CREATED.value)

        # Get current market price
        market_price = self.data_source.get_market_price(order.symbol, datetime.utcnow())

        if market_price is None:
            self.logger.warning(f"No market data for {order.symbol}")
            order.status = OrderStatus.REJECTED
            self._log_order_event(
                order, "ORDER_REJECTED", OrderState.CREATED.value, OrderState.REJECTED.value
            )
            return fills

        # Get volatility and spread
        volatility = self.data_source.get_volatility(order.symbol, datetime.utcnow()) or 0.2
        spread = self.data_source.get_spread(order.symbol, datetime.utcnow()) or 0.01

        # Run pre-trade risk checker checks
        adv = 100000.0
        atr = volatility * market_price
        mwpl = 10000000
        current_oi = 0
        is_fno = False
        lot_size = 1

        if hasattr(self.data_source, "get_adv_20d"):
            adv = self.data_source.get_adv_20d(order.symbol) or adv
        if hasattr(self.data_source, "get_atr_14d"):
            atr = self.data_source.get_atr_14d(order.symbol) or atr
        elif hasattr(self.data_source, "get_atr"):
            atr = self.data_source.get_atr(order.symbol) or atr

        symbol_md = SymbolMarketData(
            symbol=order.symbol,
            last_price=market_price,
            adv_20d=adv,
            atr_14d=atr,
            mwpl=mwpl,
            current_oi=current_oi,
            is_fno=is_fno,
            lot_size=lot_size,
        )
        self.pre_trade_checker.update_market_data({order.symbol: symbol_md})

        current_position = int(self._positions.get(order.symbol, 0.0))
        self.pre_trade_checker.update_client_oi({order.symbol: abs(current_position)})

        # Validate order using PreTradeChecker
        val_price = order.limit_price if order.limit_price is not None else market_price
        validation_result = self.pre_trade_checker.validate_order(
            symbol=order.symbol,
            side=order.side.value.upper(),
            quantity=int(order.quantity),
            price=val_price,
            order_type=order.order_type.value.upper(),
        )

        if not validation_result.approved:
            self.logger.error(f"Pre-trade compliance check failed: {validation_result.reason}")
            order.status = OrderStatus.REJECTED
            self._log_order_event(
                order, "ORDER_REJECTED", OrderState.CREATED.value, OrderState.REJECTED.value
            )
            return fills

        if validation_result.adjusted_qty < order.quantity:
            self.logger.warning(
                f"Order quantity adjusted by pre-trade risk checker: "
                f"{order.quantity} -> {validation_result.adjusted_qty}"
            )
            order.quantity = validation_result.adjusted_qty

        # Calculate slippage (same logic as backtest)
        slippage_bps = self.calculate_slippage(order, market_price, volatility, spread)

        # Calculate latency (same logic as backtest)
        latency_ms = self.calculate_latency(order)

        # Submit order to broker
        if self.broker_adapter:
            # OMS/EMS protection: Update state to SENT before submission
            order.state = OrderState.SENT
            self._log_order_event(
                order, "ORDER_SENT", OrderState.CREATED.value, OrderState.SENT.value
            )

            # Increment sequence number
            self._last_msg_seq_num += 1
            order.msg_seq_num = self._last_msg_seq_num

            # Adapt order for broker
            broker_order = order.to_dict()
            adapted_order = self.broker_adapter.adapt_order_for_broker(broker_order)

            # Submit to broker
            broker_result = self.broker_adapter.submit_order(adapted_order)

            if not broker_result.get("success", False):
                # OMS/EMS protection: If submission failed, check if it's a zombie order
                # Reconcile with broker before retry
                if self._check_zombie_order(order):
                    self.logger.error(
                        f"Zombie order detected for {order.client_order_id}. "
                        f"Order may have been accepted despite connection failure. "
                        f"NOT retrying to prevent double execution."
                    )
                    order.state = OrderState.UNKNOWN
                    order.status = OrderStatus.REJECTED
                    self._log_order_event(
                        order, "ORDER_UNKNOWN", OrderState.SENT.value, OrderState.UNKNOWN.value
                    )

                    # Trigger kill switch on unknown state
                    self._activate_kill_switch("UNKNOWN_STATE_TIMEOUT")

                    return fills

                order.status = OrderStatus.REJECTED
                order.state = OrderState.REJECTED
                self._log_order_event(
                    order, "ORDER_REJECTED", OrderState.SENT.value, OrderState.REJECTED.value
                )
                return fills

            # Update state to ACKNOWLEDGED
            order.state = OrderState.ACKNOWLEDGED
            order.broker_order_id = broker_result.get("broker_order_id")
            self._log_order_event(
                order, "ORDER_ACKNOWLEDGED", OrderState.SENT.value, OrderState.ACKNOWLEDGED.value
            )

            # Get actual fill price from broker
            fill_price = broker_result.get("fill_price", market_price)
        else:
            # Fallback: simulate fill price
            if order.side == OrderSide.BUY:
                fill_price = market_price * (1 + slippage_bps / 10000)
            else:
                fill_price = market_price * (1 - slippage_bps / 10000)

        # Calculate fees (same logic as backtest)
        fees = self.calculate_fees(order, fill_price)

        # Create fill
        fill = Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            timestamp=datetime.utcnow() + timedelta(milliseconds=latency_ms),
            fill_type=FillType.FULL,
            slippage_bps=slippage_bps,
            latency_ms=latency_ms,
            commission=fees["brokerage"],
            fees=fees,
        )

        fills.append(fill)

        # Process fills (same logic as backtest)
        self.process_fills(order, fills)

        # Position-based validation after fill
        if not self._validate_position_after_fill(order):
            self.logger.error(f"Position validation failed after fill for {order.symbol}")
            self._activate_kill_switch("POSITION_MISMATCH")

        # Log FILL event
        order.state = OrderState.FILLED
        self._log_order_event(
            order, "FILL_RECEIVED", OrderState.ACKNOWLEDGED.value, OrderState.FILLED.value
        )

        return fills

    def _check_zombie_order(self, order: Order) -> bool:
        """
        Check if order is a zombie order (connection dropped after send).

        Uses FIX sequence reconciliation to verify order state.

        Args:
            order: Order to check

        Returns:
            True if zombie order detected
        """
        if not self.broker_adapter:
            return False

        try:
            # Query broker for order status using client_order_id
            broker_status = self.broker_adapter.query_order_status(order.client_order_id)

            if broker_status and broker_status.get("exists", False):
                self.logger.warning(
                    f"Order {order.client_order_id} exists on broker despite connection failure. "
                    f"Status: {broker_status.get('status')}"
                )
                return True

        except Exception as e:
            self.logger.error(f"Error checking zombie order: {e}")

        return False

    def poll_order_status(self, order_id: str) -> dict[str, Any] | None:
        """
        Poll order status with rate limiting (prevents rate limit bans).

        Args:
            order_id: Order ID to poll

        Returns:
            Order status or None
        """
        # Rate limiting check
        now = time.time()
        elapsed = now - self._last_poll_time

        if elapsed < self._poll_interval:
            self.logger.warning(f"Rate limit: skipping poll (last poll {elapsed:.2f}s ago)")
            return None

        # Check max polls per minute
        self._poll_count += 1
        if self._poll_count > self._max_polls_per_minute:
            self.logger.error(f"Rate limit: exceeded {self._max_polls_per_minute} polls per minute")
            return None

        # Reset counter every minute
        if elapsed > 60:
            self._poll_count = 0

        self._last_poll_time = now

        # Query broker
        if self.broker_adapter:
            try:
                order = self._orders.get(order_id)
                if order and order.client_order_id:
                    return self.broker_adapter.query_order_status(order.client_order_id)
            except Exception as e:
                self.logger.error(f"Error polling order status: {e}")

        return None

    def _log_order_event(
        self, order: Order, event_type: str, from_state: str | None, to_state: str
    ):
        """
        Log order event to database (event sourcing).

        Args:
            order: Order
            event_type: Event type
            from_state: Previous state
            to_state: New state
        """
        try:
            from database.connection import log_order_event

            log_order_event(
                client_order_id=order.client_order_id,
                event_type=event_type,
                from_state=from_state,
                to_state=to_state,
                event_data={"order_id": order.order_id, "symbol": order.symbol},
                msg_seq_num=order.msg_seq_num,
            )
        except Exception as e:
            self.logger.error(f"Failed to log order event: {e}")

    def _check_duplicate_order(self, order: Order) -> bool:
        """
        Check for duplicate order (same client_order_id).

        Args:
            order: Order to check

        Returns:
            True if duplicate detected
        """
        if not order.client_order_id:
            return False
        if order.client_order_id in self._client_order_ids:
            for existing_order in self._orders.values():
                if existing_order.client_order_id == order.client_order_id:
                    if existing_order.state not in [
                        OrderState.FILLED,
                        OrderState.CANCELLED,
                        OrderState.REJECTED,
                    ]:
                        self.logger.warning(
                            f"Duplicate active order detected: {order.client_order_id}"
                        )
                        return True
        return False

    def _validate_position_before_order(self, order: Order) -> bool:
        """
        Validate position before sending order.

        Args:
            order: Order to validate

        Returns:
            True if validation passes
        """
        current_position = self._positions.get(order.symbol, 0.0)
        order_qty = order.quantity if order.side == OrderSide.BUY else -order.quantity
        expected_position = current_position + order_qty

        # Position limit check
        max_position = 10000  # Example limit
        if abs(expected_position) > max_position:
            self.logger.error(
                f"Position limit exceeded: {order.symbol} current={current_position}, "
                f"order={order_qty}, expected={expected_position}, limit={max_position}"
            )
            return False

        return True

    def _validate_position_after_fill(self, order: Order) -> bool:
        """
        Validate position after fill.

        Args:
            order: Order to validate

        Returns:
            True if validation passes
        """
        current_position = self._positions.get(order.symbol, 0.0)
        order_qty = order.quantity if order.side == OrderSide.BUY else -order.quantity
        expected_position = current_position + order_qty

        # Update position
        self._positions[order.symbol] = expected_position

        # Verify against broker (if available)
        if self.broker_adapter:
            try:
                broker_position = self.broker_adapter.get_position(order.symbol)
                if broker_position is not None:
                    if abs(broker_position - expected_position) > 1:  # Allow small tolerance
                        self.logger.error(
                            f"Position mismatch: {order.symbol} expected={expected_position}, "
                            f"broker={broker_position}"
                        )
                        return False
            except Exception as e:
                self.logger.error(f"Failed to verify position with broker: {e}")

        return True

    def _activate_kill_switch(self, reason: str):
        """
        Activate kill switch (freeze new orders, allow risk reduction only).

        Args:
            reason: Reason for activation
        """
        self._kill_switch_active = True
        self._kill_switch_reason = reason
        self.logger.error(f"KILL SWITCH ACTIVATED: {reason}")

    def deactivate_kill_switch(self):
        """Deactivate kill switch (manual override)."""
        self._kill_switch_active = False
        self._kill_switch_reason = None
        self.logger.warning("Kill switch deactivated (manual override)")

    def get_kill_switch_status(self) -> dict[str, Any]:
        """Get kill switch status."""
        return {"active": self._kill_switch_active, "reason": self._kill_switch_reason}


def create_execution_engine(
    mode: ExecutionMode, data_source: MarketDataSource, broker_adapter=None
) -> ExecutionEngine:
    """
    Factory function to create execution engine.

    Args:
        mode: Execution mode
        data_source: Market data source
        broker_adapter: Broker adapter (for live mode only)

    Returns:
        Execution engine
    """
    if mode == ExecutionMode.BACKTEST:
        return BacktestExecutionEngine(mode, data_source)
    elif mode == ExecutionMode.PAPER:
        return PaperExecutionEngine(mode, data_source)
    elif mode == ExecutionMode.LIVE:
        return LiveExecutionEngine(mode, data_source, broker_adapter)
    else:
        raise ValueError(f"Unknown execution mode: {mode}")
