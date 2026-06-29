"""
Order Management System (OMS)

Institutional-grade OMS that manages the full order lifecycle:
PENDING → SENT → PARTIAL_FILL → FILLED → CANCELLED → REJECTED

Key responsibilities:
- Pre-trade compliance (SEBI limits, fat-finger, restricted list)
- Order state tracking with audit trail
- Deduplication to prevent double-sends
- Broker timeout handling with "UNKNOWN" state reconciliation
- Position inventory tracking

This is the WHAT layer — decides what orders to place.
The EMS (Execution Management System) handles HOW to execute them.
"""

import threading
import time
import uuid
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from risk_governance.pre_trade.pre_trade_checks import (
    PreTradeChecker,
    PreTradeConfig,
    SymbolMarketData,
)
from utils.logger import get_logger
from utils.time_utils import now_ist

logger = get_logger("oms")


class OrderStatus(str, Enum):
    """Full order lifecycle states."""

    PENDING = "pending"  # Created, not yet validated
    VALIDATED = "validated"  # Passed pre-trade checks
    SENT = "sent"  # Sent to broker
    ACKNOWLEDGED = "acknowledged"  # Broker acknowledged receipt
    PARTIAL_FILL = "partial_fill"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    UNKNOWN = "unknown"  # Broker timeout — needs reconciliation
    EXPIRED = "expired"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class ManagedOrder:
    """
    A fully tracked order with complete audit trail.
    """

    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    client_order_id: str = ""  # Our internal ID (for dedup)
    broker_order_id: str = ""  # Broker's ID (received after send)

    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.LIMIT
    quantity: int = 0
    price: float = 0.0
    stop_price: float = 0.0

    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    average_fill_price: float = 0.0
    remaining_quantity: int = 0

    # Audit trail
    created_at: datetime = field(default_factory=now_ist)
    sent_at: datetime | None = None
    filled_at: datetime | None = None
    cancelled_at: datetime | None = None

    # Risk metadata
    stop_loss: float = 0.0
    target_price: float = 0.0
    setup_type: str = ""  # e.g., "ORB_LONG", "VWAP_PULLBACK"
    confidence_score: float = 0.0

    # Rejection reason (if rejected)
    rejection_reason: str = ""

    # Status history for audit
    status_history: list[dict] = field(default_factory=list)

    def update_status(self, new_status: OrderStatus, reason: str = "") -> None:
        """Update order status with audit trail."""
        old_status = self.status
        self.status = new_status
        self.status_history.append(
            {
                "from": old_status.value,
                "to": new_status.value,
                "at": now_ist().isoformat(),
                "reason": reason,
            }
        )
        if new_status == OrderStatus.SENT:
            self.sent_at = now_ist()
        elif new_status == OrderStatus.FILLED:
            self.filled_at = now_ist()
        elif new_status == OrderStatus.CANCELLED:
            self.cancelled_at = now_ist()

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "status": self.status.value,
            "filled_qty": self.filled_quantity,
            "avg_fill": self.average_fill_price,
            "setup": self.setup_type,
            "confidence": self.confidence_score,
        }


@dataclass
class Position:
    """Current position in an instrument."""

    symbol: str
    quantity: int = 0  # Positive = long, negative = short
    average_cost: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0

    def update_price(self, price: float) -> None:
        self.current_price = price
        if self.quantity != 0:
            self.unrealized_pnl = (price - self.average_cost) * self.quantity


@dataclass
class PreTradeResult:
    """Result of pre-trade compliance check."""

    approved: bool
    reason: str = ""
    adjusted_quantity: int = 0  # May be reduced by risk checks
    warnings: list[str] = field(default_factory=list)


class WashTradeGuard:
    def __init__(self, dedup_ttl_sec: float = 60.0, max_size: int = 10000):
        self._pending = {}  # symbol -> {'BUY': set(), 'SELL': set()}
        self._dedup = OrderedDict()
        self._ttl = dedup_ttl_sec
        self._max = max_size

    def check_wash(self, symbol: str, side: str, strategy_id: str) -> tuple[bool, str]:
        side_upper = side.upper()
        opposite = "SELL" if side_upper == "BUY" else "BUY"
        conflict = self._pending.get(symbol, {}).get(opposite, set())
        if conflict:
            return True, f"wash trade: {conflict} has {opposite} pending"
        if symbol not in self._pending:
            self._pending[symbol] = {"BUY": set(), "SELL": set()}
        self._pending[symbol][side_upper].add(strategy_id)
        return False, "ok"

    def clear(self, symbol: str, side: str, strategy_id: str):
        self._pending.get(symbol, {}).get(side.upper(), set()).discard(strategy_id)

    def is_duplicate(self, order_id: str) -> bool:
        now = time.monotonic()
        while self._dedup and next(iter(self._dedup.values())) < now:
            self._dedup.popitem(last=False)
        while len(self._dedup) >= self._max:
            self._dedup.popitem(last=False)
        if order_id in self._dedup:
            return True
        self._dedup[order_id] = now + self._ttl
        return False


class OrderManagementSystem:
    """
    Institutional-grade Order Management System.

    Handles:
    - Order creation with pre-trade validation
    - Order lifecycle tracking
    - Position management
    - Daily P&L tracking
    - Deduplication
    - Compliance enforcement
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 0.02,
        max_order_value_inr: float = 5_000_000,
        max_order_pct_of_adv: float = 0.10,
        max_price_deviation_atr: float = 3.0,
        max_trades_per_day: int = 50,
        max_single_position_pct: float = 0.05,
        max_sector_exposure_pct: float = 0.25,
        nav: float = 10_000_000,  # ₹1 Crore default NAV
    ):
        # Configuration
        self._max_daily_loss_pct = max_daily_loss_pct
        self._max_order_value_inr = max_order_value_inr
        self._max_order_pct_of_adv = max_order_pct_of_adv
        self._max_price_deviation_atr = max_price_deviation_atr
        self._max_trades_per_day = max_trades_per_day
        self._max_single_position_pct = max_single_position_pct
        self._max_sector_exposure_pct = max_sector_exposure_pct
        self._nav = nav

        # State
        self._orders: dict[str, ManagedOrder] = {}
        self._positions: dict[str, Position] = {}
        self._daily_pnl: float = 0.0
        self._daily_trade_count: int = 0
        self._peak_nav: float = nav
        self._restricted_symbols: set[str] = set()
        self._sector_map: dict[str, str] = {}  # Symbol -> Sector mapping
        self._sector_exposures = defaultdict(float)

        # Fill idempotency
        import threading

        self._processed_fill_ids: set[str] = set()
        self._fill_id_lock = threading.Lock()
        self._order_locks: dict[str, threading.Lock] = {}

        # Recent order tracking for daily reset (bounded to prevent memory leak)
        from collections import deque

        self._recent_order_ids = deque(maxlen=10000)

        # Cross-strategy position intent map
        self._pending_intents: dict[str, dict[str, tuple[OrderSide, int]]] = defaultdict(dict)

        # Deduplication and Wash Trade prevention
        self._wash_guard = WashTradeGuard()

        # Callbacks
        self._on_order_update: Callable | None = None

        # Circuit breaker state
        self._trading_halted = False
        self._halt_reason = ""

        # Compliance Risk Checker
        self.pre_trade_checker = PreTradeChecker(
            PreTradeConfig(
                max_qty_pct_of_adv=self._max_order_pct_of_adv,
                max_price_atr_multiple=self._max_price_deviation_atr,
                max_order_notional=self._max_order_value_inr,
                sebi_client_oi_limit_pct=self._max_single_position_pct,
            )
        )

        logger.info(f"OMS initialized | NAV=₹{nav:,.0f} | MaxDailyLoss={max_daily_loss_pct:.1%}")

    @property
    def is_halted(self) -> bool:
        return self._trading_halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def positions(self) -> dict[str, Position]:
        return self._positions

    def get_position(self, symbol: str) -> Position:
        return self._positions.get(symbol, Position(symbol=symbol))

    @property
    def open_orders(self) -> list[ManagedOrder]:
        return self.get_open_orders()

    def get_open_orders(self, symbol: str | None = None) -> list[ManagedOrder]:
        orders = [
            o
            for o in self._orders.values()
            if o.status
            in (
                OrderStatus.PENDING,
                OrderStatus.VALIDATED,
                OrderStatus.SENT,
                OrderStatus.ACKNOWLEDGED,
                OrderStatus.PARTIAL_FILL,
            )
        ]
        if symbol:
            return [o for o in orders if o.symbol == symbol]
        return orders

    def has_pending_cancels(self, symbol: str) -> bool:
        # In a real system, we'd check for a CANCELLING state.
        # Here we just check if there are any open orders left for the symbol.
        return len(self.get_open_orders(symbol)) > 0

    def set_restricted_symbols(self, symbols: set[str]) -> None:
        """Set the list of restricted/banned symbols."""
        self._restricted_symbols = symbols
        self.pre_trade_checker.load_restricted_list(list(symbols))

    def set_sector_map(self, sector_map: dict[str, str]) -> None:
        """Set symbol-to-sector mapping for concentration checks."""
        self._sector_map = sector_map

    def set_on_order_update(self, callback: Callable) -> None:
        """Register callback for order status updates."""
        self._on_order_update = callback

    def halt_trading(self, reason: str) -> None:
        """Emergency halt all trading."""
        self._trading_halted = True
        self._halt_reason = reason
        logger.critical(f"🚨 TRADING HALTED: {reason}")

    def resume_trading(self) -> None:
        """Resume trading after halt."""
        self._trading_halted = False
        self._halt_reason = ""
        logger.info("✅ Trading RESUMED")

    def halt_symbol(self, symbol: str) -> None:
        """Halt trading for a specific symbol."""
        self._restricted_symbols.add(symbol)
        self.pre_trade_checker.load_restricted_list(list(self._restricted_symbols))
        logger.critical(f"Trading HALTED for symbol: {symbol}")

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: float,
        order_type: OrderType = OrderType.LIMIT,
        stop_loss: float = 0.0,
        target_price: float = 0.0,
        setup_type: str = "",
        confidence_score: float = 0.0,
        client_order_id: str = "",
        adv: float = 0.0,
        last_price: float = 0.0,
        atr: float = 0.0,
    ) -> ManagedOrder:
        """
        Create and validate a new order.

        Returns the order with status VALIDATED (ready for EMS) or REJECTED.
        """
        order = ManagedOrder(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            remaining_quantity=quantity,
            price=price,
            stop_loss=stop_loss,
            target_price=target_price,
            setup_type=setup_type,
            confidence_score=confidence_score,
            client_order_id=client_order_id or str(uuid.uuid4())[:8],
        )

        # Initialize locks
        lock = threading.Lock()
        self._order_locks[order.order_id] = lock
        self._order_locks[order.client_order_id] = lock

        # Run pre-trade checks
        result = self._pre_trade_check(order, adv=adv, last_price=last_price, atr=atr)

        if result.approved:
            order.update_status(OrderStatus.VALIDATED, "Pre-trade checks passed")
            if result.adjusted_quantity != quantity and result.adjusted_quantity > 0:
                order.quantity = result.adjusted_quantity
                order.remaining_quantity = result.adjusted_quantity
                logger.warning(
                    f"[{symbol}] Order quantity adjusted: {quantity} → {result.adjusted_quantity}"
                )

            # Check internal crossing engine to prevent wash trades and net intents
            crossed = self._cross_orders_internally(order)

            self._orders[order.order_id] = order
            if self._on_order_update:
                self._on_order_update(order)

            if crossed:
                logger.info(
                    f"[{symbol}] Order {order.order_id} was completely filled internally via crossing."
                )
                self._order_locks.pop(order.order_id, None)
                self._order_locks.pop(order.client_order_id, None)
            else:
                logger.info(
                    f"[{symbol}] Order VALIDATED | {side.value} {order.quantity}@{price:.2f} | "
                    f"Setup={setup_type} | Confidence={confidence_score:.1f}"
                )
        else:
            order.update_status(OrderStatus.REJECTED, result.reason)
            order.rejection_reason = result.reason
            self._release_borrow_if_needed(order)
            self._orders[order.order_id] = order
            if self._on_order_update:
                self._on_order_update(order)
            # Remove locks
            self._order_locks.pop(order.order_id, None)
            self._order_locks.pop(order.client_order_id, None)
            logger.warning(f"[{symbol}] Order REJECTED: {result.reason}")

        return order

    def _pre_trade_check(
        self,
        order: ManagedOrder,
        adv: float = 0.0,
        last_price: float = 0.0,
        atr: float = 0.0,
    ) -> PreTradeResult:
        """
        Run all pre-trade compliance checks.

        Checks:
        1. Trading halt status
        2. Deduplication
        3. Restricted symbol list
        4. Fat-finger: order value
        5. Fat-finger: price deviation
        6. Fat-finger: ADV participation
        7. Daily trade count limit
        8. Daily loss limit
        9. Position concentration limit
        10. Sector exposure limit
        """
        warnings = []

        # 1. Trading halt
        if self._trading_halted:
            return PreTradeResult(False, f"Trading halted: {self._halt_reason}")

        # 2. Deduplication and Wash Trades
        if self._wash_guard.is_duplicate(order.client_order_id):
            return PreTradeResult(False, f"Duplicate order: {order.client_order_id}")

        wash_issue, wash_msg = self._wash_guard.check_wash(
            order.symbol, order.side.value, order.setup_type or "manual"
        )
        if wash_issue:
            return PreTradeResult(False, wash_msg)

        # 3-6. Institutional Pre-Trade Risk Checks (Fat-finger, restricted, borrow, SEBI, notional)
        self.pre_trade_checker.update_market_data(
            {
                order.symbol: SymbolMarketData(
                    symbol=order.symbol,
                    last_price=last_price,
                    adv_20d=adv,
                    atr_14d=atr,
                    mwpl=0,
                    current_oi=0,
                    is_fno=False,
                )
            }
        )

        checker_result = self.pre_trade_checker.validate_order(
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            price=order.price,
            order_type=order.order_type.value.upper(),
            current_position=self.get_position(order.symbol).quantity,
        )

        if not checker_result.approved:
            return PreTradeResult(False, checker_result.reason)

        adjusted_qty = checker_result.adjusted_qty
        warnings.extend(checker_result.warnings)

        # 7. Daily trade count
        if self._daily_trade_count >= self._max_trades_per_day:
            return PreTradeResult(False, f"Daily trade limit reached ({self._max_trades_per_day})")

        # 8. Daily loss limit
        daily_loss_limit = self._nav * self._max_daily_loss_pct
        if self._daily_pnl < -daily_loss_limit:
            return PreTradeResult(
                False,
                f"Daily loss limit breached: PnL=₹{self._daily_pnl:,.0f}, "
                f"Limit=-₹{daily_loss_limit:,.0f}",
            )

        # 9. Position concentration
        existing_pos = self._positions.get(order.symbol, Position(symbol=order.symbol))
        new_pos_value = abs(existing_pos.quantity + adjusted_qty) * order.price
        max_pos_value = self._nav * self._max_single_position_pct
        if new_pos_value > max_pos_value:
            return PreTradeResult(
                False,
                f"Position in {order.symbol} would be ₹{new_pos_value:,.0f}, "
                f"exceeding {self._max_single_position_pct:.0%} of NAV",
            )

        # 10. Sector exposure
        sector = self._sector_map.get(order.symbol, "unknown")
        if sector != "unknown":
            sector_exposure = self._sector_exposures.get(sector, 0.0)
            new_sector_exposure = sector_exposure + (adjusted_qty * order.price)
            max_sector = self._nav * self._max_sector_exposure_pct
            if new_sector_exposure > max_sector:
                return PreTradeResult(
                    False,
                    f"Sector '{sector}' exposure would be ₹{new_sector_exposure:,.0f}, "
                    f"exceeding {self._max_sector_exposure_pct:.0%} of NAV",
                )

        return PreTradeResult(
            approved=True,
            adjusted_quantity=adjusted_qty,
            warnings=warnings,
        )

    def on_fill(
        self, order_id: str, filled_qty: int, fill_price: float, fill_id: str = None
    ) -> None:
        """Handle a fill event from the EMS/broker with idempotency."""
        if fill_id:
            with self._fill_id_lock:
                if fill_id in self._processed_fill_ids:
                    logger.warning(f"Ignoring duplicate fill_id {fill_id} for order {order_id}")
                    return
                self._processed_fill_ids.add(fill_id)

        order = self._orders.get(order_id)
        if not order:
            logger.error(f"Fill for unknown order {order_id}")
            return

        # Acquire lock
        lock = self._order_locks.get(order_id)
        if not lock:
            lock = threading.Lock()
            self._order_locks[order_id] = lock
            if order.client_order_id:
                self._order_locks[order.client_order_id] = lock

        lock.acquire()
        try:
            old_cost = order.average_fill_price * order.filled_quantity
            new_cost = fill_price * filled_qty

            order.filled_quantity += filled_qty
            order.remaining_quantity -= filled_qty

            if order.filled_quantity > 0:
                order.average_fill_price = (old_cost + new_cost) / order.filled_quantity

            if order.remaining_quantity <= 0:
                order.update_status(OrderStatus.FILLED, f"Fully filled at {fill_price:.2f}")
                self._wash_guard.clear(order.symbol, order.side.value, order.setup_type or "manual")
                # Remove from intents
                self._clear_intent(order)
                # Remove locks
                self._order_locks.pop(order.order_id, None)
                self._order_locks.pop(order.client_order_id, None)
            else:
                order.update_status(
                    OrderStatus.PARTIAL_FILL, f"Partial fill: {filled_qty}@{fill_price:.2f}"
                )
        finally:
            lock.release()

        # Update position
        self._update_position(order.symbol, order.side, filled_qty, fill_price)
        self._daily_trade_count += 1

        if self._on_order_update:
            self._on_order_update(order)

        logger.info(
            f"[{order.symbol}] FILL | {order.side.value} {filled_qty}@{fill_price:.2f} | "
            f"Total filled: {order.filled_quantity}/{order.quantity}"
        )

    def mark_cancelled(self, order_id: str, reason: str = "Cancelled") -> None:
        """Mark an order cancelled and release reserved borrow if required."""
        order = self._orders.get(order_id)
        if not order:
            logger.error(f"Cancel for unknown order {order_id}")
            return

        lock = self._order_locks.get(order_id)
        if lock:
            lock.acquire()
        try:
            order.update_status(OrderStatus.CANCELLED, reason)
            self._wash_guard.clear(order.symbol, order.side.value, order.setup_type or "manual")
            self._release_borrow_if_needed(order)
            if self._on_order_update:
                self._on_order_update(order)
            self._clear_intent(order)
            self._order_locks.pop(order.order_id, None)
            self._order_locks.pop(order.client_order_id, None)
        finally:
            if lock:
                lock.release()

    def _release_borrow_if_needed(self, order: ManagedOrder) -> None:
        """Publish borrow release payload to Redis stream:borrow:release (fire-and-forget)."""
        if order.side != OrderSide.SELL or order.quantity <= 0:
            return

        try:
            import os

            import redis

            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            client = redis.Redis.from_url(redis_url, decode_responses=True)
            payload = {
                "symbol": order.symbol,
                "quantity": str(order.quantity),
                "timestamp": str(time.time()),
            }
            client.xadd("stream:borrow:release", payload)
            logger.info(
                f"Published borrow release to stream:borrow:release for {order.symbol} qty={order.quantity}"
            )
        except Exception as e:
            logger.error(f"Failed to publish borrow release for {order.symbol} to Redis: {e}")

    def _cross_orders_internally(self, order: ManagedOrder) -> bool:
        """
        Check for opposing intents from other strategies.
        If found, cross them internally at the current last price (mid-price).
        Returns True if the order was completely crossed/filled internally.
        """
        symbol = order.symbol
        side = order.side
        qty = order.quantity
        strategy_id = order.setup_type or "manual"

        # Add current intent
        self._pending_intents[symbol][strategy_id] = (side, qty)

        opposing_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY

        # Find opposing intents from other strategies
        crossed_qty = 0
        for other_strat, (other_side, other_qty) in list(self._pending_intents[symbol].items()):
            if other_strat == strategy_id or other_side != opposing_side or other_qty <= 0:
                continue

            # We can cross up to min(qty - crossed_qty, other_qty)
            cross_amount = min(qty - crossed_qty, other_qty)
            if cross_amount <= 0:
                continue

            logger.info(
                f"[CROSS] Internally crossing {cross_amount} of {symbol} between {strategy_id} and {other_strat}"
            )

            # Find the corresponding open order for other_strat to update it
            other_order = None
            for o in self.get_open_orders(symbol):
                if (o.setup_type or "manual") == other_strat and o.side == opposing_side:
                    other_order = o
                    break

            # Get current mid price (or last price)
            md = self.pre_trade_checker._market_data.get(symbol)
            last_price = md.last_price if md else (order.price or 100.0)

            # Execute fill internally for both sides
            if other_order:
                self.on_fill(
                    other_order.order_id,
                    cross_amount,
                    last_price,
                    fill_id=f"CROSS-{uuid.uuid4().hex[:8]}",
                )

            # Update current order fill
            order.filled_quantity += cross_amount
            order.remaining_quantity -= cross_amount
            order.average_fill_price = last_price

            # Update other strategy's intent remaining quantity
            new_other_qty = other_qty - cross_amount
            if new_other_qty > 0:
                self._pending_intents[symbol][other_strat] = (other_side, new_other_qty)
            else:
                self._pending_intents[symbol].pop(other_strat, None)

            crossed_qty += cross_amount
            if crossed_qty >= qty:
                break

        # Update current intent remaining quantity
        remaining_qty = qty - crossed_qty
        if remaining_qty > 0:
            self._pending_intents[symbol][strategy_id] = (side, remaining_qty)
        else:
            self._pending_intents[symbol].pop(strategy_id, None)

        if crossed_qty > 0:
            # Some or all filled internally
            if order.remaining_quantity <= 0:
                order.status = OrderStatus.FILLED
                return True
            else:
                order.status = OrderStatus.PARTIAL_FILL

        return False

    def _clear_intent(self, order: ManagedOrder) -> None:
        symbol = order.symbol
        strategy_id = order.setup_type or "manual"
        if symbol in self._pending_intents and strategy_id in self._pending_intents[symbol]:
            self._pending_intents[symbol].pop(strategy_id, None)

    def _update_position(self, symbol: str, side: OrderSide, qty: int, price: float) -> None:
        """Update position after a fill."""
        if symbol not in self._positions:
            self._positions[symbol] = Position(symbol=symbol)

        pos = self._positions[symbol]
        signed_qty = qty if side == OrderSide.BUY else -qty

        old_market_value = abs(pos.market_value) if pos.quantity != 0 else 0.0

        if pos.quantity == 0:
            pos.quantity = signed_qty
            pos.average_cost = price
        elif (pos.quantity > 0 and side == OrderSide.BUY) or (
            pos.quantity < 0 and side == OrderSide.SELL
        ):
            # Adding to position
            total_cost = pos.average_cost * abs(pos.quantity) + price * qty
            pos.quantity += signed_qty
            pos.average_cost = total_cost / abs(pos.quantity) if pos.quantity != 0 else 0
        else:
            # Reducing/closing/flipping position
            close_qty = min(qty, abs(pos.quantity))
            realized = (price - pos.average_cost) * close_qty
            if pos.quantity < 0:
                realized = -realized  # Short position P&L is inverted
            pos.realized_pnl += realized
            self._daily_pnl += realized

            pos.quantity += signed_qty
            if pos.quantity == 0:
                pos.average_cost = 0.0
            elif abs(signed_qty) > close_qty:
                # Flipped position
                pos.average_cost = price

        # Update last price mark-to-market on fill
        pos.update_price(price)

        # Update sector cache
        sector = self._sector_map.get(symbol, "unknown")
        if sector != "unknown":
            new_market_value = abs(pos.market_value)
            self._sector_exposures[sector] += new_market_value - old_market_value

    def update_market_prices(self, prices: dict[str, float]) -> None:
        """Update all position mark-to-market prices."""
        for symbol, price in prices.items():
            if symbol in self._positions:
                pos = self._positions[symbol]
                old_val = abs(pos.market_value)
                pos.update_price(price)
                new_val = abs(pos.market_value)

                sector = self._sector_map.get(symbol, "unknown")
                if sector != "unknown":
                    self._sector_exposures[sector] += new_val - old_val

    def get_total_exposure(self) -> float:
        """Total gross exposure across all positions."""
        return sum(abs(p.market_value) for p in self._positions.values())

    def get_net_exposure(self) -> float:
        """Net exposure (long - short)."""
        return sum(p.market_value for p in self._positions.values())

    def reset_daily(self) -> None:
        """Reset daily counters for new trading day."""
        self._daily_pnl = 0.0
        self._daily_trade_count = 0
        self._recent_order_ids.clear()
        self._trading_halted = False
        self._halt_reason = ""
        for pos in self._positions.values():
            pos.realized_pnl = 0.0
        logger.info("OMS daily counters RESET")

    def get_all_filled_orders(self) -> list[ManagedOrder]:
        """Return all fully filled orders."""
        return [o for o in self._orders.values() if o.status == OrderStatus.FILLED]

    def inject_external_fill(self, order_id: str, filled_qty: int, fill_price: float) -> None:
        """Inject a fill from an external system/reconciliation."""
        logger.info(f"Injecting external fill for {order_id}: {filled_qty}@{fill_price}")
        self.on_fill(order_id, filled_qty, fill_price)

    def get_status_report(self) -> dict:
        """Generate a full status report."""
        return {
            "halted": self._trading_halted,
            "halt_reason": self._halt_reason,
            "daily_pnl": self._daily_pnl,
            "daily_trades": self._daily_trade_count,
            "open_orders": len(self.open_orders),
            "total_orders": len(self._orders),
            "positions": {
                s: {"qty": p.quantity, "avg_cost": p.average_cost, "pnl": p.unrealized_pnl}
                for s, p in self._positions.items()
                if not p.is_flat
            },
            "gross_exposure": self.get_total_exposure(),
            "net_exposure": self.get_net_exposure(),
            "nav": self._nav,
        }

    def handle_wal_operation(self, operation: str, payload: dict[str, Any]) -> None:
        """Reconstruct state from WAL log entry during crash recovery."""
        if operation == "order_update":
            order_id = payload.get("order_id")
            if not order_id:
                return

            status_val = payload.get("status")
            status = (
                OrderStatus(status_val)
                if status_val in OrderStatus.__members__.values()
                else OrderStatus.PENDING
            )

            side_val = payload.get("side")
            side = (
                OrderSide(side_val) if side_val in OrderSide.__members__.values() else OrderSide.BUY
            )

            type_val = payload.get("type")
            order_type = (
                OrderType(type_val)
                if type_val in OrderType.__members__.values()
                else OrderType.LIMIT
            )

            order = ManagedOrder(
                order_id=order_id,
                symbol=payload.get("symbol", ""),
                side=side,
                order_type=order_type,
                quantity=int(payload.get("quantity", 0)),
                price=float(payload.get("price", 0.0)),
                stop_price=float(payload.get("stop_price", 0.0)),
                status=status,
                filled_quantity=int(payload.get("filled_qty", 0)),
                average_fill_price=float(payload.get("avg_fill", 0.0)),
                remaining_quantity=max(
                    0, int(payload.get("quantity", 0)) - int(payload.get("filled_qty", 0))
                ),
            )
            order.broker_order_id = payload.get("broker_order_id", "")

            # Re-register or update order in local dictionary
            self._orders[order_id] = order

            # If the restored order is filled, update position state
            if order.status == OrderStatus.FILLED and order.filled_quantity > 0:
                self._update_position(
                    order.symbol, order.side, order.filled_quantity, order.average_fill_price
                )

            logger.info(
                f"WAL Replay: Restored order {order_id} | symbol={order.symbol} | status={order.status.value}"
            )
