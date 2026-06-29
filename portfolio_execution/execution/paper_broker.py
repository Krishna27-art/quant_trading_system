"""
Paper Broker Adapter

A local simulator for paper trading and backtesting that accurately
implements the BaseExecutionAdapter interface. It tracks positions and validates
borrows for short selling natively.
"""

import asyncio
import uuid
from typing import Any

from portfolio_execution.execution.base import BaseExecutionAdapter
from portfolio_execution.oms import ManagedOrder
from utils.logger import get_logger
from utils.time_utils import now_ist

logger = get_logger("paper_broker")


def check_borrow_secured(symbol: str, qty: int) -> bool:
    """Verify if a borrow reservation exists in the DB (fail-closed)."""
    try:
        from database.db_sync import SessionLocal
        from risk_governance.pre_trade.borrow_manager import BorrowStatus

        if not SessionLocal:
            return False
        from sqlalchemy import text

        with SessionLocal() as db:
            result = db.execute(
                text("""
                    SELECT SUM(qty)
                    FROM borrow_reservations
                    WHERE symbol = :symbol AND status = :reserved
                """),
                {
                    "symbol": symbol,
                    "reserved": BorrowStatus.RESERVED.value,
                },
            ).fetchone()
            reserved_qty = result[0] if result and result[0] is not None else 0
            if reserved_qty >= qty:
                return True
            logger.error(
                f"Borrow verification failed at broker adapter: Symbol={symbol}, RequestedQty={qty}, ReservedQty={reserved_qty}"
            )
            return False
    except Exception as e:
        logger.critical(f"Database error during broker-level borrow check for {symbol}: {e}")
        return False


class PaperBrokerAdapter(BaseExecutionAdapter):
    """
    Paper trading broker that simulates fills instantly.
    """

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def heartbeat(self):
        return True

    IS_LIVE = False

    def __init__(self, slippage_bps: float = 5.0, fill_rate: float = 1.0):
        self._slippage_bps = slippage_bps
        self._fill_rate = fill_rate
        self._connected = False
        self._orders: dict[str, dict] = {}

    async def get_quote(self, symbols: list[str]) -> dict[str, float]:
        """Fetch current prices for symbols. Simulates price for paper broker."""
        return dict.fromkeys(symbols, 100.0)

    async def get_margin(self) -> dict[str, float]:
        """Fetch margin/collateral availability."""
        return {"available": 10_000_000.0}

    async def place_order(self, order: ManagedOrder) -> str:
        """Submit a new order to the paper broker."""
        # Second-layer check for short selling / borrow bypass
        if order.side.name.lower() in ("sell", "short"):
            long_qty = 0
            try:
                positions = await self.get_positions()
                for p in positions:
                    if p["symbol"] == order.symbol:
                        long_qty = p["quantity"]
                        break
            except Exception as e:
                logger.error(f"Failed to fetch positions for borrow check: {e}")

            short_needed = order.quantity - max(0, long_qty)
            if short_needed > 0:
                if not check_borrow_secured(order.symbol, short_needed):
                    logger.critical(
                        f"ABORT ORDER: Naked Short Selling Detected in PaperBrokerAdapter! Symbol={order.symbol}, Side={order.side.name}, Qty={order.quantity}, LongQty={long_qty}, ShortNeeded={short_needed}"
                    )
                    raise RuntimeError("Naked short selling detected")

        broker_id = f"PAPER-{uuid.uuid4().hex[:8].upper()}"

        # Simulate slippage
        price = order.price or 100.0
        slippage = price * (self._slippage_bps / 10000.0)
        fill_price = price + slippage if order.side.name.lower() == "buy" else price - slippage

        self._orders[broker_id] = {
            "symbol": order.symbol,
            "side": order.side.name.upper(),
            "quantity": order.quantity,
            "fill_price": round(fill_price, 2),
            "status": "filled",
            "timestamp": now_ist(),
        }

        # Simulate network latency
        await asyncio.sleep(0.01)
        return broker_id

    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an open order."""
        await asyncio.sleep(0.01)
        if broker_order_id in self._orders:
            self._orders[broker_order_id]["status"] = "cancelled"
            return True
        return False

    async def get_order_status(self, broker_order_id: str) -> dict[str, Any] | None:
        """Get current status of an order."""
        await asyncio.sleep(0.005)
        return self._orders.get(broker_order_id)

    async def get_positions(self) -> list[dict[str, Any]]:
        """Get current positions."""
        await asyncio.sleep(0.01)
        positions = {}
        for _order_id, o in self._orders.items():
            if o.get("status") == "filled":
                sym = o["symbol"]
                side = o["side"]
                qty = o["quantity"]
                signed_qty = qty if side == "BUY" else -qty
                price = o["fill_price"]

                if sym not in positions:
                    positions[sym] = {"symbol": sym, "quantity": 0, "average_price": 0.0}

                pos = positions[sym]
                old_qty = pos["quantity"]
                new_qty = old_qty + signed_qty

                if new_qty != 0:
                    if (
                        old_qty == 0
                        or (old_qty > 0 and signed_qty > 0)
                        or (old_qty < 0 and signed_qty < 0)
                    ):
                        pos["average_price"] = (
                            pos["average_price"] * abs(old_qty) + price * qty
                        ) / abs(new_qty)
                else:
                    pos["average_price"] = 0.0
                pos["quantity"] = new_qty

        return [p for p in positions.values() if p["quantity"] != 0]

    async def cancel_all_pending_orders(self) -> list[str]:
        """Emergency order cancellation."""
        await asyncio.sleep(0.01)
        return []

    async def exit_all_positions(self) -> list[str]:
        """Emergency liquidator."""
        await asyncio.sleep(0.01)
        positions = await self.get_positions()
        exit_ids = []
        for p in positions:
            sym = p["symbol"]
            qty = p["quantity"]
            if qty == 0:
                continue
            side = "BUY" if qty < 0 else "SELL"
            # create mock order
            from portfolio_execution.oms import ManagedOrder, OrderSide, OrderType

            order = ManagedOrder(
                symbol=sym,
                side=OrderSide[side],
                quantity=abs(qty),
                price=100.0,
                order_type=OrderType.MARKET,
            )
            exit_id = await self.place_order(order)
            exit_ids.append(exit_id)
        return exit_ids
