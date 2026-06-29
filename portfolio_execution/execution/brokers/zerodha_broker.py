import asyncio
import os
from typing import Any

from kiteconnect import KiteConnect

from portfolio_execution.execution.base import BaseExecutionAdapter
from portfolio_execution.oms import ManagedOrder
from utils.logger import get_logger

logger = get_logger("zerodha_broker_adapter")


class ZerodhaBrokerAdapter(BaseExecutionAdapter):
    IS_LIVE = True

    def __init__(self):
        self.api_key = os.getenv("ZERODHA_API_KEY")
        self.access_token = os.getenv("ZERODHA_ACCESS_TOKEN")
        self.kite = None
        self._connected = False

    async def connect(self) -> bool:
        if not self.api_key or not self.access_token:
            logger.error("Missing ZERODHA_API_KEY or ZERODHA_ACCESS_TOKEN env variables.")
            self._connected = False
            return False
        try:
            self.kite = KiteConnect(api_key=self.api_key)
            self.kite.set_access_token(self.access_token)
            # Verify connectivity by fetching profile info
            await asyncio.to_thread(self.kite.profile)
            self._connected = True
            logger.info("Successfully connected to Zerodha Kite Connect.")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Zerodha: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        self.kite = None
        self._connected = False
        logger.info("Disconnected from Zerodha.")

    async def heartbeat(self) -> bool:
        if not self._connected or not self.kite:
            return False
        try:
            await asyncio.to_thread(self.kite.profile)
            return True
        except Exception:
            return False

    async def get_quote(self, symbols: list[str]) -> dict[str, float]:
        if not self._connected or not self.kite:
            raise RuntimeError("Broker not connected")
        try:
            formatted_symbols = [f"NSE:{sym}" if ":" not in sym else sym for sym in symbols]
            quotes = await asyncio.to_thread(self.kite.quote, formatted_symbols)
            return {
                sym.split(":")[-1]: float(quotes[sym]["last_price"])
                for sym in formatted_symbols
                if sym in quotes
            }
        except Exception as e:
            logger.error(f"Failed to get quote from Zerodha: {e}")
            raise

    async def get_margin(self) -> dict[str, float]:
        if not self._connected or not self.kite:
            raise RuntimeError("Broker not connected")
        try:
            margins = await asyncio.to_thread(self.kite.margins)
            equity_margin = margins.get("equity", {})
            return {
                "available": float(equity_margin.get("net", 0.0)),
                "used": float(equity_margin.get("utilised", 0.0)),
            }
        except Exception as e:
            logger.error(f"Failed to get margins from Zerodha: {e}")
            raise

    async def get_positions(self) -> list[dict[str, Any]]:
        if not self._connected or not self.kite:
            raise RuntimeError("Broker not connected")
        try:
            positions_data = await asyncio.to_thread(self.kite.positions)
            net_positions = positions_data.get("net", [])
            return [
                {
                    "symbol": pos["tradingsymbol"],
                    "quantity": int(pos["quantity"]),
                    "average_price": float(pos["average_price"]),
                }
                for pos in net_positions
            ]
        except Exception as e:
            logger.error(f"Failed to get positions from Zerodha: {e}")
            raise

    async def place_order(self, order: ManagedOrder) -> str:
        if not self._connected or not self.kite:
            raise RuntimeError("Broker not connected")
        try:
            tx_type = (
                self.kite.TRANSACTION_TYPE_BUY
                if order.side.name.lower() == "buy"
                else self.kite.TRANSACTION_TYPE_SELL
            )
            ord_type = (
                self.kite.ORDER_TYPE_LIMIT
                if order.order_type.name.lower() == "limit"
                else self.kite.ORDER_TYPE_MARKET
            )

            broker_order_id = await asyncio.to_thread(
                self.kite.place_order,
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NSE,
                tradingsymbol=order.symbol,
                transaction_type=tx_type,
                quantity=order.quantity,
                product=self.kite.PRODUCT_MIS,
                order_type=ord_type,
                price=float(order.price) if ord_type == self.kite.ORDER_TYPE_LIMIT else None,
                validity=self.kite.VALIDITY_DAY,
            )
            return broker_order_id
        except Exception as e:
            logger.error(f"Failed to place order at Zerodha: {e}")
            raise

    async def cancel_order(self, broker_order_id: str) -> bool:
        if not self._connected or not self.kite:
            raise RuntimeError("Broker not connected")
        try:
            await asyncio.to_thread(
                self.kite.cancel_order, variety=self.kite.VARIETY_REGULAR, order_id=broker_order_id
            )
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {broker_order_id} at Zerodha: {e}")
            return False

    async def get_order_status(self, broker_order_id: str) -> dict[str, Any]:
        if not self._connected or not self.kite:
            raise RuntimeError("Broker not connected")
        try:
            history = await asyncio.to_thread(self.kite.order_history, broker_order_id)
            if not history:
                return {"status": "REJECTED", "fill_price": 0.0}
            last_update = history[-1]
            status_str = str(last_update.get("status", "")).upper()

            if status_str == "COMPLETE":
                std_status = "COMPLETE"
            elif status_str in ("REJECTED", "CANCELLED"):
                std_status = status_str
            else:
                std_status = "OPEN"

            return {
                "status": std_status,
                "fill_price": float(last_update.get("average_price", 0.0)),
            }
        except Exception as e:
            logger.error(f"Failed to get order status for {broker_order_id} at Zerodha: {e}")
            raise

    async def exit_all_positions(self) -> list[str]:
        positions = await self.get_positions()
        exit_order_ids = []
        for pos in positions:
            qty = pos["quantity"]
            if qty == 0:
                continue
            side = "SELL" if qty > 0 else "BUY"
            tx_type = (
                self.kite.TRANSACTION_TYPE_SELL
                if side == "SELL"
                else self.kite.TRANSACTION_TYPE_BUY
            )

            try:
                order_id = await asyncio.to_thread(
                    self.kite.place_order,
                    variety=self.kite.VARIETY_REGULAR,
                    exchange=self.kite.EXCHANGE_NSE,
                    tradingsymbol=pos["symbol"],
                    transaction_type=tx_type,
                    quantity=abs(qty),
                    product=self.kite.PRODUCT_MIS,
                    order_type=self.kite.ORDER_TYPE_MARKET,
                    validity=self.kite.VALIDITY_DAY,
                )
                exit_order_ids.append(order_id)
            except Exception as e:
                logger.error(f"Failed to place exit order for {pos['symbol']}: {e}")
        return exit_order_ids

    async def cancel_all_pending_orders(self) -> list[str]:
        if not self._connected or not self.kite:
            return []
        try:
            orders = await asyncio.to_thread(self.kite.orders)
            cancelled_ids = []
            for order in orders:
                status = str(order.get("status", "")).upper()
                if status in ("OPEN", "TRIGGER PENDING"):
                    order_id = order["order_id"]
                    try:
                        await asyncio.to_thread(
                            self.kite.cancel_order,
                            variety=order.get("variety", self.kite.VARIETY_REGULAR),
                            order_id=order_id,
                        )
                        cancelled_ids.append(order_id)
                    except Exception as e:
                        logger.error(f"Failed to cancel pending order {order_id}: {e}")
            return cancelled_ids
        except Exception as e:
            logger.error(f"Failed to fetch orders for cancellation: {e}")
            return []
