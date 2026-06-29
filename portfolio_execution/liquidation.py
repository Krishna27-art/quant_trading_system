import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum


class LiquidationState(Enum):
    IDLE = "idle"
    CANCELLING = "cancelling"
    CONFIRMING = "confirming"
    LIQUIDATING = "liquidating"


class SafeLiquidationEngine:
    def __init__(self, oms, broker, headers, wal):
        self.oms = oms
        self.broker = broker
        self.headers = headers
        self.wal = wal
        self._state = LiquidationState.IDLE
        self._executor = ThreadPoolExecutor(max_workers=5)

    async def liquidate_symbol(self, symbol: str, reason: str):
        if self._state != LiquidationState.IDLE:
            return

        self._state = LiquidationState.CANCELLING
        self.wal.write("LIQUIDATION_START", {"symbol": symbol, "reason": reason})

        # STEP 1: Cancel every open order first
        open_orders = self.oms.get_open_orders(symbol)
        cancel_tasks = []
        for order in open_orders:
            cancel_tasks.append(self._cancel_order(order.client_order_id))

        if cancel_tasks:
            await asyncio.gather(*cancel_tasks, return_exceptions=True)

        # STEP 2: Wait for cancels to confirm — max 5 seconds
        self._state = LiquidationState.CONFIRMING
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            still_open = self.oms.get_open_orders(symbol)
            if not still_open:
                break
            await asyncio.sleep(0.1)

        # STEP 3: Get NET position AFTER all cancels confirmed
        self._state = LiquidationState.LIQUIDATING
        position = self.oms.get_position(symbol)
        net_qty = position.net_qty if position else 0

        if abs(net_qty) < 1:
            self.wal.write("LIQUIDATION_SKIP", {"symbol": symbol, "reason": "already_flat"})
            self._state = LiquidationState.IDLE
            return

        # STEP 4: ONE market order for net position only
        side = "SELL" if net_qty > 0 else "BUY"
        qty = abs(int(net_qty))

        self.wal.write("LIQUIDATION_ORDER", {"symbol": symbol, "side": side, "qty": qty})

        try:
            loop = asyncio.get_running_loop()
            resp = await loop.run_in_executor(
                self._executor, lambda: self.broker.market_order_eq(symbol, qty, side, self.headers)
            )
            self.wal.write("LIQUIDATION_DONE", {"response": str(resp)})
        except Exception as e:
            self.wal.write("LIQUIDATION_FAIL", {"error": str(e)})
        finally:
            self._state = LiquidationState.IDLE

    async def _cancel_order(self, order_id: str):
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                self._executor, lambda: self.broker.cancel_order(order_id, self.headers)
            )
        except Exception as e:
            self.wal.write("CANCEL_FAILED", {"order_id": order_id, "error": str(e)})
