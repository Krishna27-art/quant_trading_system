from typing import Any

import numpy as np


class MarketSimulator:
    """
    Realistic backtesting/simulation fill engine for Research OS v2.
    Implements:
    - 1-candle/bar execution delay (signal at t executed at t+1 Open).
    - Bid-Ask Spread cost.
    - Slippage using ADV model.
    - Capped volume/partial fills under liquidity constraints.
    """

    def __init__(
        self,
        spread_bps: float = 2.0,
        commission_pct: float = 0.05,
        max_bar_vol_pct: float = 0.10,
        slippage_coef: float = 0.1,
    ):
        self.spread_bps = spread_bps
        self.commission_pct = commission_pct / 100.0
        self.max_bar_vol_pct = max_bar_vol_pct
        self.slippage_coef = slippage_coef

    def simulate_fill(
        self,
        side: str,
        target_qty: int,
        next_bar_open: float,
        next_bar_high: float,
        next_bar_low: float,
        next_bar_volume: float,
        adv: float = 1000000.0,
    ) -> dict[str, Any]:
        """
        Simulates an order fill on the next bar.
        Buys execute at the Ask (higher), Sells at the Bid (lower).
        """
        side = side.upper()

        # 1. Liquidity constraints: Cap execution quantity at max_bar_vol_pct of next bar's volume
        max_allowed_qty = max(0, int(next_bar_volume * self.max_bar_vol_pct))
        filled_qty = min(target_qty, max_allowed_qty)

        if filled_qty <= 0:
            return {
                "filled_qty": 0,
                "execution_price": 0.0,
                "slippage": 0.0,
                "commission": 0.0,
                "status": "REJECTED_NO_LIQUIDITY",
            }

        # 2. Spread adjustment (Buys executed at Ask, Sells at Bid)
        half_spread_pct = (self.spread_bps / 10000.0) / 2.0
        base_price = next_bar_open

        if side == "BUY":
            price_with_spread = base_price * (1.0 + half_spread_pct)
        else:
            price_with_spread = base_price * (1.0 - half_spread_pct)

        # 3. Slippage Model: square-root slippage based on volume participation
        # Slippage = base_price * slippage_coef * sqrt(filled_qty / ADV)
        participation = filled_qty / max(adv, 1.0)
        slippage_pct = self.slippage_coef * np.sqrt(participation)
        slippage_val = base_price * slippage_pct

        if side == "BUY":
            execution_price = price_with_spread + slippage_val
        else:
            execution_price = price_with_spread - slippage_val

        # Verify boundary checks (execution price must be within high/low range)
        execution_price = max(next_bar_low, min(next_bar_high, execution_price))

        # 4. Commission Calculation
        commission = execution_price * filled_qty * self.commission_pct

        status = "FILLED" if filled_qty == target_qty else "PARTIALLY_FILLED"

        return {
            "filled_qty": filled_qty,
            "execution_price": round(execution_price, 2),
            "slippage": round(slippage_val * filled_qty, 2),
            "commission": round(commission, 2),
            "status": status,
        }
