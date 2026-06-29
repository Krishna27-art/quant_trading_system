"""
Non-Linear Slippage Model with Dynamic Spread Execution Halts.
"""

from utils.logger import get_logger

logger = get_logger(__name__)


class NonLinearSlippageModel:
    def __init__(self, max_spread_bps: float = 20.0):
        self.max_spread_bps = max_spread_bps

    def estimate_slippage(self, order_qty: float, daily_volume: float, volatility: float) -> float:
        """
        Almgren-Chriss inspired non-linear slippage estimate.
        Impact = volatility * sqrt(order_qty / daily_volume)
        """
        if daily_volume <= 0:
            return float("inf")

        participation_rate = order_qty / daily_volume
        impact = volatility * (participation_rate**0.5)
        return impact

    def check_dynamic_spread_halt(self, bid: float, ask: float) -> bool:
        """
        Dynamically halts execution if the real-time spread widens beyond the maximum threshold.
        Returns True if execution should be halted.
        """
        if bid <= 0:
            return True

        spread_bps = ((ask - bid) / bid) * 10000
        if spread_bps > self.max_spread_bps:
            logger.warning(
                f"Spread widened to {spread_bps:.1f} bps. Halting execution (Max: {self.max_spread_bps})."
            )
            return True
        return False
