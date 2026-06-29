"""
Advanced Execution Algorithms: VWAP and Implementation Shortfall (IS).
"""

from prediction_intelligence.order import OrderSide


class VWAPExecutionAlgo:
    def __init__(self, target_volume_curve: list[float]):
        """
        target_volume_curve: List of floats representing the expected percentage
        of daily volume traded in each time bucket (e.g., 5-minute bins).
        """
        self.volume_curve = target_volume_curve
        self.current_bucket = 0

    def get_slice_quantity(self, total_qty: float) -> float:
        """
        Determines the quantity to execute in the current time bucket based on the historical volume curve.
        """
        if self.current_bucket < len(self.volume_curve):
            pct = self.volume_curve[self.current_bucket]
            self.current_bucket += 1
            return total_qty * pct
        return 0.0


class ImplementationShortfallAlgo:
    def __init__(self, arrival_price: float, risk_aversion: float = 0.5):
        self.arrival_price = arrival_price
        self.risk_aversion = risk_aversion

    def determine_aggression(self, current_price: float, side: OrderSide) -> float:
        """
        Returns a scaling factor [0.0 to 1.0] representing how aggressively to cross the spread.
        If price moves away from arrival price, we speed up execution (if risk averse).
        """
        if side == OrderSide.BUY:
            slippage = (current_price - self.arrival_price) / self.arrival_price
        else:
            slippage = (self.arrival_price - current_price) / self.arrival_price

        # If slippage is positive (price moved against us), increase aggression based on risk_aversion
        aggression = min(1.0, max(0.1, 0.5 + (slippage * 100 * self.risk_aversion)))
        return aggression
