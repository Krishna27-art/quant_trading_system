"""
Transaction Cost Modeling

Institutional-grade transaction cost modeling for backtesting.
Accounts for commissions, slippage, market impact, and financing costs.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from utils.logger import get_logger

logger = get_logger("transaction_costs")


class TransactionCostModel(BaseModel):
    """Transaction cost model configuration."""

    # Commission
    commission_rate: float = Field(default=0.001, description="Commission rate (0.1%)")
    commission_fixed: float = Field(default=0.0, description="Fixed commission per trade")

    # Slippage
    slippage_rate: float = Field(default=0.001, description="Slippage rate (0.1%)")
    slippage_fixed: float = Field(default=0.0, description="Fixed slippage per share")

    # Market impact
    market_impact_factor: float = Field(default=0.0001, description="Market impact factor")
    volume_participation_rate: float = Field(default=0.1, description="Volume participation rate")

    # Financing costs
    long_borrow_rate: float = Field(default=0.05, description="Long borrowing rate (annual)")
    short_borrow_rate: float = Field(default=0.06, description="Short borrowing rate (annual)")

    # Taxes
    securities_transaction_tax: float = Field(default=0.0, description="Securities transaction tax")

    # STT (Indian market specific)
    stt_rate: float = Field(default=0.00025, description="STT rate (0.025%)")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class TransactionCostCalculator:
    """
    Transaction cost calculator for backtesting.

    Accounts for Indian market-specific costs like STT.
    """

    def __init__(self, model: TransactionCostModel | None = None):
        """Initialize the transaction cost calculator."""
        self.model = model or TransactionCostModel()
        self.logger = logger

    def calculate_buy_cost(self, shares: float, price: float, volume: float | None = None) -> float:
        """
        Calculate total cost for buy transaction.

        Args:
            shares: Number of shares
            price: Price per share
            volume: Trading volume (for market impact calculation)

        Returns:
            Total transaction cost
        """
        trade_value = shares * price

        # Commission
        commission = self._calculate_commission(trade_value, shares)

        # Slippage
        slippage = self._calculate_slippage(shares, price, volume)

        # Market impact
        market_impact = self._calculate_market_impact(trade_value, volume)

        # STT (Indian market)
        stt = self._calculate_stt(trade_value)

        total_cost = commission + slippage + market_impact + stt

        self.logger.debug(
            f"Buy cost: shares={shares:.2f}, price={price:.2f}, "
            f"commission={commission:.2f}, slippage={slippage:.2f}, "
            f"market_impact={market_impact:.2f}, stt={stt:.2f}, total={total_cost:.2f}"
        )

        return total_cost

    def calculate_sell_cost(
        self,
        shares: float,
        price: float,
        volume: float | None = None,
        holding_period_days: float | None = None,
    ) -> float:
        """
        Calculate total cost for sell transaction.

        Args:
            shares: Number of shares
            price: Price per share
            volume: Trading volume (for market impact calculation)
            holding_period_days: Holding period in days (for financing costs)

        Returns:
            Total transaction cost
        """
        trade_value = shares * price

        # Commission
        commission = self._calculate_commission(trade_value, shares)

        # Slippage
        slippage = self._calculate_slippage(shares, price, volume)

        # Market impact
        market_impact = self._calculate_market_impact(trade_value, volume)

        # STT (higher for sell transactions in India)
        stt = self._calculate_stt(trade_value, is_sell=True)

        # Financing costs (if holding period provided)
        financing_cost = 0.0
        if holding_period_days:
            financing_cost = self._calculate_financing_cost(
                trade_value, holding_period_days, is_long=False
            )

        total_cost = commission + slippage + market_impact + stt + financing_cost

        self.logger.debug(
            f"Sell cost: shares={shares:.2f}, price={price:.2f}, "
            f"commission={commission:.2f}, slippage={slippage:.2f}, "
            f"market_impact={market_impact:.2f}, stt={stt:.2f}, "
            f"financing={financing_cost:.2f}, total={total_cost:.2f}"
        )

        return total_cost

    def _calculate_commission(self, trade_value: float, shares: float) -> float:
        """Calculate commission cost."""
        variable_commission = trade_value * self.model.commission_rate
        fixed_commission = shares * self.model.commission_fixed

        return variable_commission + fixed_commission

    def _calculate_slippage(self, shares: float, price: float, volume: float | None) -> float:
        """Calculate slippage cost."""
        trade_value = shares * price

        variable_slippage = trade_value * self.model.slippage_rate
        fixed_slippage = shares * self.model.slippage_fixed

        return variable_slippage + fixed_slippage

    def _calculate_market_impact(self, trade_value: float, volume: float | None) -> float:
        """Calculate market impact cost."""
        if volume is None or volume == 0:
            return 0.0

        # Simple market impact model: impact = factor * (trade_value / volume)^2
        participation_rate = trade_value / volume

        # Cap participation rate
        participation_rate = min(participation_rate, 1.0)

        market_impact = self.model.market_impact_factor * (participation_rate**2) * trade_value

        return market_impact

    def _calculate_stt(self, trade_value: float, is_sell: bool = False) -> float:
        """
        Calculate Securities Transaction Tax (STT).

        Indian market specific:
        - Buy: 0.025% on equity delivery
        - Sell: 0.025% on equity delivery
        - Intraday: 0.025% on sell side only
        """
        stt_rate = self.model.stt_rate

        # STT is typically charged on both buy and sell for delivery trades
        # but at different rates in some cases
        # For simplicity, we use the same rate for both
        return trade_value * stt_rate

    def _calculate_financing_cost(
        self, trade_value: float, holding_period_days: float, is_long: bool = True
    ) -> float:
        """Calculate financing cost."""
        # Annual rate
        annual_rate = self.model.long_borrow_rate if is_long else self.model.short_borrow_rate

        # Daily rate
        daily_rate = annual_rate / 365.0

        # Total financing cost
        financing_cost = trade_value * daily_rate * holding_period_days

        return financing_cost

    def calculate_total_round_trip_cost(
        self,
        shares: float,
        buy_price: float,
        sell_price: float,
        buy_volume: float | None = None,
        sell_volume: float | None = None,
        holding_period_days: float | None = None,
    ) -> dict[str, float]:
        """
        Calculate total round-trip transaction costs.

        Args:
            shares: Number of shares
            buy_price: Buy price per share
            sell_price: Sell price per share
            buy_volume: Buy volume
            sell_volume: Sell volume
            holding_period_days: Holding period in days

        Returns:
            Dictionary with cost breakdown
        """
        buy_cost = self.calculate_buy_cost(shares, buy_price, buy_volume)
        sell_cost = self.calculate_sell_cost(shares, sell_price, sell_volume, holding_period_days)

        total_cost = buy_cost + sell_cost

        # Cost as percentage of trade value
        avg_price = (buy_price + sell_price) / 2
        trade_value = shares * avg_price
        cost_percentage = (total_cost / trade_value) * 100 if trade_value > 0 else 0.0

        return {
            "buy_cost": buy_cost,
            "sell_cost": sell_cost,
            "total_cost": total_cost,
            "cost_percentage": cost_percentage,
            "trade_value": trade_value,
        }

    def estimate_impact_on_returns(
        self,
        expected_return: float,
        shares: float,
        entry_price: float,
        exit_price: float,
        holding_period_days: float,
    ) -> float:
        """
        Estimate impact of transaction costs on expected returns.

        Args:
            expected_return: Expected return (as decimal)
            shares: Number of shares
            entry_price: Entry price
            exit_price: Exit price
            holding_period_days: Holding period in days

        Returns:
            Net return after transaction costs
        """
        # Calculate round-trip costs
        cost_breakdown = self.calculate_total_round_trip_cost(
            shares, entry_price, exit_price, None, None, holding_period_days
        )

        # Calculate gross P&L
        gross_pnl = (exit_price - entry_price) * shares

        # Net P&L after costs
        net_pnl = gross_pnl - cost_breakdown["total_cost"]

        # Net return
        investment = shares * entry_price
        net_return = net_pnl / investment if investment > 0 else 0.0

        return net_return


# Global transaction cost calculator instance
transaction_cost_calculator = TransactionCostCalculator()
