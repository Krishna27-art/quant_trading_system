"""
Alpha Capacity Measurement

Measures capacity constraints for trading strategies.
Critical for institutional scaling - strategies that work with small capital
often fail when scaled due to market impact, slippage, and crowding.

Capacity Formula:
Capacity ∝ ADV × Participation Rate

Metrics:
- ADV Consumption
- Turnover
- Impact Cost
- Slippage
- Crowding Risk
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("capacity_measurement")


@dataclass
class CapacityMetrics:
    """Capacity metrics for a strategy."""

    strategy_name: str
    symbol: str
    adv: float  # Average Daily Volume
    adv_consumption: float  # % of ADV consumed by strategy
    turnover: float  # Portfolio turnover rate
    impact_cost: float  # Estimated market impact cost
    slippage: float  # Estimated slippage
    capacity_estimate: float  # Estimated capacity in currency
    participation_rate: float  # % of ADV participation
    crowding_risk: float  # Crowding risk score (0-1)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "adv": self.adv,
            "adv_consumption": self.adv_consumption,
            "turnover": self.turnover,
            "impact_cost": self.impact_cost,
            "slippage": self.slippage,
            "capacity_estimate": self.capacity_estimate,
            "participation_rate": self.participation_rate,
            "crowding_risk": self.crowding_risk,
        }


class CapacityCalculator:
    """
    Calculates capacity metrics for trading strategies.

    Uses market data and trade execution data to estimate
    strategy capacity and identify scaling constraints.
    """

    def __init__(self):
        """Initialize capacity calculator."""
        self.logger = logger

    def calculate_adv(self, volume_data: pd.Series, window_days: int = 20) -> float:
        """
        Calculate Average Daily Volume (ADV).

        Args:
            volume_data: Daily volume series
            window_days: Window for ADV calculation

        Returns:
            ADV value
        """
        if len(volume_data) < window_days:
            return volume_data.mean() if len(volume_data) > 0 else 0.0

        adv = volume_data.rolling(window=window_days).mean().iloc[-1]
        return adv

    def calculate_adv_consumption(self, trade_volume: float, adv: float) -> float:
        """
        Calculate ADV consumption.

        Args:
            trade_volume: Daily trade volume
            adv: Average Daily Volume

        Returns:
            ADV consumption as percentage
        """
        if adv == 0:
            return 0.0

        consumption = (trade_volume / adv) * 100
        return consumption

    def calculate_turnover(self, positions: pd.DataFrame, window_days: int = 20) -> float:
        """
        Calculate portfolio turnover rate.

        Args:
            positions: Position history DataFrame
            window_days: Window for turnover calculation

        Returns:
            Turnover rate (annualized)
        """
        if len(positions) < 2:
            return 0.0

        # Calculate daily turnover
        position_changes = positions.diff().abs().sum(axis=1)
        portfolio_value = positions.abs().sum(axis=1)

        daily_turnover = (position_changes / (portfolio_value + 1e-6)).mean()

        # Annualize (252 trading days)
        annualized_turnover = daily_turnover * 252

        return annualized_turnover

    def estimate_impact_cost(
        self, trade_size: float, adv: float, participation_rate: float, volatility: float = 0.02
    ) -> float:
        """
        Estimate market impact cost using Almgren-Chriss model.

        Impact Cost = σ * √(Participation Rate)

        Args:
            trade_size: Trade size
            adv: Average Daily Volume
            participation_rate: Participation rate (0-1)
            volatility: Daily volatility

        Returns:
            Estimated impact cost as percentage
        """
        if adv == 0:
            return 0.0

        # Almgren-Chriss model
        impact_cost = volatility * np.sqrt(participation_rate)

        return impact_cost

    def estimate_slippage(self, trade_size: float, adv: float, spread: float = 0.001) -> float:
        """
        Estimate slippage cost.

        Args:
            trade_size: Trade size
            adv: Average Daily Volume
            spread: Bid-ask spread

        Returns:
            Estimated slippage as percentage
        """
        if adv == 0:
            return 0.0

        # Slippage increases with trade size relative to ADV
        size_ratio = trade_size / adv
        slippage = spread * (1 + size_ratio)

        return slippage

    def estimate_capacity(
        self, adv: float, max_participation_rate: float = 0.05, max_impact_cost: float = 0.005
    ) -> float:
        """
        Estimate strategy capacity.

        Capacity = ADV × Max Participation Rate

        Args:
            adv: Average Daily Volume
            max_participation_rate: Maximum acceptable participation rate
            max_impact_cost: Maximum acceptable impact cost

        Returns:
            Estimated capacity in currency units
        """
        # Capacity based on participation rate
        capacity_by_participation = adv * max_participation_rate

        # Capacity based on impact cost constraint
        # Solve for participation rate: σ * √(p) = max_impact_cost
        # √(p) = max_impact_cost / σ
        # p = (max_impact_cost / σ)^2
        volatility = 0.02  # Assume 2% daily volatility
        max_participation_by_impact = (max_impact_cost / volatility) ** 2
        capacity_by_impact = adv * min(max_participation_by_impact, max_participation_rate)

        # Take the more conservative estimate
        capacity = min(capacity_by_participation, capacity_by_impact)

        return capacity

    def calculate_crowding_risk(
        self, adv_consumption: float, strategy_popularity: float = 0.5
    ) -> float:
        """
        Calculate crowding risk score.

        Args:
            adv_consumption: ADV consumption percentage
            strategy_popularity: Strategy popularity (0-1)

        Returns:
            Crowding risk score (0-1)
        """
        # Crowding risk increases with ADV consumption and strategy popularity
        crowding_risk = (adv_consumption / 100) * strategy_popularity

        # Cap at 1
        crowding_risk = min(crowding_risk, 1.0)

        return crowding_risk

    def calculate_all_metrics(
        self,
        strategy_name: str,
        symbol: str,
        trade_volume: float,
        volume_data: pd.Series,
        positions: pd.DataFrame | None = None,
        spread: float = 0.001,
        volatility: float = 0.02,
    ) -> CapacityMetrics:
        """
        Calculate all capacity metrics.

        Args:
            strategy_name: Strategy name
            symbol: Symbol
            trade_volume: Daily trade volume
            volume_data: Historical volume data
            positions: Position history (optional)
            spread: Bid-ask spread
            volatility: Daily volatility

        Returns:
            CapacityMetrics object
        """
        # Calculate ADV
        adv = self.calculate_adv(volume_data)

        # Calculate ADV consumption
        adv_consumption = self.calculate_adv_consumption(trade_volume, adv)

        # Calculate participation rate
        participation_rate = adv_consumption / 100

        # Calculate turnover
        turnover = self.calculate_turnover(positions) if positions is not None else 0.0

        # Estimate impact cost
        impact_cost = self.estimate_impact_cost(trade_volume, adv, participation_rate, volatility)

        # Estimate slippage
        slippage = self.estimate_slippage(trade_volume, adv, spread)

        # Estimate capacity
        capacity_estimate = self.estimate_capacity(adv)

        # Calculate crowding risk
        crowding_risk = self.calculate_crowding_risk(adv_consumption)

        metrics = CapacityMetrics(
            strategy_name=strategy_name,
            symbol=symbol,
            adv=adv,
            adv_consumption=adv_consumption,
            turnover=turnover,
            impact_cost=impact_cost,
            slippage=slippage,
            capacity_estimate=capacity_estimate,
            participation_rate=participation_rate,
            crowding_risk=crowding_risk,
        )

        self.logger.info(
            f"Calculated capacity metrics for {symbol}: "
            f"ADV={adv:.0f}, Consumption={adv_consumption:.2f}%, "
            f"Capacity={capacity_estimate:.0f}"
        )

        return metrics


def calculate_strategy_capacity(
    strategy_name: str, symbol: str, trade_volume: float, volume_data: pd.Series
) -> CapacityMetrics:
    """
    Calculate capacity metrics for a strategy.

    Args:
        strategy_name: Strategy name
        symbol: Symbol
        trade_volume: Daily trade volume
        volume_data: Historical volume data

    Returns:
        CapacityMetrics object
    """
    calculator = CapacityCalculator()
    return calculator.calculate_all_metrics(strategy_name, symbol, trade_volume, volume_data)
