"""
Slippage Model

Models slippage for backtesting validation.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from utils.logger import get_logger

logger = get_logger("slippage_model")


class SlippageModelType(str, Enum):
    """Types of slippage models."""

    LINEAR = "linear"
    SQUARE_ROOT = "square_root"
    PERCENTAGE = "percentage"
    FIXED = "fixed"
    VOLATILITY_ADJUSTED = "volatility_adjusted"


@dataclass
class SlippageParameters:
    """Slippage model parameters."""

    base_slippage: float = 0.0005  # 0.05%
    volume_impact: float = 0.0001  # Volume impact coefficient
    volatility_impact: float = 0.0002  # Volatility impact coefficient
    time_decay: float = 0.0  # Time decay coefficient
    min_slippage: float = 0.0
    max_slippage: float = 0.01  # 1% max

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "base_slippage": self.base_slippage,
            "volume_impact": self.volume_impact,
            "volatility_impact": self.volatility_impact,
            "time_decay": self.time_decay,
            "min_slippage": self.min_slippage,
            "max_slippage": self.max_slippage,
        }


class SlippageModel:
    """
    Slippage model for backtesting.

    Models slippage based on order size, market conditions,
    and execution timing.
    """

    def __init__(self, model_type: SlippageModelType = SlippageModelType.LINEAR):
        """
        Initialize slippage model.

        Args:
            model_type: Type of slippage model
        """
        self.model_type = model_type
        self.parameters = SlippageParameters()
        self.logger = logger

        self.logger.info(f"SlippageModel initialized with {model_type.value} model")

    def calculate_slippage(
        self,
        order_quantity: int,
        market_volume: float,
        market_price: float,
        volatility: float | None = None,
        time_since_open: float | None = None,
    ) -> float:
        """
        Calculate slippage for an order.

        Args:
            order_quantity: Order quantity
            market_volume: Current market volume
            market_price: Current market price
            volatility: Market volatility (optional)
            time_since_open: Time since market open in seconds (optional)

        Returns:
            Slippage as percentage (e.g., 0.001 for 0.1%)
        """
        if self.model_type == SlippageModelType.LINEAR:
            return self._linear_slippage(order_quantity, market_volume)
        elif self.model_type == SlippageModelType.SQUARE_ROOT:
            return self._square_root_slippage(order_quantity, market_volume)
        elif self.model_type == SlippageModelType.PERCENTAGE:
            return self._percentage_slippage(order_quantity, market_volume)
        elif self.model_type == SlippageModelType.FIXED:
            return self._fixed_slippage()
        elif self.model_type == SlippageModelType.VOLATILITY_ADJUSTED:
            return self._volatility_adjusted_slippage(
                order_quantity, market_volume, volatility, time_since_open
            )
        else:
            return self.parameters.base_slippage

    def _linear_slippage(self, order_quantity: int, market_volume: float) -> float:
        """
        Linear slippage model.

        Args:
            order_quantity: Order quantity
            market_volume: Market volume

        Returns:
            Slippage percentage
        """
        # Slippage increases linearly with order size relative to volume
        volume_ratio = order_quantity / market_volume if market_volume > 0 else 0.0
        slippage = self.parameters.base_slippage + (volume_ratio * self.parameters.volume_impact)

        return self._clamp_slippage(slippage)

    def _square_root_slippage(self, order_quantity: int, market_volume: float) -> float:
        """
        Square root slippage model.

        Args:
            order_quantity: Order quantity
            market_volume: Market volume

        Returns:
            Slippage percentage
        """
        # Slippage increases with square root of volume ratio
        volume_ratio = order_quantity / market_volume if market_volume > 0 else 0.0
        slippage = self.parameters.base_slippage + (
            np.sqrt(volume_ratio) * self.parameters.volume_impact
        )

        return self._clamp_slippage(slippage)

    def _percentage_slippage(self, order_quantity: int, market_volume: float) -> float:
        """
        Percentage slippage model.

        Args:
            order_quantity: Order quantity
            market_volume: Market volume

        Returns:
            Slippage percentage
        """
        # Slippage as percentage of order size
        volume_ratio = order_quantity / market_volume if market_volume > 0 else 0.0
        slippage = self.parameters.base_slippage * (1 + volume_ratio)

        return self._clamp_slippage(slippage)

    def _fixed_slippage(self) -> float:
        """
        Fixed slippage model.

        Returns:
            Fixed slippage percentage
        """
        return self.parameters.base_slippage

    def _volatility_adjusted_slippage(
        self,
        order_quantity: int,
        market_volume: float,
        volatility: float | None,
        time_since_open: float | None,
    ) -> float:
        """
        Volatility-adjusted slippage model.

        Args:
            order_quantity: Order quantity
            market_volume: Market volume
            volatility: Market volatility
            time_since_open: Time since market open

        Returns:
            Slippage percentage
        """
        # Base slippage from volume
        volume_ratio = order_quantity / market_volume if market_volume > 0 else 0.0
        base_slippage = self.parameters.base_slippage + (
            volume_ratio * self.parameters.volume_impact
        )

        # Adjust for volatility
        if volatility:
            volatility_adjustment = volatility * self.parameters.volatility_impact
            base_slippage += volatility_adjustment

        # Adjust for time decay (slippage decreases over time)
        if time_since_open:
            time_hours = time_since_open / 3600.0
            time_adjustment = np.exp(-self.parameters.time_decay * time_hours)
            base_slippage *= time_adjustment

        return self._clamp_slippage(base_slippage)

    def _clamp_slippage(self, slippage: float) -> float:
        """
        Clamp slippage to min/max bounds.

        Args:
            slippage: Calculated slippage

        Returns:
            Clamped slippage
        """
        return max(self.parameters.min_slippage, min(self.parameters.max_slippage, slippage))

    def update_parameters(self, parameters: SlippageParameters):
        """
        Update slippage parameters.

        Args:
            parameters: New parameters
        """
        self.parameters = parameters
        self.logger.info(f"Slippage parameters updated: {parameters.to_dict()}")

    def validate_slippage(self, actual_slippage: float, tolerance: float = 0.5) -> dict[str, Any]:
        """
        Validate actual slippage against model.

        Args:
            actual_slippage: Actual slippage observed
            tolerance: Tolerance percentage (default 50%)

        Returns:
            Validation result
        """
        expected_slippage = self.parameters.base_slippage
        deviation = abs(actual_slippage - expected_slippage)
        tolerance_slippage = expected_slippage * (1 + tolerance)

        is_valid = actual_slippage <= tolerance_slippage

        return {
            "expected_slippage": expected_slippage,
            "actual_slippage": actual_slippage,
            "deviation": deviation,
            "tolerance": tolerance,
            "is_valid": is_valid,
            "message": "Slippage within tolerance" if is_valid else "Slippage exceeds tolerance",
        }

    def get_status(self) -> dict[str, Any]:
        """
        Get model status.

        Returns:
            Status dictionary
        """
        return {
            "model_type": self.model_type.value,
            "parameters": self.parameters.to_dict(),
            "timestamp": datetime.utcnow().isoformat(),
        }
