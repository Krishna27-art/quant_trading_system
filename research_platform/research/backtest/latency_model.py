"""
Latency Model

Models execution latency for backtesting validation.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from utils.logger import get_logger

logger = get_logger("latency_model")


class LatencyModelType(str, Enum):
    """Types of latency models."""

    FIXED = "fixed"
    NORMAL = "normal"
    UNIFORM = "uniform"
    TIME_OF_DAY = "time_of_day"
    VOLUME_DEPENDENT = "volume_dependent"


@dataclass
class LatencyParameters:
    """Latency model parameters."""

    base_latency_ms: float = 10.0  # Base latency in milliseconds
    std_dev_ms: float = 5.0  # Standard deviation
    min_latency_ms: float = 1.0  # Minimum latency
    max_latency_ms: float = 100.0  # Maximum latency
    opening_latency_ms: float = 20.0  # Higher latency at open
    closing_latency_ms: float = 15.0  # Higher latency at close
    volume_impact: float = 0.001  # Latency impact from volume

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "base_latency_ms": self.base_latency_ms,
            "std_dev_ms": self.std_dev_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "opening_latency_ms": self.opening_latency_ms,
            "closing_latency_ms": self.closing_latency_ms,
            "volume_impact": self.volume_impact,
        }


class LatencyModel:
    """
    Latency model for backtesting.

    Models execution latency based on time of day,
    market conditions, and order characteristics.
    """

    def __init__(self, model_type: LatencyModelType = LatencyModelType.NORMAL):
        """
        Initialize latency model.

        Args:
            model_type: Type of latency model
        """
        self.model_type = model_type
        self.parameters = LatencyParameters()
        self.logger = logger

        self.logger.info(f"LatencyModel initialized with {model_type.value} model")

    def calculate_latency(
        self,
        timestamp: datetime | None = None,
        market_volume: float | None = None,
        order_size: int | None = None,
    ) -> float:
        """
        Calculate execution latency.

        Args:
            timestamp: Order timestamp
            market_volume: Current market volume
            order_size: Order size

        Returns:
            Latency in milliseconds
        """
        if self.model_type == LatencyModelType.FIXED:
            return self._fixed_latency()
        elif self.model_type == LatencyModelType.NORMAL:
            return self._normal_latency()
        elif self.model_type == LatencyModelType.UNIFORM:
            return self._uniform_latency()
        elif self.model_type == LatencyModelType.TIME_OF_DAY:
            return self._time_of_day_latency(timestamp)
        elif self.model_type == LatencyModelType.VOLUME_DEPENDENT:
            return self._volume_dependent_latency(market_volume, order_size)
        else:
            return self.parameters.base_latency_ms

    def _fixed_latency(self) -> float:
        """
        Fixed latency model.

        Returns:
            Fixed latency in milliseconds
        """
        return self.parameters.base_latency_ms

    def _normal_latency(self) -> float:
        """
        Normal distribution latency model.

        Returns:
            Latency in milliseconds
        """
        latency = np.random.normal(self.parameters.base_latency_ms, self.parameters.std_dev_ms)
        return self._clamp_latency(latency)

    def _uniform_latency(self) -> float:
        """
        Uniform distribution latency model.

        Returns:
            Latency in milliseconds
        """
        latency = np.random.uniform(self.parameters.min_latency_ms, self.parameters.max_latency_ms)
        return latency

    def _time_of_day_latency(self, timestamp: datetime | None) -> float:
        """
        Time-of-day dependent latency model.

        Args:
            timestamp: Order timestamp

        Returns:
            Latency in milliseconds
        """
        if not timestamp:
            return self.parameters.base_latency_ms

        # Get hour of day (0-23)
        hour = timestamp.hour

        # Higher latency at open (9:00-10:00) and close (15:00-15:30)
        if 9 <= hour < 10:
            latency = self.parameters.opening_latency_ms
        elif 15 <= hour < 16:
            latency = self.parameters.closing_latency_ms
        else:
            latency = self.parameters.base_latency_ms

        # Add some randomness
        latency += np.random.normal(0, self.parameters.std_dev_ms * 0.5)

        return self._clamp_latency(latency)

    def _volume_dependent_latency(
        self, market_volume: float | None, order_size: int | None
    ) -> float:
        """
        Volume-dependent latency model.

        Args:
            market_volume: Market volume
            order_size: Order size

        Returns:
            Latency in milliseconds
        """
        latency = self.parameters.base_latency_ms

        # Increase latency with market volume
        if market_volume:
            volume_factor = min(1.0, market_volume / 10000000.0)  # Normalize to 10M
            latency += volume_factor * 10.0

        # Increase latency with order size
        if order_size:
            size_factor = min(1.0, order_size / 10000.0)  # Normalize to 10K
            latency += size_factor * 5.0

        # Add randomness
        latency += np.random.normal(0, self.parameters.std_dev_ms * 0.5)

        return self._clamp_latency(latency)

    def _clamp_latency(self, latency: float) -> float:
        """
        Clamp latency to min/max bounds.

        Args:
            latency: Calculated latency

        Returns:
            Clamped latency
        """
        return max(self.parameters.min_latency_ms, min(self.parameters.max_latency_ms, latency))

    def update_parameters(self, parameters: LatencyParameters):
        """
        Update latency parameters.

        Args:
            parameters: New parameters
        """
        self.parameters = parameters
        self.logger.info(f"Latency parameters updated: {parameters.to_dict()}")

    def validate_latency(self, actual_latency: float, tolerance: float = 2.0) -> dict[str, Any]:
        """
        Validate actual latency against model.

        Args:
            actual_latency: Actual latency observed in milliseconds
            tolerance: Tolerance multiplier (default 2x)

        Returns:
            Validation result
        """
        expected_latency = self.parameters.base_latency_ms
        tolerance_latency = expected_latency * tolerance

        is_valid = actual_latency <= tolerance_latency

        return {
            "expected_latency_ms": expected_latency,
            "actual_latency_ms": actual_latency,
            "tolerance_ms": tolerance_latency,
            "is_valid": is_valid,
            "message": "Latency within tolerance" if is_valid else "Latency exceeds tolerance",
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
