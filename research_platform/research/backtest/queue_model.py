"""
Queue Position Model

Models queue position for order execution validation.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from utils.logger import get_logger

logger = get_logger("queue_model")


class QueueModelType(str, Enum):
    """Types of queue models."""

    FIFO = "fifo"
    PRIORITY = "priority"
    PRO_RATA = "pro_rata"
    RANDOM = "random"
    SIZE_PRIORITY = "size_priority"


@dataclass
class QueueParameters:
    """Queue model parameters."""

    base_position: int = 5  # Base queue position
    queue_depth: int = 100  # Total queue depth
    priority_factor: float = 0.1  # Priority weighting
    size_impact: float = 0.001  # Size impact on position
    time_decay: float = 0.01  # Time decay of position

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "base_position": self.base_position,
            "queue_depth": self.queue_depth,
            "priority_factor": self.priority_factor,
            "size_impact": self.size_impact,
            "time_decay": self.time_decay,
        }


@dataclass
class QueueState:
    """Current queue state."""

    symbol: str
    side: str  # buy/sell
    queue_depth: int
    current_position: int
    orders_in_queue: int
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "queue_depth": self.queue_depth,
            "current_position": self.current_position,
            "orders_in_queue": self.orders_in_queue,
            "timestamp": self.timestamp.isoformat(),
        }


class QueuePositionModel:
    """
    Queue position model for backtesting.

    Models queue position based on order priority,
    size, and timing.
    """

    def __init__(self, model_type: QueueModelType = QueueModelType.PRIORITY):
        """
        Initialize queue position model.

        Args:
            model_type: Type of queue model
        """
        self.model_type = model_type
        self.parameters = QueueParameters()
        self.logger = logger

        # Queue state tracking
        self._queue_states: dict[str, QueueState] = {}

        self.logger.info(f"QueuePositionModel initialized with {model_type.value} model")

    def calculate_queue_position(
        self,
        symbol: str,
        side: str,
        order_size: int,
        timestamp: datetime | None = None,
        priority: float | None = None,
    ) -> int:
        """
        Calculate queue position for an order.

        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            order_size: Order size
            timestamp: Order timestamp
            priority: Order priority (0-1, higher is better)

        Returns:
            Queue position (1-based, 1 is first)
        """
        if self.model_type == QueueModelType.FIFO:
            return self._fifo_position(symbol, side)
        elif self.model_type == QueueModelType.PRIORITY:
            return self._priority_position(symbol, side, priority)
        elif self.model_type == QueueModelType.PRO_RATA:
            return self._pro_rata_position(symbol, side, order_size)
        elif self.model_type == QueueModelType.RANDOM:
            return self._random_position()
        elif self.model_type == QueueModelType.SIZE_PRIORITY:
            return self._size_priority_position(symbol, side, order_size, priority)
        else:
            return self.parameters.base_position

    def _fifo_position(self, symbol: str, side: str) -> int:
        """
        FIFO queue position.

        Args:
            symbol: Trading symbol
            side: Order side

        Returns:
            Queue position
        """
        # Update queue state
        queue_state = self._get_or_create_queue_state(symbol, side)
        queue_state.current_position += 1
        queue_state.orders_in_queue += 1

        return min(queue_state.current_position, self.parameters.queue_depth)

    def _priority_position(self, symbol: str, side: str, priority: float | None) -> int:
        """
        Priority-based queue position.

        Args:
            symbol: Trading symbol
            side: Order side
            priority: Order priority

        Returns:
            Queue position
        """
        queue_state = self._get_or_create_queue_state(symbol, side)

        # Higher priority = better position (lower number)
        if priority is not None:
            # Priority adjustment: higher priority moves closer to front
            position_adjustment = int(
                priority * self.parameters.queue_depth * self.parameters.priority_factor
            )
            position = max(1, self.parameters.base_position - position_adjustment)
        else:
            position = queue_state.current_position + 1

        queue_state.current_position = position
        queue_state.orders_in_queue += 1

        return min(position, self.parameters.queue_depth)

    def _pro_rata_position(self, symbol: str, side: str, order_size: int) -> int:
        """
        Pro-rata queue position.

        Args:
            symbol: Trading symbol
            side: Order side
            order_size: Order size

        Returns:
            Queue position
        """
        queue_state = self._get_or_create_queue_state(symbol, side)

        # Larger orders get worse positions
        size_factor = min(1.0, order_size / 10000.0)  # Normalize to 10K
        position = self.parameters.base_position + int(
            size_factor * self.parameters.queue_depth * self.parameters.size_impact
        )

        queue_state.current_position = position
        queue_state.orders_in_queue += 1

        return min(position, self.parameters.queue_depth)

    def _random_position(self) -> int:
        """
        Random queue position.

        Returns:
            Queue position
        """
        return np.random.randint(1, self.parameters.queue_depth + 1)

    def _size_priority_position(
        self, symbol: str, side: str, order_size: int, priority: float | None
    ) -> int:
        """
        Size-priority combined queue position.

        Args:
            symbol: Trading symbol
            side: Order side
            order_size: Order size
            priority: Order priority

        Returns:
            Queue position
        """
        queue_state = self._get_or_create_queue_state(symbol, side)

        # Combine size and priority factors
        size_factor = min(1.0, order_size / 10000.0)
        priority_factor = (1.0 - priority) if priority else 0.5

        position = self.parameters.base_position + int(
            (
                size_factor * self.parameters.size_impact
                + priority_factor * self.parameters.priority_factor
            )
            * self.parameters.queue_depth
        )

        queue_state.current_position = position
        queue_state.orders_in_queue += 1

        return min(position, self.parameters.queue_depth)

    def _get_or_create_queue_state(self, symbol: str, side: str) -> QueueState:
        """
        Get or create queue state for symbol/side.

        Args:
            symbol: Trading symbol
            side: Order side

        Returns:
            Queue state
        """
        key = f"{symbol}_{side}"

        if key not in self._queue_states:
            self._queue_states[key] = QueueState(
                symbol=symbol,
                side=side,
                queue_depth=self.parameters.queue_depth,
                current_position=self.parameters.base_position,
                orders_in_queue=0,
                timestamp=datetime.utcnow(),
            )

        return self._queue_states[key]

    def update_parameters(self, parameters: QueueParameters):
        """
        Update queue parameters.

        Args:
            parameters: New parameters
        """
        self.parameters = parameters
        self.logger.info(f"Queue parameters updated: {parameters.to_dict()}")

    def get_queue_state(self, symbol: str, side: str) -> QueueState | None:
        """
        Get current queue state.

        Args:
            symbol: Trading symbol
            side: Order side

        Returns:
            Queue state or None
        """
        key = f"{symbol}_{side}"
        return self._queue_states.get(key)

    def validate_queue_position(
        self, actual_position: int, expected_position: int, tolerance: int = 10
    ) -> dict[str, Any]:
        """
        Validate actual queue position against model.

        Args:
            actual_position: Actual queue position
            expected_position: Expected queue position
            tolerance: Position tolerance

        Returns:
            Validation result
        """
        deviation = abs(actual_position - expected_position)
        is_valid = deviation <= tolerance

        return {
            "expected_position": expected_position,
            "actual_position": actual_position,
            "deviation": deviation,
            "tolerance": tolerance,
            "is_valid": is_valid,
            "message": (
                "Queue position within tolerance"
                if is_valid
                else "Queue position exceeds tolerance"
            ),
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
            "active_queues": len(self._queue_states),
            "timestamp": datetime.utcnow().isoformat(),
        }
