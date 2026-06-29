"""
Circuit Limits Detection

Detects upper/lower circuit limits in Indian markets.
NSE has circuit limits of 5%, 10%, 20% based on stock category.
"""

from datetime import date
from datetime import date as dt_date
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from utils.logger import get_logger

logger = get_logger("circuit_limits")


class CircuitLimitType(str, Enum):
    """Types of circuit limits."""

    UPPER = "upper"
    LOWER = "lower"


class CircuitCategory(str, Enum):
    """NSE circuit limit categories."""

    CATEGORY_A = "category_a"  # 20% limit
    CATEGORY_B = "category_b"  # 10% limit
    CATEGORY_C = "category_c"  # 5% limit
    CATEGORY_D = "category_d"  # 2% limit


class CircuitEvent(BaseModel):
    """Circuit limit event."""

    symbol: str = Field(..., description="Stock symbol")
    date: dt_date = Field(..., description="Date of circuit event")
    circuit_type: CircuitLimitType = Field(..., description="Upper or lower circuit")
    limit_percentage: float = Field(..., description="Circuit limit percentage")
    close_price: float = Field(..., description="Close price")
    circuit_price: float = Field(..., description="Circuit trigger price")

    # Volume impact
    volume: float = Field(default=0.0, description="Volume on circuit day")
    avg_volume_20d: float = Field(default=0.0, description="Average 20-day volume")
    volume_ratio: float = Field(default=0.0, description="Volume ratio to average")

    # Price impact
    next_day_return: float | None = Field(None, description="Next day return")

    class Config:
        json_encoders = {date: lambda v: v.isoformat()}


class CircuitLimitsDetector:
    """
    Detector for circuit limits in Indian markets.

    NSE has circuit limits based on stock category:
    - Category A: 20%
    - Category B: 10%
    - Category C: 5%
    - Category D: 2%
    """

    def __init__(self):
        """Initialize the circuit limits detector."""
        self.logger = logger

        # Circuit limits by category
        self.circuit_limits = {
            CircuitCategory.CATEGORY_A: 0.20,
            CircuitCategory.CATEGORY_B: 0.10,
            CircuitCategory.CATEGORY_C: 0.05,
            CircuitCategory.CATEGORY_D: 0.02,
        }

        # Default category (most stocks are Category A)
        self.default_category = CircuitCategory.CATEGORY_A

    def detect_circuits(
        self,
        price_data: pd.DataFrame,
        category_mapping: dict[str, CircuitCategory] | None = None,
    ) -> list[CircuitEvent]:
        """
        Detect circuit limit events.

        Args:
            price_data: DataFrame with OHLCV data (date, symbol, open, high, low, close, volume)
            category_mapping: Optional mapping of symbols to circuit categories

        Returns:
            List of circuit events
        """
        self.logger.info("Detecting circuit limit events")

        circuit_events = []

        # Group by symbol
        for symbol in price_data["symbol"].unique():
            symbol_data = price_data[price_data["symbol"] == symbol].copy()
            symbol_data = symbol_data.sort_values("date")

            # Get circuit limit for this symbol
            category = (
                category_mapping.get(symbol, self.default_category)
                if category_mapping
                else self.default_category
            )
            limit = self.circuit_limits[category]

            # Detect circuits
            symbol_circuits = self._detect_symbol_circuits(symbol_data, symbol, limit)
            circuit_events.extend(symbol_circuits)

        self.logger.info(f"Detected {len(circuit_events)} circuit events")

        return circuit_events

    def _detect_symbol_circuits(
        self, symbol_data: pd.DataFrame, symbol: str, limit: float
    ) -> list[CircuitEvent]:
        """Detect circuits for a single symbol."""
        circuits = []

        for i in range(len(symbol_data)):
            row = symbol_data.iloc[i]

            if i == 0:
                continue

            prev_row = symbol_data.iloc[i - 1]
            prev_close = prev_row["close"]

            # Calculate circuit prices
            upper_circuit = prev_close * (1 + limit)
            lower_circuit = prev_close * (1 - limit)

            # Check if hit upper circuit
            if row["high"] >= upper_circuit * 0.999:  # Allow small tolerance
                # Calculate average volume
                avg_volume = self._calculate_avg_volume(symbol_data, i, 20)

                circuit_event = CircuitEvent(
                    symbol=symbol,
                    date=pd.to_datetime(row["date"]).date(),
                    circuit_type=CircuitLimitType.UPPER,
                    limit_percentage=limit,
                    close_price=row["close"],
                    circuit_price=upper_circuit,
                    volume=row["volume"],
                    avg_volume_20d=avg_volume,
                    volume_ratio=row["volume"] / avg_volume if avg_volume > 0 else 0.0,
                )

                # Calculate next day return if possible
                if i < len(symbol_data) - 1:
                    next_row = symbol_data.iloc[i + 1]
                    circuit_event.next_day_return = (next_row["close"] - row["close"]) / row[
                        "close"
                    ]

                circuits.append(circuit_event)

            # Check if hit lower circuit
            elif row["low"] <= lower_circuit * 1.001:  # Allow small tolerance
                avg_volume = self._calculate_avg_volume(symbol_data, i, 20)

                circuit_event = CircuitEvent(
                    symbol=symbol,
                    date=pd.to_datetime(row["date"]).date(),
                    circuit_type=CircuitLimitType.LOWER,
                    limit_percentage=limit,
                    close_price=row["close"],
                    circuit_price=lower_circuit,
                    volume=row["volume"],
                    avg_volume_20d=avg_volume,
                    volume_ratio=row["volume"] / avg_volume if avg_volume > 0 else 0.0,
                )

                if i < len(symbol_data) - 1:
                    next_row = symbol_data.iloc[i + 1]
                    circuit_event.next_day_return = (next_row["close"] - row["close"]) / row[
                        "close"
                    ]

                circuits.append(circuit_event)

        return circuits

    def _calculate_avg_volume(self, data: pd.DataFrame, index: int, window: int) -> float:
        """Calculate average volume for window."""
        start_index = max(0, index - window)
        window_data = data.iloc[start_index:index]

        if len(window_data) == 0:
            return 0.0

        return window_data["volume"].mean()

    def get_circuit_frequency(
        self, circuit_events: list[CircuitEvent], symbol: str | None = None
    ) -> dict[str, Any]:
        """
        Get circuit frequency statistics.

        Args:
            circuit_events: List of circuit events
            symbol: Optional symbol to filter by

        Returns:
            Dictionary with frequency statistics
        """
        events = [e for e in circuit_events if e.symbol == symbol] if symbol else circuit_events

        if not events:
            return {
                "total_circuits": 0,
                "upper_circuits": 0,
                "lower_circuits": 0,
                "avg_volume_ratio": 0.0,
            }

        upper_circuits = sum(1 for e in events if e.circuit_type == CircuitLimitType.UPPER)
        lower_circuits = sum(1 for e in events if e.circuit_type == CircuitLimitType.LOWER)

        avg_volume_ratio = np.mean([e.volume_ratio for e in events if e.volume_ratio > 0])

        return {
            "total_circuits": len(events),
            "upper_circuits": upper_circuits,
            "lower_circuits": lower_circuits,
            "avg_volume_ratio": avg_volume_ratio,
        }

    def get_circuit_impact_analysis(self, circuit_events: list[CircuitEvent]) -> dict[str, Any]:
        """
        Analyze impact of circuit events.

        Args:
            circuit_events: List of circuit events

        Returns:
            Dictionary with impact analysis
        """
        if not circuit_events:
            return {}

        # Filter events with next day returns
        events_with_returns = [e for e in circuit_events if e.next_day_return is not None]

        if not events_with_returns:
            return {}

        # Calculate next day returns
        next_day_returns = [e.next_day_return for e in events_with_returns]

        # Separate by circuit type
        upper_returns = [
            e.next_day_return
            for e in events_with_returns
            if e.circuit_type == CircuitLimitType.UPPER
        ]
        lower_returns = [
            e.next_day_return
            for e in events_with_returns
            if e.circuit_type == CircuitLimitType.LOWER
        ]

        return {
            "total_events_with_returns": len(events_with_returns),
            "avg_next_day_return": np.mean(next_day_returns),
            "avg_upper_circuit_return": np.mean(upper_returns) if upper_returns else 0.0,
            "avg_lower_circuit_return": np.mean(lower_returns) if lower_returns else 0.0,
            "upper_circuit_reversal_rate": (
                sum(1 for r in upper_returns if r < 0) / len(upper_returns)
                if upper_returns
                else 0.0
            ),
            "lower_circuit_reversal_rate": (
                sum(1 for r in lower_returns if r > 0) / len(lower_returns)
                if lower_returns
                else 0.0
            ),
        }


# Global circuit limits detector instance
circuit_limits_detector = CircuitLimitsDetector()
