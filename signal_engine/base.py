"""
Signal Engine Base Classes

Defines core data structures and base classes for signal generation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger("signal_engine.base")


class SignalDirection(Enum):
    """Signal direction."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class SignalCategory(Enum):
    """Signal categories."""
    TECHNICAL = "technical"
    VOLUME = "volume"
    OPTIONS = "options"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    MACRO = "macro"
    SECTOR = "sector"
    MARKET = "market"


@dataclass
class Signal:
    """
    Represents a single trading signal.
    
    Every signal explains itself with:
    - Name
    - Category
    - Score (0-100)
    - Direction
    - Confidence (0-100%)
    - Reason (explanation)
    - Raw values used
    """
    name: str
    category: SignalCategory
    score: float
    direction: SignalDirection
    confidence: float
    reason: str
    raw_values: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "category": self.category.value,
            "score": round(self.score, 2),
            "direction": self.direction.value,
            "confidence": round(self.confidence, 2),
            "reason": self.reason,
            "raw_values": self.raw_values,
            "timestamp": self.timestamp.isoformat(),
        }
    
    def is_strong(self, min_score: float = 70.0) -> bool:
        """Check if signal is strong."""
        return self.score >= min_score
    
    def is_bullish(self) -> bool:
        """Check if signal is bullish."""
        return self.direction == SignalDirection.BULLISH
    
    def is_bearish(self) -> bool:
        """Check if signal is bearish."""
        return self.direction == SignalDirection.BEARISH


@dataclass
class SignalSet:
    """
    Collection of signals for a single stock at a point in time.
    
    Contains signals from all categories:
    - Technical
    - Volume
    - Options
    - Fundamental
    - Sentiment
    - Macro
    - Sector
    """
    symbol: str
    timestamp: datetime
    signals: Dict[SignalCategory, Signal] = field(default_factory=dict)
    
    def add_signal(self, signal: Signal) -> None:
        """Add a signal to the set."""
        self.signals[signal.category] = signal
    
    def get_signal(self, category: SignalCategory) -> Optional[Signal]:
        """Get signal by category."""
        return self.signals.get(category)
    
    def get_average_score(self) -> float:
        """Calculate average score across all signals."""
        if not self.signals:
            return 0.0
        return sum(s.score for s in self.signals.values()) / len(self.signals)
    
    def get_dominant_direction(self) -> SignalDirection:
        """Get dominant direction across all signals."""
        if not self.signals:
            return SignalDirection.NEUTRAL
        
        bullish_count = sum(1 for s in self.signals.values() if s.is_bullish())
        bearish_count = sum(1 for s in self.signals.values() if s.is_bearish())
        
        if bullish_count > bearish_count:
            return SignalDirection.BULLISH
        elif bearish_count > bullish_count:
            return SignalDirection.BEARISH
        else:
            return SignalDirection.NEUTRAL
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "signals": {cat.value: sig.to_dict() for cat, sig in self.signals.items()},
            "average_score": round(self.get_average_score(), 2),
            "dominant_direction": self.get_dominant_direction().value,
        }


@dataclass
class SignalFilterResult:
    """Result of signal filtering."""
    passed: bool
    reason: str
    filtered_signals: Dict[SignalCategory, Signal]
    overall_score: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "reason": self.reason,
            "filtered_signals": {cat.value: sig.to_dict() for cat, sig in self.filtered_signals.items()},
            "overall_score": round(self.overall_score, 2),
        }


@dataclass
class SignalRanking:
    """Ranked signal set for comparison."""
    symbol: str
    signal_set: SignalSet
    rank: int
    overall_score: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "rank": self.rank,
            "overall_score": round(self.overall_score, 2),
            "signal_set": self.signal_set.to_dict(),
        }


class BaseSignalGenerator:
    """
    Base class for all signal generators.
    
    Every signal generator must:
    1. Inherit from this class
    2. Implement generate() method
    3. Return a Signal object
    4. Provide clear reasoning
    """
    
    def __init__(self, name: str, category: SignalCategory):
        """
        Initialize signal generator.
        
        Args:
            name: Name of the signal generator
            category: Category of the signal
        """
        self.name = name
        self.category = category
        self._logger = get_logger(f"signal_engine.{category.value}.{name}")
    
    def generate(self, data: Dict[str, Any]) -> Signal:
        """
        Generate a signal from input data.
        
        Args:
            data: Dictionary containing required features
            
        Returns:
            Signal object
        """
        raise NotImplementedError("Subclasses must implement generate()")
    
    def _calculate_score(
        self,
        bullish_indicators: int,
        bearish_indicators: int,
        neutral_indicators: int = 0,
    ) -> tuple[float, SignalDirection]:
        """
        Calculate score and direction from indicator counts.
        
        Args:
            bullish_indicators: Number of bullish indicators
            bearish_indicators: Number of bearish indicators
            neutral_indicators: Number of neutral indicators
            
        Returns:
            Tuple of (score, direction)
        """
        total = bullish_indicators + bearish_indicators + neutral_indicators
        
        if total == 0:
            return 50.0, SignalDirection.NEUTRAL
        
        # Calculate score (0-100)
        bullish_ratio = bullish_indicators / total
        score = bullish_ratio * 100
        
        # Determine direction
        if score >= 60:
            direction = SignalDirection.BULLISH
        elif score <= 40:
            direction = SignalDirection.BEARISH
        else:
            direction = SignalDirection.NEUTRAL
        
        return score, direction
    
    def _validate_required_fields(self, data: Dict[str, Any], required_fields: List[str]) -> None:
        """
        Validate that required fields are present in data.
        
        Args:
            data: Input data dictionary
            required_fields: List of required field names
            
        Raises:
            ValueError: If required fields are missing
        """
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
