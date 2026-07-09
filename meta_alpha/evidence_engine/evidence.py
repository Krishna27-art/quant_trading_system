"""
Evidence

Represents a piece of evidence from an alpha source.
Everything becomes evidence instead of direct signals.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

from utils.logger import get_logger

logger = get_logger("meta_alpha.evidence_engine")


class SignalDirection(Enum):
    """Signal direction enumeration."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class EvidenceCategory(Enum):
    """Evidence category enumeration."""
    TREND = "trend"
    MOMENTUM = "momentum"
    RELATIVE_STRENGTH = "relative_strength"
    LIQUIDITY = "liquidity"
    OPTIONS = "options"
    FUNDAMENTALS = "fundamentals"
    MACRO = "macro"
    SENTIMENT = "sentiment"
    MARKET_BREADTH = "market_breadth"
    SECTOR_ROTATION = "sector_rotation"
    CONDITIONAL_INTERACTION = "conditional_interaction"


@dataclass
class Evidence:
    """
    Represents a piece of evidence from an alpha source.
    
    Fields:
        source: Name of the evidence source (e.g., "RSI", "MACD", "LSTM")
        factor_name: Specific factor name
        category: Category of evidence
        signal_direction: Bullish/Bearish/Neutral
        strength: Signal strength (0 to 1)
        confidence: Confidence in this evidence (0 to 1)
        timestamp: When this evidence was generated
        metadata: Additional metadata
        quality_score: Optional quality score (0 to 100)
        weight: Optional weight for fusion
    """
    source: str
    factor_name: str
    category: str
    signal_direction: str
    strength: float
    confidence: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    quality_score: Optional[float] = None
    weight: Optional[float] = None
    
    def validate(self) -> tuple[bool, List[str]]:
        """
        Validate evidence fields.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check strength
        if not (0.0 <= self.strength <= 1.0):
            errors.append(f"Strength must be between 0 and 1, got {self.strength}")
        
        # Check confidence
        if not (0.0 <= self.confidence <= 1.0):
            errors.append(f"Confidence must be between 0 and 1, got {self.confidence}")
        
        # Check signal direction
        valid_directions = [d.value for d in SignalDirection]
        if self.signal_direction not in valid_directions:
            errors.append(f"Invalid signal direction: {self.signal_direction}")
        
        # Check category
        valid_categories = [c.value for c in EvidenceCategory]
        if self.category not in valid_categories:
            errors.append(f"Invalid category: {self.category}")
        
        # Check quality score if present
        if self.quality_score is not None and not (0.0 <= self.quality_score <= 100.0):
            errors.append(f"Quality score must be between 0 and 100, got {self.quality_score}")
        
        # Check weight if present
        if self.weight is not None and not (0.0 <= self.weight <= 1.0):
            errors.append(f"Weight must be between 0 and 1, got {self.weight}")
        
        return len(errors) == 0, errors
    
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize evidence to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "source": self.source,
            "factor_name": self.factor_name,
            "category": self.category,
            "signal_direction": self.signal_direction,
            "strength": round(self.strength, 4),
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "quality_score": round(self.quality_score, 4) if self.quality_score is not None else None,
            "weight": round(self.weight, 4) if self.weight is not None else None,
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "Evidence":
        """
        Deserialize evidence from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Evidence object
        """
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        return cls(
            source=data["source"],
            factor_name=data["factor_name"],
            category=data["category"],
            signal_direction=data["signal_direction"],
            strength=data["strength"],
            confidence=data["confidence"],
            timestamp=timestamp,
            metadata=data.get("metadata"),
            quality_score=data.get("quality_score"),
            weight=data.get("weight"),
        )
    
    @classmethod
    def from_factor(
        cls,
        source: str,
        factor_name: str,
        category: str,
        factor_value: float,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Evidence":
        """
        Create evidence from factor value.
        
        Args:
            source: Evidence source
            factor_name: Factor name
            category: Evidence category
            factor_value: Raw factor value
            confidence: Confidence in the factor
            metadata: Optional metadata
            
        Returns:
            Evidence object
        """
        # Convert factor value to signal direction
        if factor_value > 0.5:
            direction = SignalDirection.BULLISH.value
        elif factor_value < 0.5:
            direction = SignalDirection.BEARISH.value
        else:
            direction = SignalDirection.NEUTRAL.value
        
        # Calculate strength based on distance from 0.5
        strength = abs(factor_value - 0.5) * 2
        
        return cls(
            source=source,
            factor_name=factor_name,
            category=category,
            signal_direction=direction,
            strength=strength,
            confidence=confidence,
            timestamp=datetime.now(),
            metadata=metadata,
        )
    
    def get_bullish_score(self) -> float:
        """
        Get bullish score (0 to 1).
        
        Returns:
            Bullish score
        """
        if self.signal_direction == SignalDirection.BULLISH.value:
            return self.strength * self.confidence
        elif self.signal_direction == SignalDirection.BEARISH.value:
            return -self.strength * self.confidence
        else:
            return 0.0
    
    def get_description(self) -> str:
        """
        Get human-readable description.
        
        Returns:
            Description string
        """
        parts = []
        parts.append(f"{self.source}")
        parts.append(f"{self.signal_direction.upper()}")
        parts.append(f"Strength: {self.strength:.2f}")
        parts.append(f"Confidence: {self.confidence:.2f}")
        return " | ".join(parts)
    
    def is_bullish(self) -> bool:
        """Check if evidence is bullish."""
        return self.signal_direction == SignalDirection.BULLISH.value
    
    def is_bearish(self) -> bool:
        """Check if evidence is bearish."""
        return self.signal_direction == SignalDirection.BEARISH.value
    
    def is_neutral(self) -> bool:
        """Check if evidence is neutral."""
        return self.signal_direction == SignalDirection.NEUTRAL.value
