"""
Evidence Builder

Builds evidence objects with fluent interface.
Provides convenience methods for creating evidence from various sources.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

from meta_alpha.evidence_engine.evidence import Evidence, SignalDirection, EvidenceCategory
from utils.logger import get_logger

logger = get_logger("meta_alpha.evidence_engine")


class EvidenceBuilder:
    """
    Builds evidence objects with fluent interface.
    
    Example:
        builder = EvidenceBuilder()
        evidence = builder.source("RSI").factor_name("RSI_14").category("momentum")\
            .bullish().strength(0.8).confidence(0.75).build()
    """
    
    def __init__(self):
        """Initialize evidence builder."""
        self._source: Optional[str] = None
        self._factor_name: Optional[str] = None
        self._category: Optional[str] = None
        self._signal_direction: Optional[str] = None
        self._strength: Optional[float] = None
        self._confidence: Optional[float] = None
        self._timestamp: Optional[datetime] = None
        self._metadata: Optional[Dict[str, Any]] = None
        self._quality_score: Optional[float] = None
        self._weight: Optional[float] = None
        self._logger = get_logger("meta_alpha.evidence_engine")
    
    def source(self, value: str) -> "EvidenceBuilder":
        """Set evidence source."""
        self._source = value
        return self
    
    def factor_name(self, value: str) -> "EvidenceBuilder":
        """Set factor name."""
        self._factor_name = value
        return self
    
    def category(self, value: str) -> "EvidenceBuilder":
        """Set evidence category."""
        self._category = value
        return self
    
    def bullish(self) -> "EvidenceBuilder":
        """Set signal direction to bullish."""
        self._signal_direction = SignalDirection.BULLISH.value
        return self
    
    def bearish(self) -> "EvidenceBuilder":
        """Set signal direction to bearish."""
        self._signal_direction = SignalDirection.BEARISH.value
        return self
    
    def neutral(self) -> "EvidenceBuilder":
        """Set signal direction to neutral."""
        self._signal_direction = SignalDirection.NEUTRAL.value
        return self
    
    def direction(self, value: str) -> "EvidenceBuilder":
        """Set signal direction."""
        self._signal_direction = value
        return self
    
    def strength(self, value: float) -> "EvidenceBuilder":
        """Set signal strength."""
        self._strength = value
        return self
    
    def confidence(self, value: float) -> "EvidenceBuilder":
        """Set confidence."""
        self._confidence = value
        return self
    
    def timestamp(self, value: datetime) -> "EvidenceBuilder":
        """Set timestamp."""
        self._timestamp = value
        return self
    
    def metadata(self, value: Dict[str, Any]) -> "EvidenceBuilder":
        """Set metadata."""
        self._metadata = value
        return self
    
    def quality_score(self, value: float) -> "EvidenceBuilder":
        """Set quality score."""
        self._quality_score = value
        return self
    
    def weight(self, value: float) -> "EvidenceBuilder":
        """Set weight."""
        self._weight = value
        return self
    
    def build(self) -> Evidence:
        """
        Build the evidence.
        
        Returns:
            Evidence object
        """
        if self._source is None:
            raise ValueError("Source is required")
        if self._factor_name is None:
            raise ValueError("Factor name is required")
        if self._category is None:
            raise ValueError("Category is required")
        if self._signal_direction is None:
            raise ValueError("Signal direction is required")
        if self._strength is None:
            raise ValueError("Strength is required")
        if self._confidence is None:
            raise ValueError("Confidence is required")
        
        if self._timestamp is None:
            self._timestamp = datetime.now()
        
        evidence = Evidence(
            source=self._source,
            factor_name=self._factor_name,
            category=self._category,
            signal_direction=self._signal_direction,
            strength=self._strength,
            confidence=self._confidence,
            timestamp=self._timestamp,
            metadata=self._metadata,
            quality_score=self._quality_score,
            weight=self._weight,
        )
        
        # Validate
        is_valid, errors = evidence.validate()
        if not is_valid:
            self._logger.warning(f"Built invalid evidence: {errors}")
        
        return evidence
    
    def reset(self) -> "EvidenceBuilder":
        """Reset builder to initial state."""
        self._source = None
        self._factor_name = None
        self._category = None
        self._signal_direction = None
        self._strength = None
        self._confidence = None
        self._timestamp = None
        self._metadata = None
        self._quality_score = None
        self._weight = None
        return self


class EvidenceFactory:
    """
    Factory for creating common evidence patterns.
    """
    
    @staticmethod
    def from_rsi(rsi_value: float, confidence: float = 0.7) -> Evidence:
        """Create evidence from RSI value."""
        if rsi_value > 70:
            direction = SignalDirection.BEARISH.value
            strength = (rsi_value - 70) / 30
        elif rsi_value < 30:
            direction = SignalDirection.BULLISH.value
            strength = (30 - rsi_value) / 30
        else:
            direction = SignalDirection.NEUTRAL.value
            strength = 0.0
        
        return Evidence(
            source="RSI",
            factor_name="RSI_14",
            category=EvidenceCategory.MOMENTUM.value,
            signal_direction=direction,
            strength=min(strength, 1.0),
            confidence=confidence,
            timestamp=datetime.now(),
            metadata={"rsi_value": rsi_value},
        )
    
    @staticmethod
    def from_macd(macd_value: float, signal_value: float, confidence: float = 0.7) -> Evidence:
        """Create evidence from MACD value."""
        if macd_value > signal_value:
            direction = SignalDirection.BULLISH.value
            strength = min(abs(macd_value - signal_value) / 0.5, 1.0)
        elif macd_value < signal_value:
            direction = SignalDirection.BEARISH.value
            strength = min(abs(macd_value - signal_value) / 0.5, 1.0)
        else:
            direction = SignalDirection.NEUTRAL.value
            strength = 0.0
        
        return Evidence(
            source="MACD",
            factor_name="MACD_12_26_9",
            category=EvidenceCategory.MOMENTUM.value,
            signal_direction=direction,
            strength=strength,
            confidence=confidence,
            timestamp=datetime.now(),
            metadata={"macd_value": macd_value, "signal_value": signal_value},
        )
    
    @staticmethod
    def from_moving_average(price: float, ma_short: float, ma_long: float, confidence: float = 0.8) -> Evidence:
        """Create evidence from moving average crossover."""
        if price > ma_short > ma_long:
            direction = SignalDirection.BULLISH.value
            strength = 0.8
        elif price < ma_short < ma_long:
            direction = SignalDirection.BEARISH.value
            strength = 0.8
        else:
            direction = SignalDirection.NEUTRAL.value
            strength = 0.3
        
        return Evidence(
            source="MA",
            factor_name="MA_Crossover",
            category=EvidenceCategory.TREND.value,
            signal_direction=direction,
            strength=strength,
            confidence=confidence,
            timestamp=datetime.now(),
            metadata={"price": price, "ma_short": ma_short, "ma_long": ma_long},
        )
    
    @staticmethod
    def from_volume(current_volume: float, avg_volume: float, confidence: float = 0.6) -> Evidence:
        """Create evidence from volume analysis."""
        ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        if ratio > 1.5:
            direction = SignalDirection.BULLISH.value
            strength = min((ratio - 1.0) / 2.0, 1.0)
        elif ratio < 0.5:
            direction = SignalDirection.BEARISH.value
            strength = min((1.0 - ratio) / 2.0, 1.0)
        else:
            direction = SignalDirection.NEUTRAL.value
            strength = 0.0
        
        return Evidence(
            source="Volume",
            factor_name="Volume_Ratio",
            category=EvidenceCategory.LIQUIDITY.value,
            signal_direction=direction,
            strength=strength,
            confidence=confidence,
            timestamp=datetime.now(),
            metadata={"current_volume": current_volume, "avg_volume": avg_volume, "ratio": ratio},
        )
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Evidence:
        """
        Create evidence from dictionary.
        
        Args:
            data: Dictionary with evidence data
            
        Returns:
            Evidence object
        """
        return Evidence.deserialize(data)


def build_evidence(
    source: str,
    factor_name: str,
    category: str,
    signal_direction: str,
    strength: float,
    confidence: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> Evidence:
    """
    Convenience function to build evidence.
    
    Args:
        source: Evidence source
        factor_name: Factor name
        category: Evidence category
        signal_direction: Signal direction
        strength: Signal strength
        confidence: Confidence
        metadata: Optional metadata
        
    Returns:
        Evidence object
    """
    return Evidence(
        source=source,
        factor_name=factor_name,
        category=category,
        signal_direction=signal_direction,
        strength=strength,
        confidence=confidence,
        timestamp=datetime.now(),
        metadata=metadata,
    )
