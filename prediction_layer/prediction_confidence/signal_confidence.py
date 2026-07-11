"""
Signal Confidence Calculator

Calculates confidence based on agreement between different signal types.
Signals include trend, volume, options, momentum, macro, etc.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

import numpy as np

from utils.logger import get_logger

logger = get_logger("prediction_layer.prediction_confidence.signal_confidence")


class SignalDirection(Enum):
    """Signal direction enumeration."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class SignalCategory(Enum):
    """Signal category enumeration."""
    TREND = "trend"
    VOLUME = "volume"
    OPTIONS = "options"
    MOMENTUM = "momentum"
    MACRO = "macro"
    SECTOR = "sector"
    NEWS = "news"
    TECHNICAL = "technical"


@dataclass
class Signal:
    """Trading signal from a specific category."""
    category: SignalCategory
    direction: SignalDirection
    strength: float
    confidence: Optional[float] = None
    source: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "category": self.category.value,
            "direction": self.direction.value,
            "strength": round(self.strength, 4),
            "confidence": round(self.confidence, 4) if self.confidence else None,
            "source": self.source,
        }


@dataclass
class SignalAgreementResult:
    """Result of signal agreement calculation."""
    agreement_score: float
    agreement_level: str
    dominant_direction: SignalDirection
    category_counts: Dict[str, int]
    category_agreement: Dict[str, float]
    average_strength: float
    participating_categories: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "agreement_score": round(self.agreement_score, 4),
            "agreement_level": self.agreement_level,
            "dominant_direction": self.dominant_direction.value,
            "category_counts": self.category_counts,
            "category_agreement": {k: round(v, 4) for k, v in self.category_agreement.items()},
            "average_strength": round(self.average_strength, 4),
            "participating_categories": self.participating_categories,
        }


class SignalConfidenceCalculator:
    """
    Calculates confidence based on signal agreement.
    
    Agreement is based on:
    - Direction consensus across signal categories
    - Signal strength
    - Individual signal confidence if available
    """
    
    def __init__(
        self,
        high_agreement_threshold: float = 0.75,
        low_agreement_threshold: float = 0.4,
    ):
        """
        Initialize signal confidence calculator.
        
        Args:
            high_agreement_threshold: Threshold for HIGH agreement
            low_agreement_threshold: Threshold for LOW agreement
        """
        self.high_agreement_threshold = high_agreement_threshold
        self.low_agreement_threshold = low_agreement_threshold
        self._logger = get_logger("prediction_layer.prediction_confidence.signal_confidence")
    
    def calculate_agreement(
        self,
        signals: List[Signal],
    ) -> SignalAgreementResult:
        """
        Calculate agreement between signals.
        
        Args:
            signals: List of Signal objects
            
        Returns:
            SignalAgreementResult
        """
        if not signals:
            self._logger.warning("No signals provided for agreement calculation")
            return SignalAgreementResult(
                agreement_score=0.0,
                agreement_level="NONE",
                dominant_direction=SignalDirection.NEUTRAL,
                category_counts={},
                category_agreement={},
                average_strength=0.0,
                participating_categories=[],
            )
        
        # Count directions by category
        category_directions = self._group_signals_by_category(signals)
        
        # Calculate dominant direction
        dominant_direction = self._get_dominant_direction(signals)
        
        # Calculate overall direction agreement
        direction_agreement = self._calculate_direction_agreement(signals)
        
        # Calculate category agreement
        category_agreement = self._calculate_category_agreement(category_directions)
        
        # Calculate average strength
        strengths = [s.strength for s in signals]
        average_strength = np.mean(strengths)
        
        # Combine direction agreement and category agreement
        agreement_score = (
            direction_agreement * 0.6 +
            category_agreement * 0.4
        )
        
        # Boost by average strength
        agreement_score = min(1.0, agreement_score * (0.5 + average_strength * 0.5))
        
        # Determine agreement level
        agreement_level = self._get_agreement_level(agreement_score)
        
        participating_categories = list(set(s.category.value for s in signals))
        category_counts = self._count_signals_by_category(signals)
        
        self._logger.info(
            f"Signal agreement calculated: {agreement_level} "
            f"(score={agreement_score:.4f}, dominant={dominant_direction.value})"
        )
        
        return SignalAgreementResult(
            agreement_score=agreement_score,
            agreement_level=agreement_level,
            dominant_direction=dominant_direction,
            category_counts=category_counts,
            category_agreement=self._get_category_agreement_dict(signals),
            average_strength=average_strength,
            participating_categories=participating_categories,
        )
    
    def _group_signals_by_category(
        self,
        signals: List[Signal],
    ) -> Dict[SignalCategory, List[Signal]]:
        """
        Group signals by category.
        
        Args:
            signals: List of Signal objects
            
        Returns:
            Dictionary mapping category to list of signals
        """
        grouped = {}
        
        for signal in signals:
            if signal.category not in grouped:
                grouped[signal.category] = []
            grouped[signal.category].append(signal)
        
        return grouped
    
    def _get_dominant_direction(
        self,
        signals: List[Signal],
    ) -> SignalDirection:
        """
        Get the dominant signal direction.
        
        Args:
            signals: List of Signal objects
            
        Returns:
            SignalDirection
        """
        if not signals:
            return SignalDirection.NEUTRAL
        
        direction_counts = {
            SignalDirection.BULLISH: 0,
            SignalDirection.BEARISH: 0,
            SignalDirection.NEUTRAL: 0,
        }
        
        for signal in signals:
            direction_counts[signal.direction] += 1
        
        max_count = max(direction_counts.values())
        dominant_direction = max(
            direction_counts,
            key=direction_counts.get,
        )
        
        return dominant_direction
    
    def _calculate_direction_agreement(
        self,
        signals: List[Signal],
    ) -> float:
        """
        Calculate direction agreement score.
        
        Args:
            signals: List of Signal objects
            
        Returns:
            Agreement score (0-1)
        """
        if not signals:
            return 0.0
        
        direction_counts = {
            SignalDirection.BULLISH: 0,
            SignalDirection.BEARISH: 0,
            SignalDirection.NEUTRAL: 0,
        }
        
        for signal in signals:
            direction_counts[signal.direction] += 1
        
        max_count = max(direction_counts.values())
        agreement = max_count / len(signals)
        
        return agreement
    
    def _calculate_category_agreement(
        self,
        category_directions: Dict[SignalCategory, List[Signal]],
    ) -> float:
        """
        Calculate agreement across categories.
        
        Args:
            category_directions: Dictionary mapping category to list of signals
            
        Returns:
            Agreement score (0-1)
        """
        if not category_directions:
            return 0.0
        
        category_agreements = []
        
        for category, signals in category_directions.items():
            if not signals:
                continue
            
            # Calculate agreement within this category
            direction_counts = {
                SignalDirection.BULLISH: 0,
                SignalDirection.BEARISH: 0,
                SignalDirection.NEUTRAL: 0,
            }
            
            for signal in signals:
                direction_counts[signal.direction] += 1
            
            max_count = max(direction_counts.values())
            agreement = max_count / len(signals)
            category_agreements.append(agreement)
        
        if not category_agreements:
            return 0.0
        
        return np.mean(category_agreements)
    
    def _count_signals_by_category(
        self,
        signals: List[Signal],
    ) -> Dict[str, int]:
        """
        Count signals by category.
        
        Args:
            signals: List of Signal objects
            
        Returns:
            Dictionary mapping category name to count
        """
        counts = {}
        
        for signal in signals:
            category_name = signal.category.value
            counts[category_name] = counts.get(category_name, 0) + 1
        
        return counts
    
    def _get_category_agreement_dict(
        self,
        signals: List[Signal],
    ) -> Dict[str, float]:
        """
        Get agreement score for each category.
        
        Args:
            signals: List of Signal objects
            
        Returns:
            Dictionary mapping category name to agreement score
        """
        category_directions = self._group_signals_by_category(signals)
        agreement_dict = {}
        
        for category, category_signals in category_directions.items():
            if not category_signals:
                continue
            
            direction_counts = {
                SignalDirection.BULLISH: 0,
                SignalDirection.BEARISH: 0,
                SignalDirection.NEUTRAL: 0,
            }
            
            for signal in category_signals:
                direction_counts[signal.direction] += 1
            
            max_count = max(direction_counts.values())
            agreement = max_count / len(category_signals)
            agreement_dict[category.value] = agreement
        
        return agreement_dict
    
    def _get_agreement_level(self, agreement_score: float) -> str:
        """
        Get agreement level from score.
        
        Args:
            agreement_score: Agreement score
            
        Returns:
            Agreement level: "HIGH", "MEDIUM", "LOW", "NONE"
        """
        if agreement_score >= self.high_agreement_threshold:
            return "HIGH"
        elif agreement_score >= self.low_agreement_threshold:
            return "MEDIUM"
        elif agreement_score > 0.0:
            return "LOW"
        else:
            return "NONE"
    
    def calculate_weighted_agreement(
        self,
        signals: List[Signal],
        category_weights: Optional[Dict[str, float]] = None,
    ) -> SignalAgreementResult:
        """
        Calculate weighted agreement between signals.
        
        Args:
            signals: List of Signal objects
            category_weights: Optional weights for each signal category
            
        Returns:
            SignalAgreementResult
        """
        if not category_weights:
            return self.calculate_agreement(signals)
        
        # Apply weights to direction counts
        weighted_direction_counts = {
            SignalDirection.BULLISH: 0.0,
            SignalDirection.BEARISH: 0.0,
            SignalDirection.NEUTRAL: 0.0,
        }
        
        total_weight = 0.0
        
        for signal in signals:
            weight = category_weights.get(signal.category.value, 1.0)
            weighted_direction_counts[signal.direction] += weight
            total_weight += weight
        
        # Normalize weighted counts
        if total_weight > 0:
            for direction in weighted_direction_counts:
                weighted_direction_counts[direction] /= total_weight
        
        # Calculate dominant direction from weighted counts
        max_weight = max(weighted_direction_counts.values())
        dominant_direction = max(
            weighted_direction_counts,
            key=weighted_direction_counts.get,
        )
        
        # Calculate weighted direction agreement
        weighted_agreement = max_weight
        
        # Calculate category agreement
        category_directions = self._group_signals_by_category(signals)
        category_agreement = self._calculate_category_agreement(category_directions)
        
        # Calculate average strength
        strengths = [s.strength for s in signals]
        average_strength = np.mean(strengths)
        
        # Combine weighted direction agreement and category agreement
        agreement_score = (
            weighted_agreement * 0.6 +
            category_agreement * 0.4
        )
        
        # Boost by average strength
        agreement_score = min(1.0, agreement_score * (0.5 + average_strength * 0.5))
        
        # Determine agreement level
        agreement_level = self._get_agreement_level(agreement_score)
        
        participating_categories = list(set(s.category.value for s in signals))
        category_counts = self._count_signals_by_category(signals)
        
        self._logger.info(
            f"Weighted signal agreement calculated: {agreement_level} "
            f"(score={agreement_score:.4f}, dominant={dominant_direction.value})"
        )
        
        return SignalAgreementResult(
            agreement_score=agreement_score,
            agreement_level=agreement_level,
            dominant_direction=dominant_direction,
            category_counts=category_counts,
            category_agreement=self._get_category_agreement_dict(signals),
            average_strength=average_strength,
            participating_categories=participating_categories,
        )


def calculate_signal_agreement(
    signals: List[Signal],
) -> SignalAgreementResult:
    """
    Convenience function to calculate signal agreement.
    
    Args:
        signals: List of Signal objects
        
    Returns:
        SignalAgreementResult
    """
    calculator = SignalConfidenceCalculator()
    return calculator.calculate_agreement(signals)
