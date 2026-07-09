"""
Signal Quality Engine

Calculates quality scores for trading signals.
Evaluates signal strength across multiple dimensions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger("signal_engine.signal_quality")


@dataclass
class QualityScore:
    """Quality score for a signal."""
    trend_quality: float
    momentum_quality: float
    volume_quality: float
    liquidity_quality: float
    macro_quality: float
    sentiment_quality: float
    options_quality: float
    consistency_quality: float
    overall_quality: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "trend_quality": round(self.trend_quality, 2),
            "momentum_quality": round(self.momentum_quality, 2),
            "volume_quality": round(self.volume_quality, 2),
            "liquidity_quality": round(self.liquidity_quality, 2),
            "macro_quality": round(self.macro_quality, 2),
            "sentiment_quality": round(self.sentiment_quality, 2),
            "options_quality": round(self.options_quality, 2),
            "consistency_quality": round(self.consistency_quality, 2),
            "overall_quality": round(self.overall_quality, 2),
        }


class SignalQualityEngine:
    """
    Calculates quality scores for trading signals.
    
    Evaluates signal strength across multiple dimensions:
    - Trend quality
    - Momentum quality
    - Volume quality
    - Liquidity quality
    - Macro quality
    - Sentiment quality
    - Options quality
    - Consistency quality
    """
    
    def __init__(
        self,
        quality_threshold: float = 70.0,
        category_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize signal quality engine.
        
        Args:
            quality_threshold: Minimum quality threshold for valid signals
            category_weights: Optional weights for each category
        """
        self.quality_threshold = quality_threshold
        self.category_weights = category_weights or {
            "trend": 1.0,
            "momentum": 1.0,
            "volume": 0.9,
            "liquidity": 0.8,
            "macro": 0.7,
            "sentiment": 0.6,
            "options": 0.9,
            "consistency": 1.0,
        }
        self._logger = get_logger("signal_engine.signal_quality")
    
    def calculate_quality(
        self,
        factor_scores: Dict[str, float],
        factor_categories: Dict[str, str],
    ) -> QualityScore:
        """
        Calculate overall signal quality.
        
        Args:
            factor_scores: Dictionary mapping factor names to scores (-1 to 1)
            factor_categories: Dictionary mapping factor names to categories
            
        Returns:
            QualityScore
        """
        # Group scores by category
        category_scores = self._group_by_category(factor_scores, factor_categories)
        
        # Calculate quality for each category
        trend_quality = self._calculate_category_quality(category_scores.get("trend", []))
        momentum_quality = self._calculate_category_quality(category_scores.get("momentum", []))
        volume_quality = self._calculate_category_quality(category_scores.get("volume", []))
        liquidity_quality = self._calculate_category_quality(category_scores.get("liquidity", []))
        macro_quality = self._calculate_category_quality(category_scores.get("macro", []))
        sentiment_quality = self._calculate_category_quality(category_scores.get("sentiment", []))
        options_quality = self._calculate_category_quality(category_scores.get("options", []))
        
        # Calculate consistency (agreement across categories)
        consistency_quality = self._calculate_consistency(category_scores)
        
        # Calculate overall quality
        overall_quality = self._calculate_overall_quality(
            trend_quality,
            momentum_quality,
            volume_quality,
            liquidity_quality,
            macro_quality,
            sentiment_quality,
            options_quality,
            consistency_quality,
        )
        
        return QualityScore(
            trend_quality=trend_quality,
            momentum_quality=momentum_quality,
            volume_quality=volume_quality,
            liquidity_quality=liquidity_quality,
            macro_quality=macro_quality,
            sentiment_quality=sentiment_quality,
            options_quality=options_quality,
            consistency_quality=consistency_quality,
            overall_quality=overall_quality,
        )
    
    def _group_by_category(
        self,
        factor_scores: Dict[str, float],
        factor_categories: Dict[str, str],
    ) -> Dict[str, List[float]]:
        """Group factor scores by category."""
        category_scores = {}
        
        for factor_name, score in factor_scores.items():
            category = factor_categories.get(factor_name, "unknown")
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(abs(score))
        
        return category_scores
    
    def _calculate_category_quality(self, scores: List[float]) -> float:
        """
        Calculate quality score for a category.
        
        Args:
            scores: List of absolute scores
            
        Returns:
            Quality score (0 to 100)
        """
        if not scores:
            return 0.0
        
        # Average score converted to 0-100 scale
        avg_score = np.mean(scores)
        quality = avg_score * 100
        
        return min(100.0, max(0.0, quality))
    
    def _calculate_consistency(self, category_scores: Dict[str, List[float]]) -> float:
        """
        Calculate consistency across categories.
        
        Args:
            category_scores: Dictionary mapping categories to scores
            
        Returns:
            Consistency score (0 to 100)
        """
        if not category_scores:
            return 0.0
        
        # Calculate average score for each category
        category_averages = []
        for scores in category_scores.values():
            if scores:
                category_averages.append(np.mean(scores))
        
        if not category_averages:
            return 0.0
        
        # Consistency is based on standard deviation
        std_dev = np.std(category_averages)
        consistency = max(0.0, 1.0 - std_dev) * 100
        
        return consistency
    
    def _calculate_overall_quality(
        self,
        trend_quality: float,
        momentum_quality: float,
        volume_quality: float,
        liquidity_quality: float,
        macro_quality: float,
        sentiment_quality: float,
        options_quality: float,
        consistency_quality: float,
    ) -> float:
        """
        Calculate overall quality score.
        
        Args:
            Individual category quality scores
            
        Returns:
            Overall quality score (0 to 100)
        """
        qualities = {
            "trend": trend_quality,
            "momentum": momentum_quality,
            "volume": volume_quality,
            "liquidity": liquidity_quality,
            "macro": macro_quality,
            "sentiment": sentiment_quality,
            "options": options_quality,
            "consistency": consistency_quality,
        }
        
        # Weighted average
        weighted_sum = 0.0
        total_weight = 0.0
        
        for category, quality in qualities.items():
            weight = self.category_weights.get(category, 1.0)
            weighted_sum += quality * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        overall = weighted_sum / total_weight
        
        return min(100.0, max(0.0, overall))
    
    def is_signal_valid(self, quality_score: QualityScore) -> bool:
        """
        Check if signal meets quality threshold.
        
        Args:
            quality_score: QualityScore to validate
            
        Returns:
            True if signal is valid
        """
        return quality_score.overall_quality >= self.quality_threshold
    
    def get_quality_rating(self, quality: float) -> str:
        """
        Get quality rating.
        
        Args:
            quality: Quality score
            
        Returns:
            Rating: "EXCELLENT", "GOOD", "FAIR", "POOR"
        """
        if quality >= 90:
            return "EXCELLENT"
        elif quality >= 75:
            return "GOOD"
        elif quality >= 60:
            return "FAIR"
        else:
            return "POOR"


def calculate_signal_quality(
    factor_scores: Dict[str, float],
    factor_categories: Dict[str, str],
    quality_threshold: float = 70.0,
) -> QualityScore:
    """
    Convenience function to calculate signal quality.
    
    Args:
        factor_scores: Dictionary mapping factor names to scores
        factor_categories: Dictionary mapping factor names to categories
        quality_threshold: Minimum quality threshold
        
    Returns:
        QualityScore
    """
    engine = SignalQualityEngine(quality_threshold=quality_threshold)
    return engine.calculate_quality(factor_scores, factor_categories)
