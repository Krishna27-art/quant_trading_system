"""
Feature Confidence Calculator

Calculates confidence based on feature quality and ranking.
High-quality features from the feature ranking engine increase confidence.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger("prediction_layer.prediction_confidence.feature_confidence")


@dataclass
class FeatureInfo:
    """Information about a feature used in prediction."""
    feature_name: str
    importance_score: float
    ranking_position: Optional[int] = None
    quality_score: Optional[float] = None
    category: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "feature_name": self.feature_name,
            "importance_score": round(self.importance_score, 4),
            "ranking_position": self.ranking_position,
            "quality_score": round(self.quality_score, 4) if self.quality_score else None,
            "category": self.category,
        }


@dataclass
class FeatureConfidenceResult:
    """Result of feature confidence calculation."""
    confidence_score: float
    confidence_level: str
    average_importance: float
    top_features_used: int
    total_features_used: int
    average_quality: float
    feature_categories: Dict[str, int]
    weak_features: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "confidence_score": round(self.confidence_score, 4),
            "confidence_level": self.confidence_level,
            "average_importance": round(self.average_importance, 4),
            "top_features_used": self.top_features_used,
            "total_features_used": self.total_features_used,
            "average_quality": round(self.average_quality, 4) if self.average_quality else None,
            "feature_categories": self.feature_categories,
            "weak_features": self.weak_features,
        }


class FeatureConfidenceCalculator:
    """
    Calculates confidence based on feature quality.
    
    Confidence is based on:
    - Feature importance scores
    - Ranking positions (top features are better)
    - Quality scores if available
    - Feature diversity across categories
    """
    
    def __init__(
        self,
        high_confidence_threshold: float = 0.7,
        low_confidence_threshold: float = 0.4,
        top_feature_threshold: int = 10,
        weak_importance_threshold: float = 0.1,
    ):
        """
        Initialize feature confidence calculator.
        
        Args:
            high_confidence_threshold: Threshold for HIGH confidence
            low_confidence_threshold: Threshold for LOW confidence
            top_feature_threshold: Position threshold for "top" features
            weak_importance_threshold: Importance threshold for weak features
        """
        self.high_confidence_threshold = high_confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.top_feature_threshold = top_feature_threshold
        self.weak_importance_threshold = weak_importance_threshold
        self._logger = get_logger("prediction_layer.prediction_confidence.feature_confidence")
    
    def calculate_confidence(
        self,
        features: List[FeatureInfo],
        total_features_in_ranking: Optional[int] = None,
    ) -> FeatureConfidenceResult:
        """
        Calculate confidence based on feature quality.
        
        Args:
            features: List of FeatureInfo objects
            total_features_in_ranking: Total number of features in ranking
            
        Returns:
            FeatureConfidenceResult
        """
        if not features:
            self._logger.warning("No features provided for confidence calculation")
            return FeatureConfidenceResult(
                confidence_score=0.0,
                confidence_level="NONE",
                average_importance=0.0,
                top_features_used=0,
                total_features_used=0,
                average_quality=0.0,
                feature_categories={},
                weak_features=[],
            )
        
        # Calculate average importance
        importance_scores = [f.importance_score for f in features]
        average_importance = np.mean(importance_scores)
        
        # Count top features used
        top_features_used = sum(
            1 for f in features
            if f.ranking_position is not None and f.ranking_position <= self.top_feature_threshold
        )
        
        # Calculate feature diversity score
        feature_categories = self._count_feature_categories(features)
        diversity_score = self._calculate_diversity_score(feature_categories, len(features))
        
        # Calculate average quality if available
        quality_scores = [f.quality_score for f in features if f.quality_score is not None]
        average_quality = np.mean(quality_scores) if quality_scores else None
        
        # Identify weak features
        weak_features = [
            f.feature_name for f in features
            if f.importance_score < self.weak_importance_threshold
        ]
        
        # Calculate ranking score
        ranking_score = self._calculate_ranking_score(
            features,
            total_features_in_ranking,
        )
        
        # Calculate overall confidence
        confidence_score = self._calculate_overall_confidence(
            average_importance,
            ranking_score,
            diversity_score,
            average_quality,
            weak_features,
            len(features),
        )
        
        # Determine confidence level
        confidence_level = self._get_confidence_level(confidence_score)
        
        self._logger.info(
            f"Feature confidence calculated: {confidence_level} "
            f"(score={confidence_score:.4f}, avg_importance={average_importance:.4f})"
        )
        
        return FeatureConfidenceResult(
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            average_importance=average_importance,
            top_features_used=top_features_used,
            total_features_used=len(features),
            average_quality=average_quality or 0.0,
            feature_categories=feature_categories,
            weak_features=weak_features,
        )
    
    def _count_feature_categories(
        self,
        features: List[FeatureInfo],
    ) -> Dict[str, int]:
        """
        Count features by category.
        
        Args:
            features: List of FeatureInfo objects
            
        Returns:
            Dictionary mapping category to count
        """
        categories = {}
        
        for feature in features:
            if feature.category:
                categories[feature.category] = categories.get(feature.category, 0) + 1
        
        return categories
    
    def _calculate_diversity_score(
        self,
        feature_categories: Dict[str, int],
        total_features: int,
    ) -> float:
        """
        Calculate feature diversity score.
        
        Args:
            feature_categories: Dictionary mapping category to count
            total_features: Total number of features
            
        Returns:
            Diversity score (0-1)
        """
        if total_features == 0:
            return 0.0
        
        # More categories = higher diversity
        num_categories = len(feature_categories)
        
        # Normalize by ideal diversity (all features in different categories)
        ideal_diversity = min(num_categories, total_features)
        diversity_score = num_categories / max(ideal_diversity, 1)
        
        return min(1.0, diversity_score)
    
    def _calculate_ranking_score(
        self,
        features: List[FeatureInfo],
        total_features_in_ranking: Optional[int],
    ) -> float:
        """
        Calculate ranking score based on feature positions.
        
        Args:
            features: List of FeatureInfo objects
            total_features_in_ranking: Total number of features in ranking
            
        Returns:
            Ranking score (0-1)
        """
        if total_features_in_ranking is None:
            # If no total, use ranking positions relative to each other
            positions = [f.ranking_position for f in features if f.ranking_position is not None]
            if not positions:
                return 0.5  # Neutral if no ranking info
            
            # Inverse of average position (lower is better)
            avg_position = np.mean(positions)
            ranking_score = 1.0 / (1.0 + avg_position / 10.0)
            return ranking_score
        
        # Calculate score based on percentile of features used
        positions = [f.ranking_position for f in features if f.ranking_position is not None]
        if not positions:
            return 0.5
        
        avg_position = np.mean(positions)
        percentile = 1.0 - (avg_position / total_features_in_ranking)
        
        return max(0.0, min(1.0, percentile))
    
    def _calculate_overall_confidence(
        self,
        average_importance: float,
        ranking_score: float,
        diversity_score: float,
        average_quality: Optional[float],
        weak_features: List[str],
        total_features: int,
    ) -> float:
        """
        Calculate overall feature confidence score.
        
        Args:
            average_importance: Average feature importance
            ranking_score: Ranking score
            diversity_score: Diversity score
            average_quality: Average quality score
            weak_features: List of weak feature names
            total_features: Total number of features
            
        Returns:
            Overall confidence score (0-1)
        """
        # Base score from importance
        confidence = average_importance * 0.4
        
        # Add ranking score
        confidence += ranking_score * 0.25
        
        # Add diversity score
        confidence += diversity_score * 0.15
        
        # Add quality score if available
        if average_quality is not None:
            confidence += average_quality * 0.2
        
        # Penalty for weak features
        weak_ratio = len(weak_features) / max(total_features, 1)
        confidence *= (1.0 - weak_ratio * 0.5)
        
        return max(0.0, min(1.0, confidence))
    
    def _get_confidence_level(self, confidence_score: float) -> str:
        """
        Get confidence level from score.
        
        Args:
            confidence_score: Confidence score
            
        Returns:
            Confidence level: "HIGH", "MEDIUM", "LOW", "NONE"
        """
        if confidence_score >= self.high_confidence_threshold:
            return "HIGH"
        elif confidence_score >= self.low_confidence_threshold:
            return "MEDIUM"
        elif confidence_score > 0.0:
            return "LOW"
        else:
            return "NONE"


def calculate_feature_confidence(
    features: List[FeatureInfo],
    total_features_in_ranking: Optional[int] = None,
) -> FeatureConfidenceResult:
    """
    Convenience function to calculate feature confidence.
    
    Args:
        features: List of FeatureInfo objects
        total_features_in_ranking: Total number of features in ranking
        
    Returns:
        FeatureConfidenceResult
    """
    calculator = FeatureConfidenceCalculator()
    return calculator.calculate_confidence(features, total_features_in_ranking)
