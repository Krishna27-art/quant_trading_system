"""
Historical Similarity Calculator

Finds similar historical market conditions to assess prediction confidence.
If similar conditions led to successful predictions, confidence increases.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("prediction_layer.prediction_confidence.historical_similarity")


@dataclass
class MarketCondition:
    """Market condition at a specific point in time."""
    date: datetime
    regime: str
    volatility: float
    trend: str
    volume_ratio: float
    sector_performance: Dict[str, float]
    vix: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "date": self.date.isoformat(),
            "regime": self.regime,
            "volatility": round(self.volatility, 4),
            "trend": self.trend,
            "volume_ratio": round(self.volume_ratio, 4),
            "sector_performance": {k: round(v, 4) for k, v in self.sector_performance.items()},
            "vix": round(self.vix, 4) if self.vix else None,
        }


@dataclass
class SimilarDay:
    """Historical day similar to current conditions."""
    date: datetime
    similarity_score: float
    prediction_successful: bool
    return_percentage: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "date": self.date.isoformat(),
            "similarity_score": round(self.similarity_score, 4),
            "prediction_successful": self.prediction_successful,
            "return_percentage": round(self.return_percentage, 4),
        }


@dataclass
class HistoricalSimilarityResult:
    """Result of historical similarity calculation."""
    confidence_score: float
    confidence_level: str
    similar_days_found: int
    successful_predictions: int
    success_rate: float
    average_return: float
    top_similar_days: List[SimilarDay]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "confidence_score": round(self.confidence_score, 4),
            "confidence_level": self.confidence_level,
            "similar_days_found": self.similar_days_found,
            "successful_predictions": self.successful_predictions,
            "success_rate": round(self.success_rate, 4),
            "average_return": round(self.average_return, 4),
            "top_similar_days": [day.to_dict() for day in self.top_similar_days],
        }


class HistoricalSimilarityCalculator:
    """
    Calculates confidence based on historical similarity.
    
    Confidence is based on:
    - Number of similar historical days found
    - Success rate of predictions on similar days
    - Average returns on similar days
    """
    
    def __init__(
        self,
        high_confidence_threshold: float = 0.7,
        low_confidence_threshold: float = 0.4,
        min_similar_days: int = 10,
        max_similar_days: int = 100,
    ):
        """
        Initialize historical similarity calculator.
        
        Args:
            high_confidence_threshold: Threshold for HIGH confidence
            low_confidence_threshold: Threshold for LOW confidence
            min_similar_days: Minimum similar days required
            max_similar_days: Maximum similar days to consider
        """
        self.high_confidence_threshold = high_confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.min_similar_days = min_similar_days
        self.max_similar_days = max_similar_days
        self._logger = get_logger("prediction_layer.prediction_confidence.historical_similarity")
    
    def calculate_similarity(
        self,
        current_condition: MarketCondition,
        historical_conditions: List[MarketCondition],
        historical_outcomes: Dict[datetime, bool],
        historical_returns: Dict[datetime, float],
    ) -> HistoricalSimilarityResult:
        """
        Calculate confidence based on historical similarity.
        
        Args:
            current_condition: Current market condition
            historical_conditions: List of historical market conditions
            historical_outcomes: Dictionary mapping date to prediction success
            historical_returns: Dictionary mapping date to return percentage
            
        Returns:
            HistoricalSimilarityResult
        """
        if not historical_conditions:
            self._logger.warning("No historical conditions provided")
            return HistoricalSimilarityResult(
                confidence_score=0.0,
                confidence_level="NONE",
                similar_days_found=0,
                successful_predictions=0,
                success_rate=0.0,
                average_return=0.0,
                top_similar_days=[],
            )
        
        # Calculate similarity scores for all historical days
        similar_days = []
        
        for hist_condition in historical_conditions:
            similarity_score = self._calculate_condition_similarity(
                current_condition,
                hist_condition,
            )
            
            # Get outcome and return if available
            prediction_successful = historical_outcomes.get(hist_condition.date, None)
            return_percentage = historical_returns.get(hist_condition.date, 0.0)
            
            similar_day = SimilarDay(
                date=hist_condition.date,
                similarity_score=similarity_score,
                prediction_successful=prediction_successful,
                return_percentage=return_percentage,
            )
            
            similar_days.append(similar_day)
        
        # Sort by similarity score (descending)
        similar_days.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Take top N similar days
        top_similar_days = similar_days[:self.max_similar_days]
        
        # Filter days with known outcomes
        days_with_outcomes = [
            day for day in top_similar_days
            if day.prediction_successful is not None
        ]
        
        similar_days_found = len(days_with_outcomes)
        
        if similar_days_found < self.min_similar_days:
            self._logger.warning(
                f"Insufficient similar days: {similar_days_found} < {self.min_similar_days}"
            )
            return HistoricalSimilarityResult(
                confidence_score=0.3,
                confidence_level="LOW",
                similar_days_found=similar_days_found,
                successful_predictions=0,
                success_rate=0.0,
                average_return=0.0,
                top_similar_days=top_similar_days[:5],
            )
        
        # Calculate success rate
        successful_predictions = sum(
            1 for day in days_with_outcomes if day.prediction_successful
        )
        success_rate = successful_predictions / similar_days_found
        
        # Calculate average return
        returns = [day.return_percentage for day in days_with_outcomes]
        average_return = np.mean(returns)
        
        # Calculate overall confidence score
        confidence_score = self._calculate_overall_confidence(
            success_rate,
            similar_days_found,
            average_return,
        )
        
        # Determine confidence level
        confidence_level = self._get_confidence_level(confidence_score)
        
        self._logger.info(
            f"Historical similarity calculated: {confidence_level} "
            f"(score={confidence_score:.4f}, success_rate={success_rate:.4f})"
        )
        
        return HistoricalSimilarityResult(
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            similar_days_found=similar_days_found,
            successful_predictions=successful_predictions,
            success_rate=success_rate,
            average_return=average_return,
            top_similar_days=top_similar_days[:10],
        )
    
    def _calculate_condition_similarity(
        self,
        current: MarketCondition,
        historical: MarketCondition,
    ) -> float:
        """
        Calculate similarity between two market conditions.
        
        Args:
            current: Current market condition
            historical: Historical market condition
            
        Returns:
            Similarity score (0-1)
        """
        similarity_components = []
        
        # Regime similarity
        regime_similarity = 1.0 if current.regime == historical.regime else 0.0
        similarity_components.append(regime_similarity * 0.3)
        
        # Volatility similarity (inverse of absolute difference)
        vol_diff = abs(current.volatility - historical.volatility)
        vol_similarity = max(0.0, 1.0 - vol_diff)
        similarity_components.append(vol_similarity * 0.2)
        
        # Trend similarity
        trend_similarity = 1.0 if current.trend == historical.trend else 0.0
        similarity_components.append(trend_similarity * 0.2)
        
        # Volume ratio similarity
        vol_ratio_diff = abs(current.volume_ratio - historical.volume_ratio)
        vol_ratio_similarity = max(0.0, 1.0 - vol_ratio_diff / 2.0)
        similarity_components.append(vol_ratio_similarity * 0.15)
        
        # Sector performance similarity
        sector_similarity = self._calculate_sector_similarity(
            current.sector_performance,
            historical.sector_performance,
        )
        similarity_components.append(sector_similarity * 0.15)
        
        return sum(similarity_components)
    
    def _calculate_sector_similarity(
        self,
        current_sectors: Dict[str, float],
        historical_sectors: Dict[str, float],
    ) -> float:
        """
        Calculate similarity between sector performances.
        
        Args:
            current_sectors: Current sector performance
            historical_sectors: Historical sector performance
            
        Returns:
            Similarity score (0-1)
        """
        if not current_sectors or not historical_sectors:
            return 0.5  # Neutral if no data
        
        # Get common sectors
        common_sectors = set(current_sectors.keys()) & set(historical_sectors.keys())
        
        if not common_sectors:
            return 0.0
        
        # Calculate correlation-like similarity
        similarities = []
        
        for sector in common_sectors:
            diff = abs(current_sectors[sector] - historical_sectors[sector])
            similarity = max(0.0, 1.0 - diff / 10.0)  # Normalize by 10% difference
            similarities.append(similarity)
        
        return np.mean(similarities)
    
    def _calculate_overall_confidence(
        self,
        success_rate: float,
        similar_days_found: int,
        average_return: float,
    ) -> float:
        """
        Calculate overall historical similarity confidence score.
        
        Args:
            success_rate: Success rate of predictions on similar days
            similar_days_found: Number of similar days found
            average_return: Average return on similar days
            
        Returns:
            Overall confidence score (0-1)
        """
        # Base score from success rate
        confidence = success_rate * 0.6
        
        # Add sample size score
        sample_size_score = min(1.0, similar_days_found / self.max_similar_days)
        confidence += sample_size_score * 0.2
        
        # Add return score (positive returns boost confidence)
        return_score = max(0.0, min(1.0, average_return / 5.0))  # Normalize by 5%
        confidence += return_score * 0.2
        
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


def calculate_historical_similarity(
    current_condition: MarketCondition,
    historical_conditions: List[MarketCondition],
    historical_outcomes: Dict[datetime, bool],
    historical_returns: Dict[datetime, float],
) -> HistoricalSimilarityResult:
    """
    Convenience function to calculate historical similarity.
    
    Args:
        current_condition: Current market condition
        historical_conditions: List of historical market conditions
        historical_outcomes: Dictionary mapping date to prediction success
        historical_returns: Dictionary mapping date to return percentage
        
    Returns:
        HistoricalSimilarityResult
    """
    calculator = HistoricalSimilarityCalculator()
    return calculator.calculate_similarity(
        current_condition,
        historical_conditions,
        historical_outcomes,
        historical_returns,
    )
