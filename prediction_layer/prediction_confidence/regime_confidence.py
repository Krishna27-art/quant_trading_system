"""
Regime Confidence Calculator

Calculates confidence based on market regime match.
Historical accuracy in similar regimes increases confidence.
"""

from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum

from utils.logger import get_logger

logger = get_logger("prediction_layer.prediction_confidence.regime_confidence")


class MarketRegime(Enum):
    """Market regime enumeration."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    STRONG_BULL = "strong_bull"
    STRONG_BEAR = "strong_bear"
    UNKNOWN = "unknown"


@dataclass
class RegimePerformance:
    """Historical performance data for a regime."""
    regime: MarketRegime
    accuracy: float
    total_predictions: int
    successful_predictions: int
    average_return: float
    sharpe_ratio: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "regime": self.regime.value,
            "accuracy": round(self.accuracy, 4),
            "total_predictions": self.total_predictions,
            "successful_predictions": self.successful_predictions,
            "average_return": round(self.average_return, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4) if self.sharpe_ratio else None,
        }


@dataclass
class RegimeConfidenceResult:
    """Result of regime confidence calculation."""
    confidence_score: float
    confidence_level: str
    current_regime: MarketRegime
    regime_match: bool
    historical_accuracy: float
    sample_size: int
    regime_similarity: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "confidence_score": round(self.confidence_score, 4),
            "confidence_level": self.confidence_level,
            "current_regime": self.current_regime.value,
            "regime_match": self.regime_match,
            "historical_accuracy": round(self.historical_accuracy, 4),
            "sample_size": self.sample_size,
            "regime_similarity": round(self.regime_similarity, 4),
        }


class RegimeConfidenceCalculator:
    """
    Calculates confidence based on market regime.
    
    Confidence is based on:
    - Historical accuracy in the current regime
    - Sample size (more data = higher confidence)
    - Regime similarity to best-performing regimes
    """
    
    def __init__(
        self,
        high_confidence_threshold: float = 0.75,
        low_confidence_threshold: float = 0.5,
        min_sample_size: int = 30,
    ):
        """
        Initialize regime confidence calculator.
        
        Args:
            high_confidence_threshold: Threshold for HIGH confidence
            low_confidence_threshold: Threshold for LOW confidence
            min_sample_size: Minimum sample size for reliable statistics
        """
        self.high_confidence_threshold = high_confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.min_sample_size = min_sample_size
        self._logger = get_logger("prediction_layer.prediction_confidence.regime_confidence")
    
    def calculate_confidence(
        self,
        current_regime: MarketRegime,
        regime_performance: Dict[MarketRegime, RegimePerformance],
        best_regime: Optional[MarketRegime] = None,
    ) -> RegimeConfidenceResult:
        """
        Calculate confidence based on regime match and historical performance.
        
        Args:
            current_regime: Current market regime
            regime_performance: Dictionary mapping regimes to performance data
            best_regime: Optional best regime for this prediction type
            
        Returns:
            RegimeConfidenceResult
        """
        # Get performance for current regime
        current_performance = regime_performance.get(current_regime)
        
        if current_performance is None:
            self._logger.warning(
                f"No performance data for regime: {current_regime.value}"
            )
            return RegimeConfidenceResult(
                confidence_score=0.3,
                confidence_level="LOW",
                current_regime=current_regime,
                regime_match=False,
                historical_accuracy=0.0,
                sample_size=0,
                regime_similarity=0.0,
            )
        
        # Calculate regime similarity to best regime
        regime_similarity = self._calculate_regime_similarity(
            current_regime,
            best_regime,
        )
        
        # Calculate sample size score
        sample_size_score = self._calculate_sample_size_score(
            current_performance.total_predictions,
        )
        
        # Calculate accuracy score
        accuracy_score = current_performance.accuracy
        
        # Combine into overall confidence
        confidence_score = self._calculate_overall_confidence(
            accuracy_score,
            sample_size_score,
            regime_similarity,
        )
        
        # Determine confidence level
        confidence_level = self._get_confidence_level(confidence_score)
        
        # Check if current regime matches best regime
        regime_match = (best_regime is not None and current_regime == best_regime)
        
        self._logger.info(
            f"Regime confidence calculated: {confidence_level} "
            f"(score={confidence_score:.4f}, regime={current_regime.value})"
        )
        
        return RegimeConfidenceResult(
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            current_regime=current_regime,
            regime_match=regime_match,
            historical_accuracy=current_performance.accuracy,
            sample_size=current_performance.total_predictions,
            regime_similarity=regime_similarity,
        )
    
    def _calculate_regime_similarity(
        self,
        current_regime: MarketRegime,
        best_regime: Optional[MarketRegime],
    ) -> float:
        """
        Calculate similarity between current and best regime.
        
        Args:
            current_regime: Current market regime
            best_regime: Best regime for this prediction type
            
        Returns:
            Similarity score (0-1)
        """
        if best_regime is None:
            return 0.5  # Neutral if no best regime specified
        
        if current_regime == best_regime:
            return 1.0
        
        # Define regime groups
        regime_groups = {
            "bull": [MarketRegime.BULL, MarketRegime.STRONG_BULL],
            "bear": [MarketRegime.BEAR, MarketRegime.STRONG_BEAR],
            "sideways": [MarketRegime.SIDEWAYS],
            "high_volatility": [MarketRegime.HIGH_VOLATILITY],
            "low_volatility": [MarketRegime.LOW_VOLATILITY],
        }
        
        # Check if regimes are in the same group
        for group, regimes in regime_groups.items():
            if current_regime in regimes and best_regime in regimes:
                return 0.7  # Partial similarity
        
        return 0.3  # Low similarity
    
    def _calculate_sample_size_score(
        self,
        sample_size: int,
    ) -> float:
        """
        Calculate sample size score.
        
        Args:
            sample_size: Number of historical predictions in this regime
            
        Returns:
            Sample size score (0-1)
        """
        if sample_size < self.min_sample_size:
            # Linear scaling for small samples
            return sample_size / self.min_sample_size
        
        # Logarithmic scaling for larger samples
        # 100 samples = 0.9, 1000 samples = 1.0
        import math
        score = 0.9 + 0.1 * math.log10(sample_size / 100.0)
        return min(1.0, max(0.0, score))
    
    def _calculate_overall_confidence(
        self,
        accuracy_score: float,
        sample_size_score: float,
        regime_similarity: float,
    ) -> float:
        """
        Calculate overall regime confidence score.
        
        Args:
            accuracy_score: Historical accuracy score
            sample_size_score: Sample size score
            regime_similarity: Regime similarity score
            
        Returns:
            Overall confidence score (0-1)
        """
        # Weighted combination
        confidence = (
            accuracy_score * 0.5 +
            sample_size_score * 0.3 +
            regime_similarity * 0.2
        )
        
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
    
    def get_regime_performance_summary(
        self,
        regime_performance: Dict[MarketRegime, RegimePerformance],
    ) -> Dict:
        """
        Get summary of regime performance.
        
        Args:
            regime_performance: Dictionary mapping regimes to performance data
            
        Returns:
            Dictionary with performance summary
        """
        summary = {
            "total_regimes": len(regime_performance),
            "best_regime": None,
            "worst_regime": None,
            "average_accuracy": 0.0,
            "total_predictions": 0,
            "regimes": {},
        }
        
        if not regime_performance:
            return summary
        
        accuracies = []
        total_preds = 0
        
        for regime, perf in regime_performance.items():
            accuracies.append(perf.accuracy)
            total_preds += perf.total_predictions
            
            summary["regimes"][regime.value] = perf.to_dict()
        
        summary["average_accuracy"] = sum(accuracies) / len(accuracies)
        summary["total_predictions"] = total_preds
        
        # Find best and worst regimes
        if regime_performance:
            best_regime = max(
                regime_performance.items(),
                key=lambda x: x[1].accuracy,
            )
            worst_regime = min(
                regime_performance.items(),
                key=lambda x: x[1].accuracy,
            )
            
            summary["best_regime"] = best_regime[0].value
            summary["worst_regime"] = worst_regime[0].value
        
        return summary


def calculate_regime_confidence(
    current_regime: MarketRegime,
    regime_performance: Dict[MarketRegime, RegimePerformance],
    best_regime: Optional[MarketRegime] = None,
) -> RegimeConfidenceResult:
    """
    Convenience function to calculate regime confidence.
    
    Args:
        current_regime: Current market regime
        regime_performance: Dictionary mapping regimes to performance data
        best_regime: Optional best regime for this prediction type
        
    Returns:
        RegimeConfidenceResult
    """
    calculator = RegimeConfidenceCalculator()
    return calculator.calculate_confidence(current_regime, regime_performance, best_regime)
