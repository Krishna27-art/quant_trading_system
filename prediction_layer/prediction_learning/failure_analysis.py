"""
Failure Analysis

Analyzes why predictions failed and categorizes failure reasons.
This is critical for continuous improvement.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

from prediction_layer.prediction_learning.prediction_history import PredictionMetadata
from prediction_layer.prediction_learning.prediction_result import PredictionResult, PredictionQuality

from utils.logger import get_logger

logger = get_logger("prediction_layer.prediction_learning.failure_analysis")


class FailureReason(Enum):
    """Failure reason enumeration."""
    WRONG_TREND = "WRONG_TREND"
    NEWS_EVENT = "NEWS_EVENT"
    FII_SELLING = "FII_SELLING"
    LOW_VOLUME = "LOW_VOLUME"
    BAD_EARNINGS = "BAD_EARNINGS"
    FALSE_BREAKOUT = "FALSE_BREAKOUT"
    WRONG_REGIME = "WRONG_REGIME"
    MODEL_ERROR = "MODEL_ERROR"
    STOP_LOSS_TOO_TIGHT = "STOP_LOSS_TOO_TIGHT"
    TARGET_TOO_AGGRESSIVE = "TARGET_TOO_AGGRESSIVE"
    ENTRY_TOO_LATE = "ENTRY_TOO_LATE"
    MARKET_REVERSAL = "MARKET_REVERSAL"
    SECTOR_WEAKNESS = "SECTOR_WEAKNESS"
    UNKNOWN = "UNKNOWN"


@dataclass
class FailureAnalysis:
    """Analysis of a failed prediction."""
    prediction_id: str
    failure_reason: FailureReason
    confidence_in_reason: float
    contributing_factors: List[str]
    lessons_learned: str
    suggested_fixes: List[str]
    preventable: bool
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "prediction_id": self.prediction_id,
            "failure_reason": self.failure_reason.value,
            "confidence_in_reason": round(self.confidence_in_reason, 4),
            "contributing_factors": self.contributing_factors,
            "lessons_learned": self.lessons_learned,
            "suggested_fixes": self.suggested_fixes,
            "preventable": self.preventable,
        }


class FailureAnalyzer:
    """
    Analyzes failed predictions to identify root causes.
    
    Analysis considers:
    - Market conditions at failure time
    - Feature values at prediction time
    - Signal agreement at prediction time
    - Regime match at prediction time
    - Entry/exit quality
    """
    
    def __init__(self):
        """Initialize failure analyzer."""
        self._logger = get_logger("prediction_layer.prediction_learning.failure_analysis")
    
    def analyze_failure(
        self,
        prediction: PredictionMetadata,
        result: PredictionResult,
        market_conditions_at_failure: Optional[Dict] = None,
    ) -> FailureAnalysis:
        """
        Analyze a failed prediction.
        
        Args:
            prediction: Original prediction metadata
            result: Prediction result
            market_conditions_at_failure: Optional market conditions at failure
            
        Returns:
            FailureAnalysis
        """
        # Determine primary failure reason
        failure_reason = self._determine_failure_reason(
            prediction,
            result,
            market_conditions_at_failure,
        )
        
        # Calculate confidence in reason
        confidence = self._calculate_confidence_in_reason(
            prediction,
            result,
            failure_reason,
        )
        
        # Identify contributing factors
        contributing_factors = self._identify_contributing_factors(
            prediction,
            result,
            failure_reason,
        )
        
        # Generate lessons learned
        lessons_learned = self._generate_lessons_learned(
            failure_reason,
            contributing_factors,
        )
        
        # Suggest fixes
        suggested_fixes = self._suggest_fixes(
            failure_reason,
            contributing_factors,
        )
        
        # Determine if preventable
        preventable = self._is_preventable(failure_reason)
        
        self._logger.info(
            f"Analyzed failure for prediction {prediction.prediction_id}: "
            f"{failure_reason.value} (confidence={confidence:.2f})"
        )
        
        return FailureAnalysis(
            prediction_id=prediction.prediction_id,
            failure_reason=failure_reason,
            confidence_in_reason=confidence,
            contributing_factors=contributing_factors,
            lessons_learned=lessons_learned,
            suggested_fixes=suggested_fixes,
            preventable=preventable,
        )
    
    def _determine_failure_reason(
        self,
        prediction: PredictionMetadata,
        result: PredictionResult,
        market_conditions: Optional[Dict],
    ) -> FailureReason:
        """
        Determine primary failure reason.
        
        Args:
            prediction: Prediction metadata
            result: Prediction result
            market_conditions: Market conditions at failure
            
        Returns:
            FailureReason
        """
        # Check stop loss quality
        if result.stop_loss_quality_score < 0.4:
            return FailureReason.STOP_LOSS_TOO_TIGHT
        
        # Check target quality
        if result.target_quality_score < 0.4:
            return FailureReason.TARGET_TOO_AGGRESSIVE
        
        # Check entry quality
        if result.entry_quality_score < 0.4:
            return FailureReason.ENTRY_TOO_LATE
        
        # Check regime match
        if market_conditions:
            current_regime = market_conditions.get("regime")
            if current_regime != prediction.market_regime:
                return FailureReason.WRONG_REGIME
        
        # Check direction correctness
        if not result.direction_correct:
            # Could be wrong trend or market reversal
            if market_conditions and market_conditions.get("trend_reversal"):
                return FailureReason.MARKET_REVERSAL
            else:
                return FailureReason.WRONG_TREND
        
        # Check for news events
        if market_conditions and market_conditions.get("news_event"):
            return FailureReason.NEWS_EVENT
        
        # Check for low volume
        if market_conditions and market_conditions.get("low_volume"):
            return FailureReason.LOW_VOLUME
        
        # Check for sector weakness
        if market_conditions and market_conditions.get("sector_weakness"):
            return FailureReason.SECTOR_WEAKNESS
        
        # Default to model error
        return FailureReason.MODEL_ERROR
    
    def _calculate_confidence_in_reason(
        self,
        prediction: PredictionMetadata,
        result: PredictionResult,
        reason: FailureReason,
    ) -> float:
        """
        Calculate confidence in the identified failure reason.
        
        Args:
            prediction: Prediction metadata
            result: Prediction result
            reason: Identified failure reason
            
        Returns:
            Confidence score (0-1)
        """
        # Base confidence
        confidence = 0.5
        
        # Boost confidence based on quality scores
        if reason == FailureReason.STOP_LOSS_TOO_TIGHT:
            confidence = 1.0 - result.stop_loss_quality_score
        elif reason == FailureReason.TARGET_TOO_AGGRESSIVE:
            confidence = 1.0 - result.target_quality_score
        elif reason == FailureReason.ENTRY_TOO_LATE:
            confidence = 1.0 - result.entry_quality_score
        elif reason == FailureReason.WRONG_TREND:
            confidence = 0.9 if not result.direction_correct else 0.3
        
        return max(0.0, min(1.0, confidence))
    
    def _identify_contributing_factors(
        self,
        prediction: PredictionMetadata,
        result: PredictionResult,
        reason: FailureReason,
    ) -> List[str]:
        """
        Identify contributing factors to failure.
        
        Args:
            prediction: Prediction metadata
            result: Prediction result
            reason: Primary failure reason
            
        Returns:
            List of contributing factors
        """
        factors = []
        
        # Check confidence level
        if prediction.confidence == "LOW":
            factors.append("Low confidence prediction")
        
        # Check probability
        if prediction.probability < 0.6:
            factors.append("Low predicted probability")
        
        # Check feature quality
        if "volume" in prediction.features and prediction.features["volume"] < 0.5:
            factors.append("Low volume feature")
        
        # Check signal agreement
        if len(prediction.signals) < 3:
            factors.append("Insufficient signals")
        
        # Check regime
        if prediction.market_regime == "sideways":
            factors.append("Sideways market regime")
        
        # Add reason-specific factors
        if reason == FailureReason.STOP_LOSS_TOO_TIGHT:
            factors.append("Stop loss placed too close to entry")
        elif reason == FailureReason.TARGET_TOO_AGGRESSIVE:
            factors.append("Target set too high")
        elif reason == FailureReason.ENTRY_TOO_LATE:
            factors.append("Entry executed after optimal price")
        
        return factors
    
    def _generate_lessons_learned(
        self,
        reason: FailureReason,
        factors: List[str],
    ) -> str:
        """
        Generate lessons learned from failure.
        
        Args:
            reason: Failure reason
            factors: Contributing factors
            
        Returns:
            Lessons learned string
        """
        lessons = []
        
        if reason == FailureReason.STOP_LOSS_TOO_TIGHT:
            lessons.append("Stop loss should be placed further from entry to avoid noise")
        elif reason == FailureReason.TARGET_TOO_AGGRESSIVE:
            lessons.append("Targets should be more conservative based on historical volatility")
        elif reason == FailureReason.ENTRY_TOO_LATE:
            lessons.append("Entry timing needs improvement - consider limit orders")
        elif reason == FailureReason.WRONG_TREND:
            lessons.append("Trend detection failed - review trend indicators")
        elif reason == FailureReason.WRONG_REGIME:
            lessons.append("Regime mismatch - avoid predictions in unfavorable regimes")
        
        if "Low confidence prediction" in factors:
            lessons.append("Low confidence predictions should be filtered out")
        
        if "Low predicted probability" in factors:
            lessons.append("Probability threshold should be increased")
        
        return ". ".join(lessons) if lessons else "No specific lessons identified"
    
    def _suggest_fixes(
        self,
        reason: FailureReason,
        factors: List[str],
    ) -> List[str]:
        """
        Suggest fixes to prevent similar failures.
        
        Args:
            reason: Failure reason
            factors: Contributing factors
            
        Returns:
            List of suggested fixes
        """
        fixes = []
        
        if reason == FailureReason.STOP_LOSS_TOO_TIGHT:
            fixes.append("Increase stop loss distance based on ATR")
            fixes.append("Use volatility-adjusted stop losses")
        elif reason == FailureReason.TARGET_TOO_AGGRESSIVE:
            fixes.append("Set targets based on historical average moves")
            fixes.append("Use partial profit taking at multiple levels")
        elif reason == FailureReason.ENTRY_TOO_LATE:
            fixes.append("Implement limit orders for better entry")
            fixes.append("Reduce slippage tolerance")
        elif reason == FailureReason.WRONG_TREND:
            fixes.append("Improve trend detection algorithms")
            fixes.append("Add confirmation indicators")
        elif reason == FailureReason.WRONG_REGIME:
            fixes.append("Add regime filter to prediction pipeline")
            fixes.append("Reduce position size in unfavorable regimes")
        
        if "Low confidence prediction" in factors:
            fixes.append("Implement minimum confidence threshold")
        
        if "Low predicted probability" in factors:
            fixes.append("Increase minimum probability threshold")
        
        if "Insufficient signals" in factors:
            fixes.append("Require minimum signal count for predictions")
        
        return fixes
    
    def _is_preventable(self, reason: FailureReason) -> bool:
        """
        Determine if failure was preventable.
        
        Args:
            reason: Failure reason
            
        Returns:
            True if preventable
        """
        preventable_reasons = {
            FailureReason.STOP_LOSS_TOO_TIGHT,
            FailureReason.TARGET_TOO_AGGRESSIVE,
            FailureReason.ENTRY_TOO_LATE,
            FailureReason.WRONG_REGIME,
            FailureReason.LOW_VOLUME,
        }
        
        return reason in preventable_reasons
    
    def get_failure_statistics(
        self,
        failures: List[FailureAnalysis],
    ) -> Dict:
        """
        Get statistics about failures.
        
        Args:
            failures: List of FailureAnalysis objects
            
        Returns:
            Dictionary with failure statistics
        """
        if not failures:
            return {
                "total_failures": 0,
                "by_reason": {},
                "preventable_count": 0,
                "preventable_percentage": 0.0,
            }
        
        total = len(failures)
        
        # Count by reason
        by_reason = {}
        for failure in failures:
            reason = failure.failure_reason.value
            by_reason[reason] = by_reason.get(reason, 0) + 1
        
        # Count preventable
        preventable_count = sum(1 for f in failures if f.preventable)
        
        return {
            "total_failures": total,
            "by_reason": by_reason,
            "preventable_count": preventable_count,
            "preventable_percentage": preventable_count / total,
        }


def analyze_failure(
    prediction: PredictionMetadata,
    result: PredictionResult,
    market_conditions_at_failure: Optional[Dict] = None,
) -> FailureAnalysis:
    """
    Convenience function to analyze a failure.
    
    Args:
        prediction: Original prediction metadata
        result: Prediction result
        market_conditions_at_failure: Optional market conditions at failure
        
    Returns:
        FailureAnalysis
    """
    analyzer = FailureAnalyzer()
    return analyzer.analyze_failure(prediction, result, market_conditions_at_failure)
