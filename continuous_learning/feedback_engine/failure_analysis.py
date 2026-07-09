"""
Failure Analysis

Analyzes failed trades to understand why they failed.
Identifies patterns and recommends improvements.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

from continuous_learning.outcome_engine.trade_outcome import TradeOutcome
from continuous_learning.attribution_engine.factor_attribution import AttributionResult
from utils.logger import get_logger

logger = get_logger("continuous_learning.feedback_engine")


class FailureReason(Enum):
    """Enumeration of failure reasons."""
    MARKET_REGIME_MISMATCH = "market_regime_mismatch"
    FACTOR_WEAKNESS = "factor_weakness"
    POOR_EXECUTION = "poor_execution"
    EXTERNAL_SHOCK = "external_shock"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CORRELATED_FACTORS = "correlated_factors"
    MODEL_DECAY = "model_decay"
    UNKNOWN = "unknown"


@dataclass
class FailureAnalysis:
    """Analysis of a failed trade."""
    prediction_id: str
    symbol: str
    failure_reason: str
    factor_performance: Dict[str, bool]
    market_regime: Optional[str]
    confidence_level: str
    recommendations: List[str]
    lessons_learned: str
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate failure analysis.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check failure reason is valid
        valid_reasons = [r.value for r in FailureReason]
        if self.failure_reason not in valid_reasons:
            errors.append(f"Invalid failure reason: {self.failure_reason}")
        
        # Check recommendations is not empty
        if not self.recommendations:
            errors.append("Recommendations cannot be empty")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "prediction_id": self.prediction_id,
            "symbol": self.symbol,
            "failure_reason": self.failure_reason,
            "factor_performance": self.factor_performance,
            "market_regime": self.market_regime,
            "confidence_level": self.confidence_level,
            "recommendations": self.recommendations,
            "lessons_learned": self.lessons_learned,
        }


class FailureAnalyzer:
    """
    Analyzes failed trades to understand why they failed.
    
    Identifies:
    - Market regime mismatches
    - Factor weaknesses
    - Execution issues
    - External shocks
    - Evidence insufficiency
    - Correlated factors
    - Model decay
    """
    
    def __init__(self):
        """Initialize failure analyzer."""
        self._logger = get_logger("continuous_learning.feedback_engine")
    
    def analyze(
        self,
        trade_outcome: TradeOutcome,
        attribution: Optional[AttributionResult] = None,
        market_regime: Optional[str] = None,
    ) -> FailureAnalysis:
        """
        Analyze a failed trade.
        
        Args:
            trade_outcome: TradeOutcome that failed
            attribution: Optional AttributionResult
            market_regime: Optional market regime
            
        Returns:
            FailureAnalysis
        """
        # Determine failure reason
        failure_reason = self._determine_failure_reason(
            trade_outcome,
            attribution,
            market_regime,
        )
        
        # Analyze factor performance
        factor_performance = self._analyze_factor_performance(attribution)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            failure_reason,
            factor_performance,
            trade_outcome,
        )
        
        # Generate lessons learned
        lessons_learned = self._generate_lessons_learned(
            failure_reason,
            factor_performance,
            trade_outcome,
        )
        
        return FailureAnalysis(
            prediction_id=trade_outcome.prediction_id,
            symbol=trade_outcome.symbol,
            failure_reason=failure_reason,
            factor_performance=factor_performance,
            market_regime=market_regime,
            confidence_level=trade_outcome.predicted_confidence,
            recommendations=recommendations,
            lessons_learned=lessons_learned,
        )
    
    def _determine_failure_reason(
        self,
        trade_outcome: TradeOutcome,
        attribution: Optional[AttributionResult],
        market_regime: Optional[str],
    ) -> str:
        """
        Determine the primary failure reason.
        
        Args:
            trade_outcome: TradeOutcome
            attribution: Optional AttributionResult
            market_regime: Optional market regime
            
        Returns:
            Failure reason string
        """
        # Check for market regime mismatch
        if market_regime and attribution:
            regime_mismatch = self._check_regime_mismatch(attribution, market_regime)
            if regime_mismatch:
                return FailureReason.MARKET_REGIME_MISMATCH.value
        
        # Check for factor weakness
        if attribution:
            factor_weakness = self._check_factor_weakness(attribution)
            if factor_weakness:
                return FailureReason.FACTOR_WEAKNESS.value
        
        # Check for low confidence
        if trade_outcome.predicted_confidence == "LOW":
            return FailureReason.INSUFFICIENT_EVIDENCE.value
        
        # Check for poor execution (stop hit quickly)
        if trade_outcome.resolved_outcome.outcome_type == "stop_hit":
            if trade_outcome.resolved_outcome.holding_period_days and trade_outcome.resolved_outcome.holding_period_days < 2:
                return FailureReason.POOR_EXECUTION.value
        
        # Check for correlated factors
        if attribution:
            correlated = self._check_correlated_factors(attribution)
            if correlated:
                return FailureReason.CORRELATED_FACTORS.value
        
        # Default to unknown
        return FailureReason.UNKNOWN.value
    
    def _check_regime_mismatch(
        self,
        attribution: AttributionResult,
        market_regime: str,
    ) -> bool:
        """
        Check if failure was due to regime mismatch.
        
        Args:
            attribution: AttributionResult
            market_regime: Market regime
            
        Returns:
            True if regime mismatch detected
        """
        # Check if most factors performed poorly in this regime
        negative_count = len(attribution.negative_contributors)
        total_count = len(attribution.positive_contributors) + len(attribution.negative_contributors)
        
        if total_count > 0:
            negative_ratio = negative_count / total_count
            if negative_ratio > 0.7:
                return True
        
        return False
    
    def _check_factor_weakness(self, attribution: AttributionResult) -> bool:
        """
        Check if failure was due to factor weakness.
        
        Args:
            attribution: AttributionResult
            
        Returns:
            True if factor weakness detected
        """
        # Check if attribution accuracy is low
        if attribution.attribution_accuracy < 0.5:
            return True
        
        return False
    
    def _check_correlated_factors(self, attribution: AttributionResult) -> bool:
        """
        Check if failure was due to correlated factors.
        
        Args:
            attribution: AttributionResult
            
        Returns:
            True if correlated factors detected
        """
        # Check if all factors moved in same direction
        all_contributors = (
            attribution.positive_contributors +
            attribution.negative_contributors
        )
        
        if len(all_contributors) > 0:
            directions = [c.direction for c in all_contributors]
            unique_directions = set(directions)
            
            # If all factors have same direction, they may be correlated
            if len(unique_directions) == 1:
                return True
        
        return False
    
    def _analyze_factor_performance(
        self,
        attribution: Optional[AttributionResult],
    ) -> Dict[str, bool]:
        """
        Analyze performance of individual factors.
        
        Args:
            attribution: Optional AttributionResult
            
        Returns:
            Dictionary mapping factor names to performance (True/False)
        """
        factor_performance = {}
        
        if attribution:
            all_contributors = (
                attribution.positive_contributors +
                attribution.negative_contributors +
                attribution.neutral_contributors
            )
            
            for contributor in all_contributors:
                factor_performance[contributor.factor_name] = contributor.was_correct
        
        return factor_performance
    
    def _generate_recommendations(
        self,
        failure_reason: str,
        factor_performance: Dict[str, bool],
        trade_outcome: TradeOutcome,
    ) -> List[str]:
        """
        Generate recommendations based on failure analysis.
        
        Args:
            failure_reason: Failure reason
            factor_performance: Factor performance
            trade_outcome: TradeOutcome
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if failure_reason == FailureReason.MARKET_REGIME_MISMATCH.value:
            recommendations.append("Reduce exposure in current market regime")
            recommendations.append("Consider regime-specific factor selection")
        
        elif failure_reason == FailureReason.FACTOR_WEAKNESS.value:
            # Identify weak factors
            weak_factors = [f for f, perf in factor_performance.items() if not perf]
            if weak_factors:
                recommendations.append(f"Reduce weight for: {', '.join(weak_factors)}")
            recommendations.append("Review factor calculation methodology")
        
        elif failure_reason == FailureReason.POOR_EXECUTION.value:
            recommendations.append("Review entry timing")
            recommendations.append("Consider wider stop loss")
        
        elif failure_reason == FailureReason.INSUFFICIENT_EVIDENCE.value:
            recommendations.append("Require higher confidence for trades")
            recommendations.append("Increase minimum evidence threshold")
        
        elif failure_reason == FailureReason.CORRELATED_FACTORS.value:
            recommendations.append("Diversify factor sources")
            recommendations.append("Reduce weight for correlated factors")
        
        elif failure_reason == FailureReason.MODEL_DECAY.value:
            recommendations.append("Consider model retraining")
            recommendations.append("Review factor performance over time")
        
        else:
            recommendations.append("Review trade parameters")
            recommendations.append("Investigate external factors")
        
        return recommendations
    
    def _generate_lessons_learned(
        self,
        failure_reason: str,
        factor_performance: Dict[str, bool],
        trade_outcome: TradeOutcome,
    ) -> str:
        """
        Generate lessons learned from failure.
        
        Args:
            failure_reason: Failure reason
            factor_performance: Factor performance
            trade_outcome: TradeOutcome
            
        Returns:
            Lessons learned string
        """
        parts = []
        
        parts.append(f"Trade failed due to: {failure_reason}")
        
        # Add factor performance summary
        if factor_performance:
            correct_count = sum(1 for perf in factor_performance.values() if perf)
            total_count = len(factor_performance)
            parts.append(f"Factor accuracy: {correct_count}/{total_count}")
        
        # Add outcome details
        parts.append(f"Outcome: {trade_outcome.resolved_outcome.outcome_type}")
        parts.append(f"Return: {trade_outcome.resolved_outcome.return_percentage:.2%}")
        
        return " | ".join(parts)
    
    def batch_analyze(
        self,
        failed_trades: List[TradeOutcome],
        attributions_by_prediction: Optional[Dict[str, AttributionResult]] = None,
        market_regimes_by_prediction: Optional[Dict[str, str]] = None,
    ) -> List[FailureAnalysis]:
        """
        Analyze multiple failed trades.
        
        Args:
            failed_trades: List of failed TradeOutcome
            attributions_by_prediction: Optional dictionary of attributions
            market_regimes_by_prediction: Optional dictionary of market regimes
            
        Returns:
            List of FailureAnalysis
        """
        analyses = []
        
        for trade_outcome in failed_trades:
            attribution = attributions_by_prediction.get(trade_outcome.prediction_id) if attributions_by_prediction else None
            market_regime = market_regimes_by_prediction.get(trade_outcome.prediction_id) if market_regimes_by_prediction else None
            
            try:
                analysis = self.analyze(trade_outcome, attribution, market_regime)
                analyses.append(analysis)
            except Exception as e:
                self._logger.error(f"Failed to analyze {trade_outcome.prediction_id}: {e}")
        
        return analyses
    
    def aggregate_failure_patterns(self, analyses: List[FailureAnalysis]) -> Dict:
        """
        Aggregate failure patterns across multiple analyses.
        
        Args:
            analyses: List of FailureAnalysis
            
        Returns:
            Dictionary with failure pattern statistics
        """
        # Count failure reasons
        reason_counts = {}
        
        # Count factor failures
        factor_failures = {}
        
        for analysis in analyses:
            # Count reasons
            reason = analysis.failure_reason
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            
            # Count factor failures
            for factor, was_correct in analysis.factor_performance.items():
                if not was_correct:
                    factor_failures[factor] = factor_failures.get(factor, 0) + 1
        
        return {
            "total_failures": len(analyses),
            "failure_reasons": reason_counts,
            "factor_failures": factor_failures,
        }


def analyze_failure(
    trade_outcome: TradeOutcome,
    attribution: Optional[AttributionResult] = None,
) -> FailureAnalysis:
    """
    Convenience function to analyze failure.
    
    Args:
        trade_outcome: TradeOutcome that failed
        attribution: Optional AttributionResult
        
    Returns:
        FailureAnalysis
    """
    analyzer = FailureAnalyzer()
    return analyzer.analyze(trade_outcome, attribution)
