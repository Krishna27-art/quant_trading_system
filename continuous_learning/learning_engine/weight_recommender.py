"""
Weight Recommender

Recommends weight changes based on historical outcomes and factor performance.
Does NOT modify production weights - only returns recommendations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from continuous_learning.outcome_engine.trade_outcome import TradeOutcome
from continuous_learning.attribution_engine.factor_attribution import AttributionResult
from continuous_learning.factor_evolution.regime_statistics import RegimeStatistics
from utils.logger import get_logger

logger = get_logger("continuous_learning.learning_engine")


@dataclass
class WeightRecommendation:
    """Weight change recommendation for a factor."""
    factor_name: str
    current_weight: float
    recommended_weight: float
    weight_change: float  # Percentage change
    confidence: float  # 0-1
    reason: str
    action: str  # "INCREASE", "DECREASE", "KEEP"
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate weight recommendation.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check weights are between 0 and 1
        if not (0.0 <= self.current_weight <= 1.0):
            errors.append(f"Current weight must be between 0 and 1, got {self.current_weight}")
        
        if not (0.0 <= self.recommended_weight <= 1.0):
            errors.append(f"Recommended weight must be between 0 and 1, got {self.recommended_weight}")
        
        # Check confidence is between 0 and 1
        if not (0.0 <= self.confidence <= 1.0):
            errors.append(f"Confidence must be between 0 and 1, got {self.confidence}")
        
        # Check action is valid
        valid_actions = ["INCREASE", "DECREASE", "KEEP"]
        if self.action not in valid_actions:
            errors.append(f"Invalid action: {self.action}")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "factor_name": self.factor_name,
            "current_weight": round(self.current_weight, 4),
            "recommended_weight": round(self.recommended_weight, 4),
            "weight_change": round(self.weight_change, 4),
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "action": self.action,
        }


@dataclass
class WeightRecommendationReport:
    """Complete weight recommendation report."""
    total_factors: int
    recommendations: List[WeightRecommendation]
    summary: Dict[str, int]
    requires_review: bool
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "total_factors": self.total_factors,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "summary": self.summary,
            "requires_review": self.requires_review,
        }


class WeightRecommender:
    """
    Recommends weight changes based on historical outcomes.
    
    Process:
    - Analyze factor performance
    - Check regime statistics
    - Calculate statistical significance
    - Recommend weight changes
    - DO NOT modify production weights
    """
    
    def __init__(
        self,
        min_trades: int = 50,
        significance_threshold: float = 0.05,
        max_weight_change: float = 0.2,
    ):
        """
        Initialize weight recommender.
        
        Args:
            min_trades: Minimum trades before recommending changes
            significance_threshold: Statistical significance threshold
            max_weight_change: Maximum weight change per recommendation
        """
        self.min_trades = min_trades
        self.significance_threshold = significance_threshold
        self.max_weight_change = max_weight_change
        self._logger = get_logger("continuous_learning.learning_engine")
    
    def recommend_weights(
        self,
        current_weights: Dict[str, float],
        trade_outcomes: List[TradeOutcome],
        attributions: List[AttributionResult],
        regime_stats: Optional[RegimeStatistics] = None,
        current_regime: str = "unknown",
    ) -> WeightRecommendationReport:
        """
        Generate weight recommendations.
        
        Args:
            current_weights: Current factor weights
            trade_outcomes: List of TradeOutcome
            attributions: List of AttributionResult
            regime_stats: Optional RegimeStatistics
            current_regime: Current market regime
            
        Returns:
            WeightRecommendationReport
        """
        recommendations = []
        
        # Calculate factor performance
        factor_performance = self._calculate_factor_performance(attributions)
        
        for factor_name, current_weight in current_weights.items():
            perf = factor_performance.get(factor_name)
            
            if not perf or perf["total_trades"] < self.min_trades:
                # Not enough data - keep current weight
                recommendations.append(WeightRecommendation(
                    factor_name=factor_name,
                    current_weight=current_weight,
                    recommended_weight=current_weight,
                    weight_change=0.0,
                    confidence=0.0,
                    reason="Insufficient data",
                    action="KEEP",
                ))
                continue
            
            # Get regime-specific performance if available
            regime_perf = None
            if regime_stats:
                regime_perf = regime_stats.get_regime_performance(current_regime, factor_name)
            
            # Calculate recommendation
            recommendation = self._calculate_recommendation(
                factor_name,
                current_weight,
                perf,
                regime_perf,
            )
            
            recommendations.append(recommendation)
        
        # Calculate summary
        summary = {
            "increase": sum(1 for r in recommendations if r.action == "INCREASE"),
            "decrease": sum(1 for r in recommendations if r.action == "DECREASE"),
            "keep": sum(1 for r in recommendations if r.action == "KEEP"),
        }
        
        # Determine if review required
        requires_review = summary["increase"] > 0 or summary["decrease"] > 0
        
        return WeightRecommendationReport(
            total_factors=len(recommendations),
            recommendations=recommendations,
            summary=summary,
            requires_review=requires_review,
        )
    
    def _calculate_factor_performance(
        self,
        attributions: List[AttributionResult],
    ) -> Dict[str, Dict]:
        """
        Calculate factor performance from attributions.
        
        Args:
            attributions: List of AttributionResult
            
        Returns:
            Dictionary mapping factor names to performance metrics
        """
        factor_stats = {}
        
        for attribution in attributions:
            all_contributors = (
                attribution.positive_contributors +
                attribution.negative_contributors +
                attribution.neutral_contributors
            )
            
            for contributor in all_contributors:
                factor_name = contributor.factor_name
                
                if factor_name not in factor_stats:
                    factor_stats[factor_name] = {
                        "total_trades": 0,
                        "correct_trades": 0,
                        "total_contribution": 0.0,
                    }
                
                stats = factor_stats[factor_name]
                stats["total_trades"] += 1
                stats["total_contribution"] += contributor.contribution_score
                
                if contributor.was_correct:
                    stats["correct_trades"] += 1
        
        # Calculate metrics
        for factor_name, stats in factor_stats.items():
            if stats["total_trades"] > 0:
                stats["accuracy"] = stats["correct_trades"] / stats["total_trades"]
                stats["avg_contribution"] = stats["total_contribution"] / stats["total_trades"]
            else:
                stats["accuracy"] = 0.0
                stats["avg_contribution"] = 0.0
        
        return factor_stats
    
    def _calculate_recommendation(
        self,
        factor_name: str,
        current_weight: float,
        perf: Dict,
        regime_perf: Optional,
    ) -> WeightRecommendation:
        """
        Calculate weight recommendation for a factor.
        
        Args:
            factor_name: Factor name
            current_weight: Current weight
            perf: Overall performance
            regime_perf: Regime-specific performance
            
        Returns:
            WeightRecommendation
        """
        accuracy = perf["accuracy"]
        avg_contribution = perf["avg_contribution"]
        
        # Use regime-specific performance if available and significant
        if regime_perf and regime_perf.trades >= self.min_trades:
            regime_accuracy = regime_perf.win_rate
            # Weight regime performance more heavily
            combined_accuracy = 0.7 * regime_accuracy + 0.3 * accuracy
        else:
            combined_accuracy = accuracy
        
        # Determine action
        if combined_accuracy > 0.6 and avg_contribution > 0.01:
            action = "INCREASE"
            weight_change = self.max_weight_change * (combined_accuracy - 0.5) * 2
            weight_change = min(weight_change, self.max_weight_change)
            confidence = combined_accuracy
            reason = f"High accuracy ({combined_accuracy:.2%}) and positive contribution"
        elif combined_accuracy < 0.4 or avg_contribution < -0.01:
            action = "DECREASE"
            weight_change = -self.max_weight_change * (0.5 - combined_accuracy) * 2
            weight_change = max(weight_change, -self.max_weight_change)
            confidence = 1.0 - combined_accuracy
            reason = f"Low accuracy ({combined_accuracy:.2%}) or negative contribution"
        else:
            action = "KEEP"
            weight_change = 0.0
            confidence = 0.5
            reason = "Performance within acceptable range"
        
        # Calculate recommended weight
        recommended_weight = current_weight + weight_change
        recommended_weight = max(0.0, min(1.0, recommended_weight))
        
        return WeightRecommendation(
            factor_name=factor_name,
            current_weight=current_weight,
            recommended_weight=recommended_weight,
            weight_change=weight_change,
            confidence=confidence,
            reason=reason,
            action=action,
        )
    
    def generate_report(self, report: WeightRecommendationReport) -> str:
        """
        Generate human-readable report.
        
        Args:
            report: WeightRecommendationReport
            
        Returns:
            Formatted report string
        """
        lines = []
        
        lines.append("WEIGHT RECOMMENDATION REPORT")
        lines.append("=" * 50)
        lines.append(f"Total Factors: {report.total_factors}")
        lines.append(f"Increase: {report.summary['increase']}")
        lines.append(f"Decrease: {report.summary['decrease']}")
        lines.append(f"Keep: {report.summary['keep']}")
        lines.append(f"Requires Review: {report.requires_review}")
        
        lines.append("\nRECOMMENDATIONS")
        lines.append("-" * 40)
        
        for rec in report.recommendations:
            lines.append(f"{rec.factor_name}:")
            lines.append(f"  Current: {rec.current_weight:.4f}")
            lines.append(f"  Recommended: {rec.recommended_weight:.4f}")
            lines.append(f"  Change: {rec.weight_change:+.4f}")
            lines.append(f"  Action: {rec.action}")
            lines.append(f"  Confidence: {rec.confidence:.2%}")
            lines.append(f"  Reason: {rec.reason}")
        
        lines.append("\nIMPORTANT: These are recommendations only.")
        lines.append("DO NOT modify production weights without review.")
        lines.append("Run backtests and walk-forward validation before promoting.")
        
        return "\n".join(lines)


def recommend_weights(
    current_weights: Dict[str, float],
    trade_outcomes: List[TradeOutcome],
    attributions: List[AttributionResult],
) -> WeightRecommendationReport:
    """
    Convenience function to recommend weights.
    
    Args:
        current_weights: Current factor weights
        trade_outcomes: List of TradeOutcome
        attributions: List of AttributionResult
        
    Returns:
        WeightRecommendationReport
    """
    recommender = WeightRecommender()
    return recommender.recommend_weights(current_weights, trade_outcomes, attributions)
