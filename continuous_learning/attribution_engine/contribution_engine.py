"""
Contribution Engine

Calculates and aggregates factor contributions across multiple trades.
Provides structured JSON output for analysis.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import json

from continuous_learning.attribution_engine.factor_attribution import AttributionResult, FactorContribution
from utils.logger import get_logger

logger = get_logger("continuous_learning.attribution_engine")


@dataclass
class ContributionSummary:
    """Summary of factor contributions."""
    total_trades: int
    total_return: float
    factor_contributions: Dict[str, Dict]
    top_positive_factors: List[tuple]
    top_negative_factors: List[tuple]
    attribution_accuracy: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "total_trades": self.total_trades,
            "total_return": round(self.total_return, 4),
            "factor_contributions": {
                k: {k2: round(v2, 4) if isinstance(v2, float) else v2 for k2, v2 in v.items()}
                for k, v in self.factor_contributions.items()
            },
            "top_positive_factors": [(k, round(v, 4)) for k, v in self.top_positive_factors],
            "top_negative_factors": [(k, round(v, 4)) for k, v in self.top_negative_factors],
            "attribution_accuracy": round(self.attribution_accuracy, 4),
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class ContributionEngine:
    """
    Calculates and aggregates factor contributions across multiple trades.
    
    Provides:
    - Per-factor contribution summaries
    - Top positive/negative contributors
    - Attribution accuracy
    - Structured JSON output
    """
    
    def __init__(self):
        """Initialize contribution engine."""
        self._logger = get_logger("continuous_learning.attribution_engine")
    
    def calculate_summary(
        self,
        attributions: List[AttributionResult],
    ) -> ContributionSummary:
        """
        Calculate contribution summary from attributions.
        
        Args:
            attributions: List of AttributionResult
            
        Returns:
            ContributionSummary
        """
        if not attributions:
            return ContributionSummary(
                total_trades=0,
                total_return=0.0,
                factor_contributions={},
                top_positive_factors=[],
                top_negative_factors=[],
                attribution_accuracy=0.0,
            )
        
        # Aggregate factor contributions
        factor_stats = self._aggregate_contributions(attributions)
        
        # Calculate total return
        total_return = sum(a.actual_return for a in attributions)
        
        # Calculate attribution accuracy
        attribution_accuracy = sum(a.attribution_accuracy for a in attributions) / len(attributions)
        
        # Get top positive and negative factors
        top_positive = self._get_top_factors(factor_stats, "positive")
        top_negative = self._get_top_factors(factor_stats, "negative")
        
        return ContributionSummary(
            total_trades=len(attributions),
            total_return=total_return,
            factor_contributions=factor_stats,
            top_positive_factors=top_positive,
            top_negative_factors=top_negative,
            attribution_accuracy=attribution_accuracy,
        )
    
    def _aggregate_contributions(
        self,
        attributions: List[AttributionResult],
    ) -> Dict[str, Dict]:
        """
        Aggregate factor contributions across attributions.
        
        Args:
            attributions: List of AttributionResult
            
        Returns:
            Dictionary mapping factor names to contribution stats
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
                        "total_contribution": 0.0,
                        "positive_count": 0,
                        "negative_count": 0,
                        "neutral_count": 0,
                        "correct_count": 0,
                        "total_count": 0,
                        "avg_contribution": 0.0,
                        "accuracy": 0.0,
                    }
                
                stats = factor_stats[factor_name]
                stats["total_contribution"] += contributor.contribution_score
                stats["total_count"] += 1
                
                if contributor.was_correct:
                    stats["correct_count"] += 1
                
                if contributor.attribution == "positive":
                    stats["positive_count"] += 1
                elif contributor.attribution == "negative":
                    stats["negative_count"] += 1
                else:
                    stats["neutral_count"] += 1
        
        # Calculate averages
        for factor_name, stats in factor_stats.items():
            if stats["total_count"] > 0:
                stats["avg_contribution"] = stats["total_contribution"] / stats["total_count"]
                stats["accuracy"] = stats["correct_count"] / stats["total_count"]
        
        return factor_stats
    
    def _get_top_factors(
        self,
        factor_stats: Dict[str, Dict],
        attribution_type: str,
        n: int = 5,
    ) -> List[tuple]:
        """
        Get top factors by attribution type.
        
        Args:
            factor_stats: Factor statistics
            attribution_type: "positive" or "negative"
            n: Number of top factors to return
            
        Returns:
            List of (factor_name, avg_contribution) tuples
        """
        if attribution_type == "positive":
            # Sort by average contribution (highest first)
            sorted_factors = sorted(
                factor_stats.items(),
                key=lambda x: x[1]["avg_contribution"],
                reverse=True,
            )
        else:
            # Sort by average contribution (lowest first)
            sorted_factors = sorted(
                factor_stats.items(),
                key=lambda x: x[1]["avg_contribution"],
            )
        
        return [(name, stats["avg_contribution"]) for name, stats in sorted_factors[:n]]
    
    def generate_report(
        self,
        summary: ContributionSummary,
    ) -> str:
        """
        Generate human-readable report.
        
        Args:
            summary: ContributionSummary
            
        Returns:
            Formatted report string
        """
        lines = []
        
        lines.append("FACTOR CONTRIBUTION REPORT")
        lines.append("=" * 50)
        lines.append(f"Total Trades: {summary.total_trades}")
        lines.append(f"Total Return: {summary.total_return:.2%}")
        lines.append(f"Attribution Accuracy: {summary.attribution_accuracy:.2%}")
        lines.append("")
        
        lines.append("TOP POSITIVE CONTRIBUTORS")
        for factor, contribution in summary.top_positive_factors:
            stats = summary.factor_contributions.get(factor, {})
            lines.append(f"  {factor}: +{contribution:.4f} (Accuracy: {stats.get('accuracy', 0):.2%})")
        
        lines.append("")
        lines.append("TOP NEGATIVE CONTRIBUTORS")
        for factor, contribution in summary.top_negative_factors:
            stats = summary.factor_contributions.get(factor, {})
            lines.append(f"  {factor}: {contribution:.4f} (Accuracy: {stats.get('accuracy', 0):.2%})")
        
        lines.append("")
        lines.append("DETAILED FACTOR STATISTICS")
        for factor, stats in summary.factor_contributions.items():
            lines.append(f"  {factor}:")
            lines.append(f"    Avg Contribution: {stats['avg_contribution']:.4f}")
            lines.append(f"    Accuracy: {stats['accuracy']:.2%}")
            lines.append(f"    Positive: {stats['positive_count']}, Negative: {stats['negative_count']}, Neutral: {stats['neutral_count']}")
        
        return "\n".join(lines)
    
    def compare_periods(
        self,
        period1_attributions: List[AttributionResult],
        period2_attributions: List[AttributionResult],
        period1_name: str = "Period 1",
        period2_name: str = "Period 2",
    ) -> Dict:
        """
        Compare factor contributions between two periods.
        
        Args:
            period1_attributions: Attributions for period 1
            period2_attributions: Attributions for period 2
            period1_name: Name for period 1
            period2_name: Name for period 2
            
        Returns:
            Comparison dictionary
        """
        summary1 = self.calculate_summary(period1_attributions)
        summary2 = self.calculate_summary(period2_attributions)
        
        comparison = {
            period1_name: summary1.to_dict(),
            period2_name: summary2.to_dict(),
            "changes": {},
        }
        
        # Calculate changes
        all_factors = set(summary1.factor_contributions.keys()) | set(summary2.factor_contributions.keys())
        
        for factor in all_factors:
            stats1 = summary1.factor_contributions.get(factor, {"avg_contribution": 0.0})
            stats2 = summary2.factor_contributions.get(factor, {"avg_contribution": 0.0})
            
            change = stats2["avg_contribution"] - stats1["avg_contribution"]
            comparison["changes"][factor] = {
                "change": round(change, 4),
                "period1": round(stats1["avg_contribution"], 4),
                "period2": round(stats2["avg_contribution"], 4),
            }
        
        return comparison


def calculate_contribution_summary(
    attributions: List[AttributionResult],
) -> ContributionSummary:
    """
    Convenience function to calculate contribution summary.
    
    Args:
        attributions: List of AttributionResult
        
    Returns:
        ContributionSummary
    """
    engine = ContributionEngine()
    return engine.calculate_summary(attributions)
