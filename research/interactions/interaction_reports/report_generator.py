"""
Interaction Report Generator

Generates reports on factor performance under different conditions.
Provides insights on when factors work best and worst.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

from research.interactions.interaction_engine.interaction_engine import InteractionResult
from research.interactions.interaction_evaluator import EvaluationResult
from research.interactions.condition_engine.condition import Condition
from utils.logger import get_logger

logger = get_logger("research.interactions.interaction_reports")


@dataclass
class FactorReport:
    """Report for a single factor."""
    factor_name: str
    works_best: List[dict]
    works_worst: List[dict]
    recommendation: str
    summary: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "factor_name": self.factor_name,
            "works_best": self.works_best,
            "works_worst": self.works_worst,
            "recommendation": self.recommendation,
            "summary": self.summary,
        }


class ReportGenerator:
    """
    Generates reports on factor performance under different conditions.
    
    Reports include:
    - Best conditions for each factor
    - Worst conditions for each factor
    - Recommendations for usage
    - Performance summaries
    """
    
    def __init__(self):
        """Initialize report generator."""
        self._logger = get_logger("research.interactions.interaction_reports")
    
    def generate_factor_report(
        self,
        factor_name: str,
        results: List[InteractionResult],
        evaluations: Optional[List[EvaluationResult]] = None,
    ) -> FactorReport:
        """
        Generate report for a single factor.
        
        Args:
            factor_name: Name of the factor
            results: List of InteractionResult for the factor
            evaluations: Optional list of EvaluationResult
            
        Returns:
            FactorReport
        """
        # Filter results for this factor
        factor_results = [r for r in results if r.factor_name == factor_name]
        
        # Sort by IC
        sorted_results = sorted(factor_results, key=lambda x: x.ic, reverse=True)
        
        # Get best conditions (top 3 PASS)
        best_conditions = [
            {
                "condition": r.condition.serialize(),
                "ic": round(r.ic, 4),
                "sharpe": round(r.sharpe, 4),
                "win_rate": round(r.win_rate, 4),
                "num_trades": r.num_trades,
            }
            for r in sorted_results
            if r.decision == "PASS"
        ][:3]
        
        # Get worst conditions (bottom 3 FAIL)
        worst_conditions = [
            {
                "condition": r.condition.serialize(),
                "ic": round(r.ic, 4),
                "sharpe": round(r.sharpe, 4),
                "win_rate": round(r.win_rate, 4),
                "num_trades": r.num_trades,
            }
            for r in sorted_results
            if r.decision == "FAIL"
        ][-3:]
        
        # Generate recommendation
        recommendation = self._generate_recommendation(best_conditions, worst_conditions)
        
        # Generate summary
        summary = self._generate_summary(factor_name, best_conditions, worst_conditions)
        
        return FactorReport(
            factor_name=factor_name,
            works_best=best_conditions,
            works_worst=worst_conditions,
            recommendation=recommendation,
            summary=summary,
        )
    
    def _generate_recommendation(
        self,
        best_conditions: List[dict],
        worst_conditions: List[dict],
    ) -> str:
        """Generate recommendation based on best/worst conditions."""
        if not best_conditions:
            return "Factor does not perform well under any tested conditions. Not recommended for use."
        
        # Extract common patterns from best conditions
        best_trends = [c["condition"].get("trend") for c in best_conditions if c["condition"].get("trend")]
        best_volatilities = [c["condition"].get("volatility") for c in best_conditions if c["condition"].get("volatility")]
        
        # Find most common values
        common_trend = max(set(best_trends), key=best_trends.count) if best_trends else None
        common_volatility = max(set(best_volatilities), key=best_volatilities.count) if best_volatilities else None
        
        recommendation_parts = []
        
        if common_trend:
            recommendation_parts.append(f"Use primarily in {common_trend} markets")
        
        if common_volatility:
            recommendation_parts.append(f"when volatility is {common_volatility}")
        
        if worst_conditions:
            worst_trends = [c["condition"].get("trend") for c in worst_conditions if c["condition"].get("trend")]
            if worst_trends:
                worst_trend = max(set(worst_trends), key=worst_trends.count)
                if worst_trend != common_trend:
                    recommendation_parts.append(f"Avoid in {worst_trend} markets")
        
        if recommendation_parts:
            return ". ".join(recommendation_parts) + "."
        else:
            return "Use with caution. Limited clear pattern in test results."
    
    def _generate_summary(
        self,
        factor_name: str,
        best_conditions: List[dict],
        worst_conditions: List[dict],
    ) -> str:
        """Generate summary of factor performance."""
        lines = [f"Factor: {factor_name}"]
        
        if best_conditions:
            lines.append("\nWorks Best:")
            for i, cond in enumerate(best_conditions, 1):
                desc = self._condition_description(cond["condition"])
                lines.append(f"  {i}. {desc} (IC: {cond['ic']}, Sharpe: {cond['sharpe']})")
        
        if worst_conditions:
            lines.append("\nWorks Worst:")
            for i, cond in enumerate(worst_conditions, 1):
                desc = self._condition_description(cond["condition"])
                lines.append(f"  {i}. {desc} (IC: {cond['ic']}, Sharpe: {cond['sharpe']})")
        
        return "\n".join(lines)
    
    def _condition_description(self, condition: dict) -> str:
        """Generate human-readable condition description."""
        parts = []
        
        if condition.get("trend"):
            parts.append(f"Trend={condition['trend']}")
        if condition.get("volatility"):
            parts.append(f"Volatility={condition['volatility']}")
        if condition.get("sector"):
            parts.append(f"Sector={condition['sector']}")
        if condition.get("liquidity"):
            parts.append(f"Liquidity={condition['liquidity']}")
        if condition.get("market_breadth"):
            parts.append(f"Breadth={condition['market_breadth']}")
        if condition.get("options_sentiment"):
            parts.append(f"Options={condition['options_sentiment']}")
        
        return " + ".join(parts) if parts else "Any Condition"
    
    def generate_batch_report(
        self,
        all_results: List[InteractionResult],
    ) -> Dict[str, FactorReport]:
        """
        Generate reports for all factors.
        
        Args:
            all_results: List of all InteractionResult
            
        Returns:
            Dictionary mapping factor names to FactorReport
        """
        # Get unique factor names
        factor_names = list(set(r.factor_name for r in all_results))
        
        reports = {}
        for factor_name in factor_names:
            report = self.generate_factor_report(factor_name, all_results)
            reports[factor_name] = report
        
        return reports
    
    def generate_failure_analysis(
        self,
        results: List[InteractionResult],
    ) -> Dict:
        """
        Analyze factor failures.
        
        Args:
            results: List of InteractionResult
            
        Returns:
            Failure analysis dictionary
        """
        failures = [r for r in results if r.decision == "FAIL"]
        
        # Group by factor
        failures_by_factor = {}
        for failure in failures:
            if failure.factor_name not in failures_by_factor:
                failures_by_factor[failure.factor_name] = []
            failures_by_factor[failure.factor_name].append(failure)
        
        # Analyze each factor's failures
        analysis = {}
        for factor_name, factor_failures in failures_by_factor.items():
            # Find common failure conditions
            failure_conditions = [f.condition.serialize() for f in factor_failures]
            
            # Count condition patterns
            trend_counts = {}
            volatility_counts = {}
            
            for cond in failure_conditions:
                if cond.get("trend"):
                    trend_counts[cond["trend"]] = trend_counts.get(cond["trend"], 0) + 1
                if cond.get("volatility"):
                    volatility_counts[cond["volatility"]] = volatility_counts.get(cond["volatility"], 0) + 1
            
            analysis[factor_name] = {
                "total_failures": len(factor_failures),
                "common_failure_trends": trend_counts,
                "common_failure_volatilities": volatility_counts,
                "failure_conditions": failure_conditions[:5],  # Top 5
            }
        
        return analysis
    
    def generate_summary_report(
        self,
        reports: Dict[str, FactorReport],
    ) -> str:
        """
        Generate overall summary report.
        
        Args:
            reports: Dictionary of FactorReport
            
        Returns:
            Formatted summary string
        """
        lines = ["INTERACTION ANALYSIS SUMMARY"]
        lines.append("=" * 50)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total Factors Analyzed: {len(reports)}")
        lines.append("")
        
        for factor_name, report in reports.items():
            lines.append(f"\n{factor_name.upper()}")
            lines.append("-" * len(factor_name))
            lines.append(report.summary)
            lines.append(f"\nRecommendation: {report.recommendation}")
            lines.append("")
        
        return "\n".join(lines)


def generate_factor_report(
    factor_name: str,
    results: List[InteractionResult],
) -> FactorReport:
    """
    Convenience function to generate factor report.
    
    Args:
        factor_name: Name of the factor
        results: List of InteractionResult
        
    Returns:
        FactorReport
    """
    generator = ReportGenerator()
    return generator.generate_factor_report(factor_name, results)
