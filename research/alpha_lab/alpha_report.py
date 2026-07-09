"""
Alpha Report Generator

Generates comprehensive reports for factor evaluation.
Combines validation results, performance metrics, and rankings into actionable reports.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.logger import get_logger

logger = get_logger("research.alpha_report")


@dataclass
class FactorReport:
    """Comprehensive report for a single factor."""
    factor_name: str
    metadata: Dict[str, Any]
    validation_result: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    ranking: Dict[str, Any]
    recommendations: List[str]
    generated_at: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "factor_name": self.factor_name,
            "metadata": self.metadata,
            "validation_result": self.validation_result,
            "performance_metrics": self.performance_metrics,
            "ranking": self.ranking,
            "recommendations": self.recommendations,
            "generated_at": self.generated_at.isoformat(),
        }


class AlphaReportGenerator:
    """
    Generates comprehensive reports for factor evaluation.
    
    Reports include:
    - Factor metadata
    - Validation results (data quality, performance, significance)
    - Performance metrics (IC, Rank IC, hit rate, Sharpe)
    - Ranking (overall score, tier, rank)
    - Recommendations (promote, reject, research)
    """
    
    def __init__(self, output_path: str = "research/factor_reports"):
        """
        Initialize report generator.
        
        Args:
            output_path: Path to save reports
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self._logger = get_logger("research.alpha_report")
    
    def generate_factor_report(
        self,
        factor_name: str,
        metadata: Dict[str, Any],
        validation_result: Dict,
        performance_metrics: Dict,
        ranking: Dict,
    ) -> FactorReport:
        """
        Generate a comprehensive factor report.
        
        Args:
            factor_name: Name of factor
            metadata: Factor metadata
            validation_result: Validation result dictionary
            performance_metrics: Performance metrics dictionary
            ranking: Ranking dictionary
            
        Returns:
            FactorReport
        """
        # Generate recommendations
        recommendations = self._generate_recommendations(
            validation_result, performance_metrics, ranking
        )
        
        report = FactorReport(
            factor_name=factor_name,
            metadata=metadata,
            validation_result=validation_result,
            performance_metrics=performance_metrics,
            ranking=ranking,
            recommendations=recommendations,
            generated_at=datetime.now(),
        )
        
        return report
    
    def _generate_recommendations(
        self,
        validation_result: Dict,
        performance_metrics: Dict,
        ranking: Dict,
    ) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        
        # Validation recommendations
        if not validation_result.get("passed", False):
            recommendations.append("REJECT: Factor failed validation")
            if not validation_result.get("data_quality_passed", False):
                recommendations.append("  - Fix data quality issues (NaN, Inf, variance)")
            if not validation_result.get("performance_passed", False):
                recommendations.append("  - Improve performance metrics (IC, hit rate)")
            if not validation_result.get("significance_passed", False):
                recommendations.append("  - Improve statistical significance")
            return recommendations
        
        # Performance recommendations
        tier = ranking.get("tier", "D")
        overall_score = ranking.get("overall_score", 0)
        
        if tier in ["S", "A"]:
            recommendations.append("PROMOTE: Factor meets production criteria")
            if overall_score >= 0.9:
                recommendations.append("  - Excellent factor, consider for top-tier signals")
        elif tier == "B":
            recommendations.append("RESEARCH: Factor shows promise but needs improvement")
            recommendations.append("  - Consider parameter optimization")
            recommendations.append("  - Test on different timeframes/sectors")
        elif tier == "C":
            recommendations.append("RESEARCH: Factor marginal, needs significant improvement")
            recommendations.append("  - Consider combining with other factors")
        else:
            recommendations.append("REJECT: Factor does not meet minimum criteria")
        
        # Specific metric recommendations
        mean_ic = performance_metrics.get("mean_ic", 0)
        if mean_ic < 0.02:
            recommendations.append("  - IC too low, factor may not be predictive")
        
        hit_rate = performance_metrics.get("hit_rate", 0)
        if hit_rate < 0.51:
            recommendations.append("  - Hit rate below random, consider inverting signal")
        
        return recommendations
    
    def save_report(self, report: FactorReport) -> str:
        """
        Save report to disk.
        
        Args:
            report: FactorReport to save
            
        Returns:
            Path to saved report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report.factor_name}_{timestamp}.json"
        filepath = self.output_path / filename
        
        with open(filepath, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        
        self._logger.info(f"Saved report to {filepath}")
        return str(filepath)
    
    def generate_summary_report(
        self,
        reports: List[FactorReport],
    ) -> pd.DataFrame:
        """
        Generate summary report for multiple factors.
        
        Args:
            reports: List of FactorReport
            
        Returns:
            DataFrame with summary
        """
        data = []
        
        for report in reports:
            data.append({
                "factor_name": report.factor_name,
                "category": report.metadata.get("category", "unknown"),
                "timeframe": report.metadata.get("timeframe", "unknown"),
                "overall_score": report.ranking.get("overall_score", 0),
                "tier": report.ranking.get("tier", "D"),
                "rank": report.ranking.get("rank", 0),
                "mean_ic": report.performance_metrics.get("mean_ic", 0),
                "mean_rank_ic": report.performance_metrics.get("mean_rank_ic", 0),
                "hit_rate": report.performance_metrics.get("hit_rate", 0),
                "validation_passed": report.validation_result.get("passed", False),
                "recommendation": report.recommendations[0] if report.recommendations else "NONE",
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values("overall_score", ascending=False)
        
        return df
    
    def generate_markdown_report(
        self,
        report: FactorReport,
    ) -> str:
        """
        Generate markdown report for human reading.
        
        Args:
            report: FactorReport
            
        Returns:
            Markdown string
        """
        md = f"""
# Factor Report: {report.factor_name}

## Metadata
- **Name**: {report.metadata.get('name', 'N/A')}
- **Category**: {report.metadata.get('category', 'N/A')}
- **Version**: {report.metadata.get('version', 'N/A')}
- **Author**: {report.metadata.get('author', 'N/A')}
- **Timeframe**: {report.metadata.get('timeframe', 'N/A')}
- **Lookback**: {report.metadata.get('lookback', 'N/A')}
- **Prediction Horizon**: {report.metadata.get('prediction_horizon', 'N/A')}
- **Description**: {report.metadata.get('description', 'N/A')}

## Validation Results
- **Passed**: {report.validation_result.get('passed', False)}
- **Overall Score**: {report.validation_result.get('overall_score', 0):.4f}
- **Data Quality Score**: {report.validation_result.get('data_quality_score', 0):.4f}
- **Performance Score**: {report.validation_result.get('performance_score', 0):.4f}
- **Significance Score**: {report.validation_result.get('significance_score', 0):.4f}

### Errors
{chr(10).join(f"- {e}" for e in report.validation_result.get('errors', []))}

### Warnings
{chr(10).join(f"- {w}" for w in report.validation_result.get('warnings', []))}

## Performance Metrics
- **Mean IC**: {report.performance_metrics.get('mean_ic', 0):.4f}
- **Std IC**: {report.performance_metrics.get('std_ic', 0):.4f}
- **Mean Rank IC**: {report.performance_metrics.get('mean_rank_ic', 0):.4f}
- **Std Rank IC**: {report.performance_metrics.get('std_rank_ic', 0):.4f}
- **Hit Rate**: {report.performance_metrics.get('hit_rate', 0):.2%}
- **IC t-stat**: {report.performance_metrics.get('ic_t_stat', 0):.4f}
- **IC p-value**: {report.performance_metrics.get('ic_p_value', 0):.4f}
- **Sample Size**: {report.performance_metrics.get('sample_size', 0)}

## Ranking
- **Overall Score**: {report.ranking.get('overall_score', 0):.4f}
- **Alpha Score**: {report.ranking.get('alpha_score', 0):.4f}
- **Information Score**: {report.ranking.get('information_score', 0):.4f}
- **Stability Score**: {report.ranking.get('stability_score', 0):.4f}
- **Regime Score**: {report.ranking.get('regime_score', 0):.4f}
- **Sector Score**: {report.ranking.get('sector_score', 0):.4f}
- **Decay Score**: {report.ranking.get('decay_score', 0):.4f}
- **Tier**: {report.ranking.get('tier', 'D')}
- **Rank**: {report.ranking.get('rank', 0)}

## Recommendations
{chr(10).join(f"- {r}" for r in report.recommendations)}

---
Generated at: {report.generated_at.isoformat()}
"""
        return md.strip()
    
    def save_markdown_report(
        self,
        report: FactorReport,
    ) -> str:
        """
        Save markdown report to disk.
        
        Args:
            report: FactorReport
            
        Returns:
            Path to saved report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report.factor_name}_{timestamp}.md"
        filepath = self.output_path / filename
        
        with open(filepath, "w") as f:
            f.write(self.generate_markdown_report(report))
        
        self._logger.info(f"Saved markdown report to {filepath}")
        return str(filepath)
