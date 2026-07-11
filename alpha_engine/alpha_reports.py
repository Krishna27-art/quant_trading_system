"""
Alpha Reports

Generates weekly research reports for alpha performance analysis.

STEP 14: Weekly Alpha Research

This module:
1. Generates weekly performance reports
2. Identifies top/worst performing categories
3. Analyzes best weight combinations
4. Tracks regime-specific performance
5. Provides actionable insights for improvement
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from alpha_engine.alpha_builder import AlphaGrade
from alpha_engine.alpha_tracker import AlphaTracker, GradePerformance
from utils.logger import get_logger

logger = get_logger("alpha_engine.reports")


@dataclass
class CategoryPerformanceReport:
    """
    Performance report for a single category.
    """
    category: str
    avg_score: float
    win_rate: float
    avg_return: float
    sharpe_ratio: float
    predictions_count: int
    rank: int  # Performance rank among categories
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "category": self.category,
            "avg_score": round(self.avg_score, 2),
            "win_rate": round(self.win_rate, 4),
            "avg_return": round(self.avg_return, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "predictions_count": self.predictions_count,
            "rank": self.rank,
        }


@dataclass
class RegimePerformanceReport:
    """
    Performance report for a specific regime.
    """
    regime: str
    predictions_count: int
    win_rate: float
    avg_return: float
    best_grade: AlphaGrade
    worst_grade: AlphaGrade
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "regime": self.regime,
            "predictions_count": self.predictions_count,
            "win_rate": round(self.win_rate, 4),
            "avg_return": round(self.avg_return, 4),
            "best_grade": self.best_grade.value,
            "worst_grade": self.worst_grade.value,
        }


@dataclass
class WeeklyAlphaReport:
    """
    Complete weekly alpha research report.
    """
    week_start: date
    week_end: date
    generated_at: datetime
    
    # Summary statistics
    total_predictions: int
    completed_predictions: int
    overall_win_rate: float
    overall_avg_return: float
    
    # Category performance
    top_categories: List[CategoryPerformanceReport]
    worst_categories: List[CategoryPerformanceReport]
    
    # Grade performance
    grade_performance: Dict[str, GradePerformance]
    
    # Regime performance
    regime_performance: List[RegimePerformanceReport]
    
    # Top performing stocks
    top_stocks: List[Dict[str, Any]]
    worst_predictions: List[Dict[str, Any]]
    
    # Insights and recommendations
    insights: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "week_start": self.week_start.isoformat(),
            "week_end": self.week_end.isoformat(),
            "generated_at": self.generated_at.isoformat(),
            "summary": {
                "total_predictions": self.total_predictions,
                "completed_predictions": self.completed_predictions,
                "overall_win_rate": round(self.overall_win_rate, 4),
                "overall_avg_return": round(self.overall_avg_return, 4),
            },
            "category_performance": {
                "top": [c.to_dict() for c in self.top_categories],
                "worst": [c.to_dict() for c in self.worst_categories],
            },
            "grade_performance": {
                grade: perf.to_dict() for grade, perf in self.grade_performance.items()
            },
            "regime_performance": [r.to_dict() for r in self.regime_performance],
            "stock_performance": {
                "top": self.top_stocks,
                "worst": self.worst_predictions,
            },
            "insights": self.insights,
            "recommendations": self.recommendations,
        }


class AlphaReports:
    """
    Generates weekly research reports for alpha performance.
    
    These reports feed back into the Feature Ranking and Signal Engine
    for continuous improvement.
    """
    
    def __init__(self, tracker: AlphaTracker):
        """
        Initialize Alpha Reports.
        
        Args:
            tracker: AlphaTracker instance with historical data
        """
        self._logger = logger
        self.tracker = tracker
    
    def generate_weekly_report(
        self,
        week_start: Optional[date] = None,
        week_end: Optional[date] = None,
    ) -> WeeklyAlphaReport:
        """
        Generate a weekly alpha research report.
        
        Args:
            week_start: Start of week (defaults to 7 days ago)
            week_end: End of week (defaults to today)
            
        Returns:
            WeeklyAlphaReport
        """
        if week_end is None:
            week_end = date.today()
        if week_start is None:
            week_start = week_end - timedelta(days=7)
        
        self._logger.info(f"Generating weekly report for {week_start} to {week_end}")
        
        # Get records for the week
        records = self.tracker.get_records_by_date_range(week_start, week_end)
        
        # Calculate summary statistics
        summary = self._calculate_summary(records)
        
        # Analyze category performance
        category_perf = self._analyze_category_performance(records)
        
        # Get grade performance
        grade_perf = self.tracker.get_grade_performance()
        
        # Analyze regime performance
        regime_perf = self._analyze_regime_performance(records)
        
        # Identify top and worst stocks
        top_stocks, worst_preds = self._analyze_stock_performance(records)
        
        # Generate insights and recommendations
        insights, recommendations = self._generate_insights(
            records,
            category_perf,
            regime_perf,
            grade_perf,
        )
        
        report = WeeklyAlphaReport(
            week_start=week_start,
            week_end=week_end,
            generated_at=datetime.now(),
            total_predictions=summary["total"],
            completed_predictions=summary["completed"],
            overall_win_rate=summary["win_rate"],
            overall_avg_return=summary["avg_return"],
            top_categories=category_perf["top"],
            worst_categories=category_perf["worst"],
            grade_performance=grade_perf,
            regime_performance=regime_perf,
            top_stocks=top_stocks,
            worst_predictions=worst_preds,
            insights=insights,
            recommendations=recommendations,
        )
        
        self._logger.info(f"Weekly report generated: {len(records)} records analyzed")
        
        return report
    
    def _calculate_summary(self, records: List) -> Dict[str, Any]:
        """Calculate summary statistics."""
        total = len(records)
        completed = [r for r in records if r.outcome.value != "pending"]
        
        if not completed:
            return {
                "total": total,
                "completed": 0,
                "win_rate": 0.0,
                "avg_return": 0.0,
            }
        
        correct = [r for r in completed if r.outcome.value == "correct"]
        win_rate = len(correct) / len(completed)
        
        returns = [r.actual_return for r in completed if r.actual_return is not None]
        avg_return = np.mean(returns) if returns else 0.0
        
        return {
            "total": total,
            "completed": len(completed),
            "win_rate": win_rate,
            "avg_return": avg_return,
        }
    
    def _analyze_category_performance(
        self,
        records: List,
    ) -> Dict[str, List[CategoryPerformanceReport]]:
        """Analyze performance by category."""
        if not records:
            return {"top": [], "worst": []}
        
        # Aggregate category scores and outcomes
        category_data: Dict[str, List[Dict]] = {}
        
        for record in records:
            if record.actual_return is None:
                continue
            
            for category, score in record.category_scores.items():
                if category not in category_data:
                    category_data[category] = []
                
                category_data[category].append({
                    "score": score,
                    "return": record.actual_return,
                    "correct": record.outcome.value == "correct",
                })
        
        # Calculate metrics for each category
        category_reports = []
        
        for category, data in category_data.items():
            scores = [d["score"] for d in data]
            returns = [d["return"] for d in data]
            correct = [d["correct"] for d in data]
            
            avg_score = np.mean(scores)
            win_rate = np.mean(correct)
            avg_return = np.mean(returns)
            sharpe = (
                np.mean(returns) / np.std(returns)
                if len(returns) > 1 and np.std(returns) > 0 else 0.0
            )
            
            report = CategoryPerformanceReport(
                category=category,
                avg_score=avg_score,
                win_rate=win_rate,
                avg_return=avg_return,
                sharpe_ratio=sharpe,
                predictions_count=len(data),
                rank=0,  # Will be set after sorting
            )
            category_reports.append(report)
        
        # Sort by Sharpe ratio
        category_reports.sort(key=lambda x: x.sharpe_ratio, reverse=True)
        
        # Assign ranks
        for i, report in enumerate(category_reports):
            report.rank = i + 1
        
        # Split into top and worst
        n_top = min(3, len(category_reports))
        n_worst = min(3, len(category_reports))
        
        return {
            "top": category_reports[:n_top],
            "worst": category_reports[-n_worst:] if category_reports else [],
        }
    
    def _analyze_regime_performance(
        self,
        records: List,
    ) -> List[RegimePerformanceReport]:
        """Analyze performance by regime."""
        if not records:
            return []
        
        # Group by regime
        regime_data: Dict[str, List] = {}
        
        for record in records:
            if record.regime not in regime_data:
                regime_data[record.regime] = []
            regime_data[record.regime].append(record)
        
        reports = []
        
        for regime, regime_records in regime_data.items():
            completed = [r for r in regime_records if r.outcome.value != "pending"]
            
            if not completed:
                continue
            
            correct = [r for r in completed if r.outcome.value == "correct"]
            win_rate = len(correct) / len(completed)
            
            returns = [r.actual_return for r in completed if r.actual_return is not None]
            avg_return = np.mean(returns) if returns else 0.0
            
            grades = [r.grade for r in completed]
            best_grade = max(grades, key=lambda g: self._grade_score(g))
            worst_grade = min(grades, key=lambda g: self._grade_score(g))
            
            reports.append(RegimePerformanceReport(
                regime=regime,
                predictions_count=len(regime_records),
                win_rate=win_rate,
                avg_return=avg_return,
                best_grade=best_grade,
                worst_grade=worst_grade,
            ))
        
        return reports
    
    def _analyze_stock_performance(
        self,
        records: List,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Analyze top and worst performing stocks."""
        completed = [r for r in records if r.actual_return is not None]
        
        if not completed:
            return [], []
        
        # Sort by return
        sorted_by_return = sorted(completed, key=lambda r: r.actual_return or 0, reverse=True)
        
        # Top 5
        top_stocks = [
            {
                "symbol": r.symbol,
                "alpha_score": r.alpha_score,
                "grade": r.grade.value,
                "actual_return": r.actual_return,
                "regime": r.regime,
            }
            for r in sorted_by_return[:5]
        ]
        
        # Worst 5
        worst_predictions = [
            {
                "symbol": r.symbol,
                "alpha_score": r.alpha_score,
                "grade": r.grade.value,
                "actual_return": r.actual_return,
                "regime": r.regime,
            }
            for r in sorted_by_return[-5:]
        ]
        
        return top_stocks, worst_predictions
    
    def _generate_insights(
        self,
        records: List,
        category_perf: Dict[str, List],
        regime_perf: List,
        grade_perf: Dict[str, GradePerformance],
    ) -> tuple[List[str], List[str]]:
        """Generate insights and recommendations."""
        insights = []
        recommendations = []
        
        # Category insights
        if category_perf["top"]:
            top_cat = category_perf["top"][0]
            insights.append(
                f"Best performing category: {top_cat.category} "
                f"(Sharpe: {top_cat.sharpe_ratio:.2f}, Win Rate: {top_cat.win_rate:.1%})"
            )
        
        if category_perf["worst"]:
            worst_cat = category_perf["worst"][0]
            insights.append(
                f"Worst performing category: {worst_cat.category} "
                f"(Sharpe: {worst_cat.sharpe_ratio:.2f}, Win Rate: {worst_cat.win_rate:.1%})"
            )
            recommendations.append(
                f"Consider reducing weight for {worst_cat.category} or investigate signal quality"
            )
        
        # Regime insights
        if regime_perf:
            best_regime = max(regime_perf, key=lambda r: r.win_rate)
            insights.append(
                f"Best regime: {best_regime.regime} "
                f"(Win Rate: {best_regime.win_rate:.1%}, Avg Return: {best_regime.avg_return:.2%})"
            )
        
        # Grade insights
        institutional_perf = grade_perf.get(AlphaGrade.INSTITUTIONAL)
        if institutional_perf and institutional_perf.total_predictions > 0:
            insights.append(
                f"Institutional grade win rate: {institutional_perf.win_rate:.1%} "
                f"({institutional_perf.total_predictions} predictions)"
            )
        
        # Overall recommendations
        completed = [r for r in records if r.outcome.value != "pending"]
        if completed:
            overall_win_rate = sum(1 for r in completed if r.outcome.value == "correct") / len(completed)
            
            if overall_win_rate < 0.5:
                recommendations.append("Overall win rate below 50% - review signal quality and filter settings")
            elif overall_win_rate < 0.6:
                recommendations.append("Win rate moderate - consider tightening filters or adjusting weights")
            else:
                recommendations.append("Win rate healthy - continue current approach")
        
        return insights, recommendations
    
    def _grade_score(self, grade: AlphaGrade) -> int:
        """Convert grade to numeric score for comparison."""
        scores = {
            AlphaGrade.INSTITUTIONAL: 5,
            AlphaGrade.EXCELLENT: 4,
            AlphaGrade.GOOD: 3,
            AlphaGrade.AVERAGE: 2,
            AlphaGrade.REJECT: 1,
        }
        return scores.get(grade, 0)
    
    def format_report_markdown(self, report: WeeklyAlphaReport) -> str:
        """Format report as markdown for display."""
        lines = [
            f"# Weekly Alpha Research Report",
            f"",
            f"**Period:** {report.week_start} to {report.week_end}",
            f"**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"## Summary",
            f"",
            f"- **Total Predictions:** {report.total_predictions}",
            f"- **Completed Predictions:** {report.completed_predictions}",
            f"- **Overall Win Rate:** {report.overall_win_rate:.1%}",
            f"- **Overall Average Return:** {report.overall_avg_return:.2%}",
            f"",
            f"## Category Performance",
            f"",
            f"### Top Categories",
            f"",
        ]
        
        for cat in report.top_categories:
            lines.append(
                f"**{cat.rank}. {cat.category.title()}** - "
                f"Sharpe: {cat.sharpe_ratio:.2f}, Win Rate: {cat.win_rate:.1%}, "
                f"Avg Return: {cat.avg_return:.2%}"
            )
        
        lines.append("")
        lines.append("### Worst Categories")
        lines.append("")
        
        for cat in report.worst_categories:
            lines.append(
                f"**{cat.category.title()}** - "
                f"Sharpe: {cat.sharpe_ratio:.2f}, Win Rate: {cat.win_rate:.1%}, "
                f"Avg Return: {cat.avg_return:.2%}"
            )
        
        lines.append("")
        lines.append("## Grade Performance")
        lines.append("")
        
        for grade_name, perf in report.grade_performance.items():
            grade_str = grade_name.value if hasattr(grade_name, 'value') else str(grade_name)
            lines.append(
                f"**{grade_str.upper()}** - "
                f"Win Rate: {perf.win_rate:.1%}, Avg Return: {perf.average_return:.2%}, "
                f"Sharpe: {perf.sharpe_ratio:.2f} ({perf.total_predictions} predictions)"
            )
        
        lines.append("")
        lines.append("## Regime Performance")
        lines.append("")
        
        for regime_perf in report.regime_performance:
            lines.append(
                f"**{regime_perf.regime}** - "
                f"Win Rate: {regime_perf.win_rate:.1%}, Avg Return: {regime_perf.avg_return:.2%}"
            )
        
        lines.append("")
        lines.append("## Top Performing Stocks")
        lines.append("")
        
        for stock in report.top_stocks:
            lines.append(
                f"**{stock['symbol']}** - Alpha: {stock['alpha_score']:.1f}, "
                f"Return: {stock['actual_return']:.2%}, Grade: {stock['grade'].upper()}"
            )
        
        lines.append("")
        lines.append("## Insights")
        lines.append("")
        
        for insight in report.insights:
            lines.append(f"- {insight}")
        
        lines.append("")
        lines.append("## Recommendations")
        lines.append("")
        
        for rec in report.recommendations:
            lines.append(f"- {rec}")
        
        return "\n".join(lines)
