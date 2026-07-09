"""
Research Dashboard

Provides a comprehensive research cockpit for continuous learning.
Displays top improving factors, declining factors, regime, calibration, win rate, expected return, weight recommendations, and retraining recommendations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import json

from continuous_learning.factor_evolution.factor_evolution import FactorEvolution
from continuous_learning.factor_evolution.regime_statistics import RegimeStatistics
from continuous_learning.calibration.calibration_monitor import CalibrationMetrics
from continuous_learning.learning_engine.weight_recommender import WeightRecommendationReport
from continuous_learning.retraining.retraining_decision import RetrainingDecision
from continuous_learning.drift_detection.feature_drift import DriftResult
from utils.logger import get_logger

logger = get_logger("continuous_learning.dashboards")


@dataclass
class DashboardData:
    """Complete dashboard data."""
    timestamp: datetime
    top_improving_factors: List[tuple]
    top_declining_factors: List[tuple]
    current_market_regime: str
    calibration_quality: str
    recent_win_rate: float
    expected_return: float
    weight_recommendations: Dict
    retraining_recommendation: Dict
    drift_alerts: List[Dict]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "top_improving_factors": [(f, round(s, 4)) for f, s in self.top_improving_factors],
            "top_declining_factors": [(f, round(s, 4)) for f, s in self.top_declining_factors],
            "current_market_regime": self.current_market_regime,
            "calibration_quality": self.calibration_quality,
            "recent_win_rate": round(self.recent_win_rate, 4),
            "expected_return": round(self.expected_return, 4),
            "weight_recommendations": self.weight_recommendations,
            "retraining_recommendation": self.retraining_recommendation,
            "drift_alerts": self.drift_alerts,
        }


class ResearchDashboard:
    """
    Provides a comprehensive research cockpit.
    
    Displays:
    - Top improving factors
    - Top declining factors
    - Current market regime
    - Calibration quality
    - Recent win rate
    - Expected return
    - Weight recommendations
    - Retraining recommendation
    - Drift alerts
    """
    
    def __init__(self):
        """Initialize research dashboard."""
        self._logger = get_logger("continuous_learning.dashboards")
    
    def generate_dashboard(
        self,
        factor_evolution: FactorEvolution,
        regime_stats: RegimeStatistics,
        calibration_metrics: CalibrationMetrics,
        weight_recommendations: WeightRecommendationReport,
        retraining_decision: RetrainingDecision,
        feature_drift: Dict[str, DriftResult],
        current_regime: str = "unknown",
    ) -> DashboardData:
        """
        Generate complete dashboard data.
        
        Args:
            factor_evolution: FactorEvolution instance
            regime_stats: RegimeStatistics instance
            calibration_metrics: CalibrationMetrics
            weight_recommendations: WeightRecommendationReport
            retraining_decision: RetrainingDecision
            feature_drift: Feature drift results
            current_regime: Current market regime
            
        Returns:
            DashboardData
        """
        # Get top improving factors
        top_improving = self._get_top_improving_factors(factor_evolution)
        
        # Get top declining factors
        top_declining = self._get_top_declining_factors(factor_evolution)
        
        # Get calibration quality
        calibration_quality = self._get_calibration_quality(calibration_metrics)
        
        # Get recent win rate
        recent_win_rate = self._get_recent_win_rate(regime_stats, current_regime)
        
        # Get expected return
        expected_return = self._get_expected_return(regime_stats, current_regime)
        
        # Get weight recommendations summary
        weight_rec_summary = self._get_weight_recommendations_summary(weight_recommendations)
        
        # Get retraining recommendation summary
        retraining_rec_summary = self._get_retraining_recommendation_summary(retraining_decision)
        
        # Get drift alerts
        drift_alerts = self._get_drift_alerts(feature_drift)
        
        return DashboardData(
            timestamp=datetime.now(),
            top_improving_factors=top_improving,
            top_declining_factors=top_declining,
            current_market_regime=current_regime,
            calibration_quality=calibration_quality,
            recent_win_rate=recent_win_rate,
            expected_return=expected_return,
            weight_recommendations=weight_rec_summary,
            retraining_recommendation=retraining_rec_summary,
            drift_alerts=drift_alerts,
        )
    
    def _get_top_improving_factors(self, factor_evolution: FactorEvolution) -> List[tuple]:
        """Get top improving factors."""
        all_factors = factor_evolution.get_all_factors()
        improving = []
        
        for factor_name in all_factors:
            trend = factor_evolution.get_performance_trend(factor_name)
            if trend == "IMPROVING":
                history = factor_evolution.get_factor_history(factor_name)
                if history:
                    current_ic = history[-1].information_coefficient
                    improving.append((factor_name, current_ic))
        
        # Sort by IC
        improving.sort(key=lambda x: x[1], reverse=True)
        return improving[:5]
    
    def _get_top_declining_factors(self, factor_evolution: FactorEvolution) -> List[tuple]:
        """Get top declining factors."""
        all_factors = factor_evolution.get_all_factors()
        declining = []
        
        for factor_name in all_factors:
            trend = factor_evolution.get_performance_trend(factor_name)
            if trend == "DECLINING":
                history = factor_evolution.get_factor_history(factor_name)
                if history:
                    current_ic = history[-1].information_coefficient
                    declining.append((factor_name, current_ic))
        
        # Sort by IC (ascending)
        declining.sort(key=lambda x: x[1])
        return declining[:5]
    
    def _get_calibration_quality(self, calibration_metrics: CalibrationMetrics) -> str:
        """Get calibration quality."""
        if calibration_metrics.is_calibrated:
            return "GOOD"
        elif calibration_metrics.calibration_error < 0.2:
            return "FAIR"
        else:
            return "POOR"
    
    def _get_recent_win_rate(self, regime_stats: RegimeStatistics, regime: str) -> float:
        """Get recent win rate."""
        perf = regime_stats.get_regime_performance(regime)
        if perf:
            return perf.win_rate
        return 0.0
    
    def _get_expected_return(self, regime_stats: RegimeStatistics, regime: str) -> float:
        """Get expected return."""
        perf = regime_stats.get_regime_performance(regime)
        if perf:
            return perf.avg_return
        return 0.0
    
    def _get_weight_recommendations_summary(self, weight_recommendations: WeightRecommendationReport) -> Dict:
        """Get weight recommendations summary."""
        return {
            "total_factors": weight_recommendations.total_factors,
            "increase": weight_recommendations.summary["increase"],
            "decrease": weight_recommendations.summary["decrease"],
            "keep": weight_recommendations.summary["keep"],
            "requires_review": weight_recommendations.requires_review,
        }
    
    def _get_retraining_recommendation_summary(self, retraining_decision: RetrainingDecision) -> Dict:
        """Get retraining recommendation summary."""
        return {
            "should_retrain": retraining_decision.should_retrain,
            "confidence": round(retraining_decision.confidence, 4),
            "priority": retraining_decision.priority,
            "reasons": retraining_decision.reasons,
        }
    
    def _get_drift_alerts(self, feature_drift: Dict[str, DriftResult]) -> List[Dict]:
        """Get drift alerts."""
        alerts = []
        
        for feature_name, drift_result in feature_drift.items():
            if drift_result.alert_level in ["MEDIUM", "HIGH"]:
                alerts.append({
                    "feature": feature_name,
                    "alert_level": drift_result.alert_level,
                    "drift_score": round(drift_result.drift_score, 4),
                    "recommended_action": drift_result.recommended_action,
                })
        
        return alerts
    
    def generate_html_report(self, dashboard_data: DashboardData) -> str:
        """
        Generate HTML report for dashboard.
        
        Args:
            dashboard_data: DashboardData
            
        Returns:
            HTML string
        """
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Research Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #1e1e1e; color: #d4d4d4; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { text-align: center; margin-bottom: 30px; }
                .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
                .card { background-color: #2d2d2d; padding: 20px; border-radius: 8px; }
                .card h3 { margin-top: 0; color: #4ec9b0; }
                .metric { font-size: 24px; font-weight: bold; margin: 10px 0; }
                .metric-label { color: #9cdcfe; }
                .list-item { margin: 5px 0; padding: 5px; background-color: #3d3d3d; border-radius: 4px; }
                .alert-high { color: #f14c4c; }
                .alert-medium { color: #cca700; }
                .alert-none { color: #4ec9b0; }
                .timestamp { text-align: right; color: #808080; font-size: 12px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Research Dashboard</h1>
                    <p class="timestamp">Last updated: {timestamp}</p>
                </div>
                
                <div class="grid">
                    <div class="card">
                        <h3>Market Regime</h3>
                        <div class="metric">{regime}</div>
                    </div>
                    
                    <div class="card">
                        <h3>Calibration Quality</h3>
                        <div class="metric {calibration_class}">{calibration}</div>
                    </div>
                    
                    <div class="card">
                        <h3>Recent Win Rate</h3>
                        <div class="metric">{win_rate:.2%}</div>
                    </div>
                    
                    <div class="card">
                        <h3>Expected Return</h3>
                        <div class="metric">{expected_return:.2%}</div>
                    </div>
                    
                    <div class="card">
                        <h3>Top Improving Factors</h3>
                        {improving_factors}
                    </div>
                    
                    <div class="card">
                        <h3>Top Declining Factors</h3>
                        {declining_factors}
                    </div>
                    
                    <div class="card">
                        <h3>Weight Recommendations</h3>
                        <p>Increase: {weight_increase} | Decrease: {weight_decrease} | Keep: {weight_keep}</p>
                        <p class="metric-label">Requires Review: {requires_review}</p>
                    </div>
                    
                    <div class="card">
                        <h3>Retraining Recommendation</h3>
                        <div class="metric {retrain_class}">{retrain_decision}</div>
                        <p class="metric-label">Priority: {priority}</p>
                        <p>Confidence: {confidence:.2%}</p>
                    </div>
                    
                    <div class="card" style="grid-column: span 2;">
                        <h3>Drift Alerts</h3>
                        {drift_alerts}
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Format improving factors
        improving_html = ""
        for factor, ic in dashboard_data.top_improving_factors:
            improving_html += f'<div class="list-item">{factor}: IC {ic:.3f}</div>'
        
        # Format declining factors
        declining_html = ""
        for factor, ic in dashboard_data.top_declining_factors:
            declining_html += f'<div class="list-item">{factor}: IC {ic:.3f}</div>'
        
        # Format drift alerts
        drift_html = ""
        for alert in dashboard_data.drift_alerts:
            alert_class = f"alert-{alert['alert_level'].lower()}"
            drift_html += f'<div class="list-item {alert_class}">{alert["feature"]}: {alert["alert_level"]} (Score: {alert["drift_score"]:.2f}) - {alert["recommended_action"]}</div>'
        
        # Format calibration class
        calibration_class = {
            "GOOD": "alert-none",
            "FAIR": "alert-medium",
            "POOR": "alert-high",
        }.get(dashboard_data.calibration_quality, "alert-none")
        
        # Format retrain class
        retrain_class = "alert-high" if dashboard_data.retraining_recommendation["should_retrain"] else "alert-none"
        retrain_decision = "YES" if dashboard_data.retraining_recommendation["should_retrain"] else "NO"
        
        return html.format(
            timestamp=dashboard_data.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            regime=dashboard_data.current_market_regime,
            calibration=dashboard_data.calibration_quality,
            calibration_class=calibration_class,
            win_rate=dashboard_data.recent_win_rate,
            expected_return=dashboard_data.expected_return,
            improving_factors=improving_html,
            declining_factors=declining_html,
            weight_increase=dashboard_data.weight_recommendations["increase"],
            weight_decrease=dashboard_data.weight_recommendations["decrease"],
            weight_keep=dashboard_data.weight_recommendations["keep"],
            requires_review="Yes" if dashboard_data.weight_recommendations["requires_review"] else "No",
            retrain_decision=retrain_decision,
            retrain_class=retrain_class,
            priority=dashboard_data.retraining_recommendation["priority"],
            confidence=dashboard_data.retraining_recommendation["confidence"],
            drift_alerts=drift_html if drift_html else "<p>No drift alerts</p>",
        )
    
    def generate_text_report(self, dashboard_data: DashboardData) -> str:
        """
        Generate text report for dashboard.
        
        Args:
            dashboard_data: DashboardData
            
        Returns:
            Text string
        """
        lines = []
        
        lines.append("RESEARCH DASHBOARD")
        lines.append("=" * 60)
        lines.append(f"Last Updated: {dashboard_data.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        lines.append("MARKET STATUS")
        lines.append("-" * 40)
        lines.append(f"Current Regime: {dashboard_data.current_market_regime}")
        lines.append(f"Calibration Quality: {dashboard_data.calibration_quality}")
        lines.append(f"Recent Win Rate: {dashboard_data.recent_win_rate:.2%}")
        lines.append(f"Expected Return: {dashboard_data.expected_return:.2%}")
        lines.append("")
        
        lines.append("TOP IMPROVING FACTORS")
        lines.append("-" * 40)
        for factor, ic in dashboard_data.top_improving_factors:
            lines.append(f"  {factor}: IC {ic:.3f}")
        lines.append("")
        
        lines.append("TOP DECLINING FACTORS")
        lines.append("-" * 40)
        for factor, ic in dashboard_data.top_declining_factors:
            lines.append(f"  {factor}: IC {ic:.3f}")
        lines.append("")
        
        lines.append("WEIGHT RECOMMENDATIONS")
        lines.append("-" * 40)
        lines.append(f"  Increase: {dashboard_data.weight_recommendations['increase']}")
        lines.append(f"  Decrease: {dashboard_data.weight_recommendations['decrease']}")
        lines.append(f"  Keep: {dashboard_data.weight_recommendations['keep']}")
        lines.append(f"  Requires Review: {dashboard_data.weight_recommendations['requires_review']}")
        lines.append("")
        
        lines.append("RETRAINING RECOMMENDATION")
        lines.append("-" * 40)
        lines.append(f"  Should Retrain: {dashboard_data.retraining_recommendation['should_retrain']}")
        lines.append(f"  Priority: {dashboard_data.retraining_recommendation['priority']}")
        lines.append(f"  Confidence: {dashboard_data.retraining_recommendation['confidence']:.2%}")
        lines.append("")
        
        lines.append("DRIFT ALERTS")
        lines.append("-" * 40)
        if dashboard_data.drift_alerts:
            for alert in dashboard_data.drift_alerts:
                lines.append(f"  {alert['feature']}: {alert['alert_level']} (Score: {alert['drift_score']:.2f})")
                lines.append(f"    Action: {alert['recommended_action']}")
        else:
            lines.append("  No drift alerts")
        
        return "\n".join(lines)


def generate_research_dashboard(
    factor_evolution: FactorEvolution,
    regime_stats: RegimeStatistics,
    calibration_metrics: CalibrationMetrics,
    weight_recommendations: WeightRecommendationReport,
    retraining_decision: RetrainingDecision,
    feature_drift: Dict[str, DriftResult],
) -> DashboardData:
    """
    Convenience function to generate research dashboard.
    
    Args:
        factor_evolution: FactorEvolution instance
        regime_stats: RegimeStatistics instance
        calibration_metrics: CalibrationMetrics
        weight_recommendations: WeightRecommendationReport
        retraining_decision: RetrainingDecision
        feature_drift: Feature drift results
        
    Returns:
        DashboardData
    """
    dashboard = ResearchDashboard()
    return dashboard.generate_dashboard(
        factor_evolution,
        regime_stats,
        calibration_metrics,
        weight_recommendations,
        retraining_decision,
        feature_drift,
    )
