"""
Improvement Suggestion Engine

Generates LLM-based improvement suggestions for the prediction system.
Uses Qwen or DeepSeek to analyze performance data and suggest improvements.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

from prediction_layer.prediction_learning.weekly_report import WeeklyReport
from prediction_layer.prediction_learning.feature_learning import FeatureLearningEngine
from prediction_layer.prediction_learning.signal_learning import SignalLearningEngine
from prediction_layer.prediction_learning.regime_learning import RegimeLearningEngine
from prediction_layer.prediction_learning.failure_analysis import FailureAnalysis

from utils.logger import get_logger

logger = get_logger("prediction_layer.prediction_learning.improvement_suggestions")


@dataclass
class ImprovementSuggestion:
    """Single improvement suggestion."""
    category: str
    priority: str
    suggestion: str
    rationale: str
    expected_impact: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "category": self.category,
            "priority": self.priority,
            "suggestion": self.suggestion,
            "rationale": self.rationale,
            "expected_impact": self.expected_impact,
        }


@dataclass
class ImprovementReport:
    """Complete improvement report."""
    generated_at: datetime
    summary: str
    suggestions: List[ImprovementSuggestion]
    feature_recommendations: List[str]
    signal_recommendations: List[str]
    regime_recommendations: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "summary": self.summary,
            "suggestions": [s.to_dict() for s in self.suggestions],
            "feature_recommendations": self.feature_recommendations,
            "signal_recommendations": self.signal_recommendations,
            "regime_recommendations": self.regime_recommendations,
        }


class ImprovementSuggestionEngine:
    """
    Generates improvement suggestions using LLM analysis.
    
    Analyzes:
    - Weekly performance reports
    - Feature learning data
    - Signal learning data
    - Regime learning data
    - Failure analysis
    """
    
    def __init__(self, llm_client: Optional[any] = None):
        """
        Initialize improvement suggestion engine.
        
        Args:
            llm_client: Optional LLM client (Qwen or DeepSeek)
        """
        self.llm_client = llm_client
        self._logger = get_logger("prediction_layer.prediction_learning.improvement_suggestions")
    
    def generate_improvement_report(
        self,
        weekly_report: WeeklyReport,
        feature_learning: FeatureLearningEngine,
        signal_learning: SignalLearningEngine,
        regime_learning: RegimeLearningEngine,
        recent_failures: List[FailureAnalysis],
    ) -> ImprovementReport:
        """
        Generate comprehensive improvement report.
        
        Args:
            weekly_report: Weekly performance report
            feature_learning: Feature learning engine
            signal_learning: Signal learning engine
            regime_learning: Regime learning engine
            recent_failures: List of recent failure analyses
            
        Returns:
            ImprovementReport
        """
        # Generate summary
        summary = self._generate_summary(
            weekly_report,
            feature_learning,
            signal_learning,
            regime_learning,
        )
        
        # Generate suggestions
        suggestions = self._generate_suggestions(
            weekly_report,
            feature_learning,
            signal_learning,
            regime_learning,
            recent_failures,
        )
        
        # Generate feature recommendations
        feature_recommendations = self._generate_feature_recommendations(
            feature_learning,
            weekly_report,
        )
        
        # Generate signal recommendations
        signal_recommendations = self._generate_signal_recommendations(
            signal_learning,
            weekly_report,
        )
        
        # Generate regime recommendations
        regime_recommendations = self._generate_regime_recommendations(
            regime_learning,
            weekly_report,
        )
        
        self._logger.info("Generated improvement report")
        
        return ImprovementReport(
            generated_at=datetime.now(),
            summary=summary,
            suggestions=suggestions,
            feature_recommendations=feature_recommendations,
            signal_recommendations=signal_recommendations,
            regime_recommendations=regime_recommendations,
        )
    
    def _generate_summary(
        self,
        weekly_report: WeeklyReport,
        feature_learning: FeatureLearningEngine,
        signal_learning: SignalLearningEngine,
        regime_learning: RegimeLearningEngine,
    ) -> str:
        """
        Generate summary of current system state.
        
        Args:
            weekly_report: Weekly performance report
            feature_learning: Feature learning engine
            signal_learning: Signal learning engine
            regime_learning: Regime learning engine
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        # Overall performance
        metrics = weekly_report.metrics
        summary_parts.append(
            f"Overall win rate: {metrics.win_rate:.1%}, "
            f"average return: {metrics.average_return:.2f}%, "
            f"Sharpe ratio: {metrics.sharpe_ratio:.2f}"
        )
        
        # Feature learning summary
        feature_summary = feature_learning.get_learning_summary()
        summary_parts.append(
            f"Tracking {feature_summary['total_features']} features, "
            f"{feature_summary['eligible_features']} with sufficient data"
        )
        
        # Signal learning summary
        signal_summary = signal_learning.get_learning_summary()
        summary_parts.append(
            f"Tracking {signal_summary['total_signals']} signals, "
            f"{signal_summary['eligible_signals']} with sufficient data"
        )
        
        # Regime learning summary
        regime_summary = regime_learning.get_learning_summary()
        summary_parts.append(
            f"Tracking {regime_summary['total_regimes']} regimes, "
            f"{regime_summary['eligible_regimes']} with sufficient data"
        )
        
        return ". ".join(summary_parts)
    
    def _generate_suggestions(
        self,
        weekly_report: WeeklyReport,
        feature_learning: FeatureLearningEngine,
        signal_learning: SignalLearningEngine,
        regime_learning: RegimeLearningEngine,
        recent_failures: List[FailureAnalysis],
    ) -> List[ImprovementSuggestion]:
        """
        Generate improvement suggestions.
        
        Args:
            weekly_report: Weekly performance report
            feature_learning: Feature learning engine
            signal_learning: Signal learning engine
            regime_learning: Regime learning engine
            recent_failures: List of recent failure analyses
            
        Returns:
            List of ImprovementSuggestion objects
        """
        suggestions = []
        
        # Analyze win rate
        if weekly_report.metrics.win_rate < 0.5:
            suggestions.append(ImprovementSuggestion(
                category="PERFORMANCE",
                priority="HIGH",
                suggestion="Increase minimum confidence threshold",
                rationale=f"Current win rate of {weekly_report.metrics.win_rate:.1%} is below 50%",
                expected_impact="Improve win rate by filtering low-confidence predictions",
            ))
        
        # Analyze top/worst features
        worst_features = feature_learning.get_worst_features(3)
        for feature in worst_features:
            if feature.win_rate < 0.4:
                suggestions.append(ImprovementSuggestion(
                    category="FEATURE",
                    priority="MEDIUM",
                    suggestion=f"Review or disable {feature.feature_name} feature",
                    rationale=f"Feature has low win rate of {feature.win_rate:.1%}",
                    expected_impact="Improve overall prediction quality",
                ))
        
        # Analyze top/worst signals
        worst_signals = signal_learning.get_worst_signals(3)
        for signal in worst_signals:
            if signal.win_rate < 0.4:
                suggestions.append(ImprovementSuggestion(
                    category="SIGNAL",
                    priority="MEDIUM",
                    suggestion=f"Reduce weight of {signal.signal_name} signal",
                    rationale=f"Signal has low win rate of {signal.win_rate:.1%}",
                    expected_impact="Improve signal aggregation quality",
                ))
        
        # Analyze regime performance
        worst_regime = regime_learning.get_worst_regime()
        if worst_regime:
            regime_perf = regime_learning.get_regime_performance(worst_regime)
            if regime_perf and regime_perf.win_rate < 0.4:
                suggestions.append(ImprovementSuggestion(
                    category="REGIME",
                    priority="HIGH",
                    suggestion=f"Add regime filter to avoid {worst_regime} market conditions",
                    rationale=f"Regime has low win rate of {regime_perf.win_rate:.1%}",
                    expected_impact="Significantly improve win rate by avoiding unfavorable conditions",
                ))
        
        # Analyze common failure reasons
        if recent_failures:
            failure_reasons = {}
            for failure in recent_failures:
                reason = failure.failure_reason.value
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            
            most_common_failure = max(failure_reasons, key=failure_reasons.get)
            if failure_reasons[most_common_failure] > len(recent_failures) * 0.3:
                suggestions.append(ImprovementSuggestion(
                    category="RISK_MANAGEMENT",
                    priority="HIGH",
                    suggestion=f"Address {most_common_failure} failures",
                    rationale=f"{most_common_failure} accounts for {failure_reasons[most_common_failure]} of recent failures",
                    expected_impact="Reduce preventable losses",
                ))
        
        # If LLM client is available, generate additional suggestions
        if self.llm_client:
            llm_suggestions = self._generate_llm_suggestions(
                weekly_report,
                feature_learning,
                signal_learning,
                regime_learning,
                recent_failures,
            )
            suggestions.extend(llm_suggestions)
        
        return suggestions
    
    def _generate_feature_recommendations(
        self,
        feature_learning: FeatureLearningEngine,
        weekly_report: WeeklyReport,
    ) -> List[str]:
        """
        Generate feature-specific recommendations.
        
        Args:
            feature_learning: Feature learning engine
            weekly_report: Weekly performance report
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        top_features = feature_learning.get_top_features(5)
        if top_features:
            recommendations.append(
                f"Top performing features: {', '.join([f.feature_name for f in top_features])}"
            )
        
        worst_features = feature_learning.get_worst_features(5)
        if worst_features:
            recommendations.append(
                f"Worst performing features: {', '.join([f.feature_name for f in worst_features])}"
            )
        
        decayed_features = feature_learning.detect_feature_decay()
        if decayed_features:
            recommendations.append(
                f"Decayed features to review: {', '.join(decayed_features)}"
            )
        
        return recommendations
    
    def _generate_signal_recommendations(
        self,
        signal_learning: SignalLearningEngine,
        weekly_report: WeeklyReport,
    ) -> List[str]:
        """
        Generate signal-specific recommendations.
        
        Args:
            signal_learning: Signal learning engine
            weekly_report: Weekly performance report
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        top_signals = signal_learning.get_top_signals(5)
        if top_signals:
            recommendations.append(
                f"Top performing signals: {', '.join([s.signal_name for s in top_signals])}"
            )
        
        worst_signals = signal_learning.get_worst_signals(5)
        if worst_signals:
            recommendations.append(
                f"Worst performing signals: {', '.join([s.signal_name for s in worst_signals])}"
            )
        
        decayed_signals = signal_learning.detect_signal_decay()
        if decayed_signals:
            recommendations.append(
                f"Decayed signals to review: {', '.join(decayed_signals)}"
            )
        
        return recommendations
    
    def _generate_regime_recommendations(
        self,
        regime_learning: RegimeLearningEngine,
        weekly_report: WeeklyReport,
    ) -> List[str]:
        """
        Generate regime-specific recommendations.
        
        Args:
            regime_learning: Regime learning engine
            weekly_report: Weekly performance report
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        best_regime = regime_learning.get_best_regime()
        if best_regime:
            recommendations.append(f"Best regime: {best_regime}")
        
        worst_regime = regime_learning.get_worst_regime()
        if worst_regime:
            recommendations.append(f"Worst regime: {worst_regime}")
        
        regime_rankings = regime_learning.get_regime_rankings()
        if regime_rankings:
            ranking_str = ", ".join([f"{r} ({wr:.1%})" for r, wr, _ in regime_rankings[:3]])
            recommendations.append(f"Regime rankings: {ranking_str}")
        
        return recommendations
    
    def _generate_llm_suggestions(
        self,
        weekly_report: WeeklyReport,
        feature_learning: FeatureLearningEngine,
        signal_learning: SignalLearningEngine,
        regime_learning: RegimeLearningEngine,
        recent_failures: List[FailureAnalysis],
    ) -> List[ImprovementSuggestion]:
        """
        Generate LLM-based improvement suggestions.
        
        Args:
            weekly_report: Weekly performance report
            feature_learning: Feature learning engine
            signal_learning: Signal learning engine
            regime_learning: Regime learning engine
            recent_failures: List of recent failure analyses
            
        Returns:
            List of ImprovementSuggestion objects
        """
        # This would call the LLM client to generate suggestions
        # For now, return empty list as placeholder
        self._logger.info("LLM-based suggestions not implemented - would use Qwen/DeepSeek")
        return []


def generate_improvement_suggestions(
    weekly_report: WeeklyReport,
    feature_learning: FeatureLearningEngine,
    signal_learning: SignalLearningEngine,
    regime_learning: RegimeLearningEngine,
    recent_failures: List[FailureAnalysis],
    llm_client: Optional[any] = None,
) -> ImprovementReport:
    """
    Convenience function to generate improvement suggestions.
    
    Args:
        weekly_report: Weekly performance report
        feature_learning: Feature learning engine
        signal_learning: Signal learning engine
        regime_learning: Regime learning engine
        recent_failures: List of recent failure analyses
        llm_client: Optional LLM client
        
    Returns:
        ImprovementReport
    """
    engine = ImprovementSuggestionEngine(llm_client)
    return engine.generate_improvement_report(
        weekly_report,
        feature_learning,
        signal_learning,
        regime_learning,
        recent_failures,
    )
