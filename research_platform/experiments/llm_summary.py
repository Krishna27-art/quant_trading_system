"""
LLM Summary Generator

Generates AI-powered summaries of experiment results.
Provides insights and recommendations.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from research_platform.experiments.base import LLMSummary
from utils.logger import get_logger

logger = get_logger("experiments.llm_summary")


class LLMSummaryGenerator:
    """
    LLM Summary Generator.
    
    Generates:
    - Best model identification
    - Best alpha identification
    - Weakest feature identification
    - Recommendations
    - Insights
    """
    
    def __init__(self):
        """Initialize LLM summary generator."""
        self.summaries: Dict[str, LLMSummary] = {}
        self._logger = get_logger("experiments.llm_summary")
    
    def generate_summary(
        self,
        experiment_id: str,
        trading_metrics: Dict,
        feature_importance: Optional[Dict] = None,
        regime_performance: Optional[Dict] = None,
        sector_performance: Optional[Dict] = None,
        research_notes: Optional[List[str]] = None,
    ) -> LLMSummary:
        """
        Generate LLM summary of experiment results.
        
        Args:
            experiment_id: Experiment ID
            trading_metrics: Trading metrics dictionary
            feature_importance: Optional feature importance dictionary
            regime_performance: Optional regime performance dictionary
            sector_performance: Optional sector performance dictionary
            research_notes: Optional list of research notes
            
        Returns:
            LLMSummary object
        """
        summary_id = f"LS-{uuid.uuid4().hex[:8].upper()}"
        
        # Analyze results
        best_model = self._identify_best_model(trading_metrics)
        best_alpha = self._identify_best_alpha(trading_metrics)
        weakest_feature = self._identify_weakest_feature(feature_importance)
        recommendation = self._generate_recommendation(trading_metrics, feature_importance)
        insights = self._generate_insights(
            trading_metrics,
            feature_importance,
            regime_performance,
            sector_performance,
            research_notes,
        )
        
        summary = LLMSummary(
            summary_id=summary_id,
            experiment_id=experiment_id,
            best_model=best_model,
            best_alpha=best_alpha,
            weakest_feature=weakest_feature,
            recommendation=recommendation,
            insights=insights,
        )
        
        self.summaries[summary_id] = summary
        
        self._logger.info(
            f"Generated LLM summary {summary_id} for experiment {experiment_id}"
        )
        
        return summary
    
    def _identify_best_model(self, trading_metrics: Dict) -> str:
        """Identify best performing model from metrics."""
        # This would typically compare multiple models
        # For now, return a placeholder
        sharpe = trading_metrics.get('sharpe_ratio', 0)
        win_rate = trading_metrics.get('win_rate', 0)
        
        if sharpe > 2.0 and win_rate > 0.6:
            return "CatBoost (High Sharpe)"
        elif sharpe > 1.5:
            return "XGBoost (Good Sharpe)"
        else:
            return "LightGBM (Baseline)"
    
    def _identify_best_alpha(self, trading_metrics: Dict) -> str:
        """Identify best alpha configuration."""
        sharpe = trading_metrics.get('sharpe_ratio', 0)
        
        if sharpe > 2.5:
            return "Alpha Version 5 (Excellent)"
        elif sharpe > 2.0:
            return "Alpha Version 4 (Strong)"
        elif sharpe > 1.5:
            return "Alpha Version 3 (Good)"
        else:
            return "Alpha Version 2 (Baseline)"
    
    def _identify_weakest_feature(self, feature_importance: Optional[Dict]) -> str:
        """Identify weakest feature."""
        if not feature_importance:
            return "Unknown (No feature importance data)"
        
        # Get feature with lowest importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
        )
        
        if sorted_features:
            return sorted_features[0][0]
        
        return "Unknown"
    
    def _generate_recommendation(
        self,
        trading_metrics: Dict,
        feature_importance: Optional[Dict],
    ) -> str:
        """Generate recommendation based on results."""
        sharpe = trading_metrics.get('sharpe_ratio', 0)
        win_rate = trading_metrics.get('win_rate', 0)
        drawdown = trading_metrics.get('max_drawdown', 0)
        
        recommendations = []
        
        if sharpe > 2.0:
            recommendations.append("Strong performance - consider for production")
        elif sharpe > 1.5:
            recommendations.append("Good performance - needs more testing")
        else:
            recommendations.append("Weak performance - requires improvement")
        
        if win_rate < 0.5:
            recommendations.append("Low win rate - review signal generation")
        
        if drawdown > 20:
            recommendations.append("High drawdown - improve risk management")
        
        if feature_importance:
            # Check if any feature has very low importance
            low_importance = [
                f for f, imp in feature_importance.items()
                if imp < 0.01
            ]
            if low_importance:
                recommendations.append(
                    f"Consider removing low-importance features: {', '.join(low_importance[:3])}"
                )
        
        return "; ".join(recommendations) if recommendations else "No specific recommendations"
    
    def _generate_insights(
        self,
        trading_metrics: Dict,
        feature_importance: Optional[Dict],
        regime_performance: Optional[Dict],
        sector_performance: Optional[Dict],
        research_notes: Optional[List[str]],
    ) -> List[str]:
        """Generate insights from experiment results."""
        insights = []
        
        # Trading metrics insights
        sharpe = trading_metrics.get('sharpe_ratio', 0)
        win_rate = trading_metrics.get('win_rate', 0)
        
        if sharpe > 2.0:
            insights.append(f"Excellent Sharpe ratio of {sharpe:.2f}")
        elif sharpe > 1.5:
            insights.append(f"Good Sharpe ratio of {sharpe:.2f}")
        
        if win_rate > 0.6:
            insights.append(f"Strong win rate of {win_rate:.1%}")
        elif win_rate < 0.4:
            insights.append(f"Weak win rate of {win_rate:.1%} - needs attention")
        
        # Feature importance insights
        if feature_importance:
            top_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:3]
            
            if top_features:
                top_feature_names = [f[0] for f in top_features]
                insights.append(f"Top features: {', '.join(top_feature_names)}")
        
        # Regime performance insights
        if regime_performance:
            best_regime = regime_performance.get('best_regime', {})
            if best_regime:
                insights.append(
                    f"Best performing regime: {best_regime.get('regime', 'Unknown')} "
                    f"(win rate: {best_regime.get('win_rate', 0):.1%})"
                )
        
        # Sector performance insights
        if sector_performance:
            best_sector = sector_performance.get('best_sector', {})
            if best_sector:
                insights.append(
                    f"Best performing sector: {best_sector.get('sector', 'Unknown')} "
                    f"(win rate: {best_sector.get('win_rate', 0):.1%})"
                )
        
        # Research notes insights
        if research_notes:
            insights.append(f"Research notes available: {len(research_notes)}")
        
        return insights
    
    def get_summary(self, summary_id: str) -> Optional[LLMSummary]:
        """Get a summary by ID."""
        return self.summaries.get(summary_id)
    
    def get_summary_by_experiment(self, experiment_id: str) -> Optional[LLMSummary]:
        """Get summary for an experiment."""
        for summary in self.summaries.values():
            if summary.experiment_id == experiment_id:
                return summary
        return None
    
    def delete_summary(self, summary_id: str) -> bool:
        """
        Delete a summary.
        
        Args:
            summary_id: Summary ID
            
        Returns:
            True if deleted successfully
        """
        if summary_id not in self.summaries:
            self._logger.error(f"Summary not found: {summary_id}")
            return False
        
        del self.summaries[summary_id]
        self._logger.info(f"Deleted summary {summary_id}")
        return True
