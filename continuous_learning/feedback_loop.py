"""
Feedback Loop Orchestrator

Integrates the continuous learning feedback loop into the research pipeline.
Coordinates outcome resolution, attribution, failure analysis, regime statistics, factor evolution, drift detection, calibration, weight recommendations, and retraining decisions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd

from continuous_learning.outcome_engine.outcome_resolver import OutcomeResolver
from continuous_learning.outcome_engine.trade_outcome import TradeOutcome
from continuous_learning.outcome_engine.outcome_validator import OutcomeValidator
from continuous_learning.attribution_engine.factor_attribution import FactorAttributor
from continuous_learning.attribution_engine.contribution_engine import ContributionEngine
from continuous_learning.feedback_engine.failure_analysis import FailureAnalyzer
from continuous_learning.factor_evolution.regime_statistics import RegimeStatistics
from continuous_learning.factor_evolution.factor_evolution import FactorEvolution
from continuous_learning.drift_detection.feature_drift import FeatureDriftDetector
from continuous_learning.drift_detection.prediction_drift import PredictionDriftDetector
from continuous_learning.drift_detection.data_drift import DataDriftDetector
from continuous_learning.calibration.calibration_monitor import CalibrationMonitor
from continuous_learning.learning_engine.weight_recommender import WeightRecommender
from continuous_learning.retraining.retraining_decision import RetrainingDecisionEngine
from continuous_learning.retraining.knowledge_database import KnowledgeDatabase
from continuous_learning.dashboards.research_dashboard import ResearchDashboard
from meta_alpha.evidence_engine.evidence import Evidence
from utils.logger import get_logger

logger = get_logger("continuous_learning.feedback_loop")


@dataclass
class FeedbackLoopResult:
    """Result of feedback loop processing."""
    trade_outcome: Optional[TradeOutcome]
    attribution: Optional
    failure_analysis: Optional
    regime_updated: bool
    factor_evolution_updated: bool
    drift_detected: bool
    calibration_checked: bool
    weight_recommendations: Optional
    retraining_decision: Optional
    knowledge_stored: bool
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "trade_outcome": self.trade_outcome.to_dict() if self.trade_outcome else None,
            "attribution": self.attribution.to_dict() if self.attribution else None,
            "failure_analysis": self.failure_analysis.to_dict() if self.failure_analysis else None,
            "regime_updated": self.regime_updated,
            "factor_evolution_updated": self.factor_evolution_updated,
            "drift_detected": self.drift_detected,
            "calibration_checked": self.calibration_checked,
            "weight_recommendations": self.weight_recommendations.to_dict() if self.weight_recommendations else None,
            "retraining_decision": self.retraining_decision.to_dict() if self.retraining_decision else None,
            "knowledge_stored": self.knowledge_stored,
        }


class FeedbackLoopOrchestrator:
    """
    Orchestrates the continuous learning feedback loop.
    
    Process:
    1. Resolve trade outcome
    2. Attribute to factors
    3. Analyze failures
    4. Update regime statistics
    5. Update factor evolution
    6. Detect drift
    7. Check calibration
    8. Recommend weight changes
    9. Decide on retraining
    10. Store in knowledge database
    """
    
    def __init__(self):
        """Initialize feedback loop orchestrator."""
        self.outcome_resolver = OutcomeResolver()
        self.outcome_validator = OutcomeValidator()
        self.factor_attributor = FactorAttributor()
        self.contribution_engine = ContributionEngine()
        self.failure_analyzer = FailureAnalyzer()
        self.regime_stats = RegimeStatistics()
        self.factor_evolution = FactorEvolution()
        self.feature_drift_detector = FeatureDriftDetector()
        self.prediction_drift_detector = PredictionDriftDetector()
        self.data_drift_detector = DataDriftDetector()
        self.calibration_monitor = CalibrationMonitor()
        self.weight_recommender = WeightRecommender()
        self.retraining_decision_engine = RetrainingDecisionEngine()
        self.knowledge_database = KnowledgeDatabase()
        self.research_dashboard = ResearchDashboard()
        
        self._logger = get_logger("continuous_learning.feedback_loop")
    
    def process_completed_trade(
        self,
        prediction_id: str,
        symbol: str,
        action: str,
        predicted_probability: float,
        predicted_confidence: str,
        expected_return: float,
        entry_price: float,
        entry_timestamp: datetime,
        target_price: Optional[float],
        stop_loss: Optional[float],
        historical_data: pd.DataFrame,
        evidence_list: List[Evidence],
        current_weights: Optional[Dict[str, float]],
        market_regime: str = "unknown",
    ) -> FeedbackLoopResult:
        """
        Process a completed trade through the feedback loop.
        
        Args:
            prediction_id: Prediction ID
            symbol: Stock symbol
            action: Trading action
            predicted_probability: Predicted probability
            predicted_confidence: Predicted confidence level
            expected_return: Expected return
            entry_price: Entry price
            entry_timestamp: Entry timestamp
            target_price: Target price
            stop_loss: Stop loss price
            historical_data: Historical OHLCV data
            evidence_list: List of Evidence
            current_weights: Current factor weights
            market_regime: Current market regime
            
        Returns:
            FeedbackLoopResult
        """
        # Step 1: Resolve outcome
        resolved_outcome = self.outcome_resolver.resolve(
            entry_price=entry_price,
            entry_timestamp=entry_timestamp,
            target_price=target_price,
            stop_loss=stop_loss,
            historical_data=historical_data,
            action=action,
        )
        
        # Step 2: Create trade outcome
        from continuous_learning.outcome_engine.trade_outcome import TradeOutcomeBuilder
        trade_outcome = (
            TradeOutcomeBuilder()
            .prediction_id(prediction_id)
            .symbol(symbol)
            .action(action)
            .predicted_probability(predicted_probability)
            .predicted_confidence(predicted_confidence)
            .expected_return(expected_return)
            .entry_price(entry_price)
            .entry_timestamp(entry_timestamp)
            .target_price(target_price)
            .stop_loss(stop_loss)
            .resolved_outcome(resolved_outcome)
            .build()
        )
        
        # Validate trade outcome
        validation = self.outcome_validator.validate(trade_outcome)
        if not validation.is_valid:
            self._logger.warning(f"Trade outcome validation failed: {validation.errors}")
        
        # Step 3: Attribute to factors
        attribution = self.factor_attributor.attribute(
            trade_outcome=trade_outcome,
            evidence_list=evidence_list,
            weights=current_weights,
        )
        
        # Step 4: Analyze failure if not successful
        failure_analysis = None
        if not trade_outcome.is_successful:
            failure_analysis = self.failure_analyzer.analyze(
                trade_outcome=trade_outcome,
                attribution=attribution,
                market_regime=market_regime,
            )
        
        # Step 5: Update regime statistics
        self.regime_stats.update(
            trade_outcome=trade_outcome,
            attribution=attribution,
            market_regime=market_regime,
        )
        regime_updated = True
        
        # Step 6: Update factor evolution
        self.factor_evolution.update(
            trade_outcome=trade_outcome,
            attribution=attribution,
        )
        factor_evolution_updated = True
        
        # Step 7: Detect drift (if enough data)
        drift_detected = False
        # This would require training data comparison - skipped for now
        
        # Step 8: Check calibration (if enough data)
        calibration_checked = False
        # This would require historical predictions - skipped for now
        
        # Step 9: Recommend weight changes (if enough data)
        weight_recommendations = None
        if current_weights:
            # This would require historical trade outcomes - skipped for now
            pass
        
        # Step 10: Decide on retraining (if enough data)
        retraining_decision = None
        # This would require drift and calibration data - skipped for now
        
        # Step 11: Store in knowledge database
        from continuous_learning.retraining.knowledge_database import KnowledgeRecord
        knowledge_record = KnowledgeRecord(
            prediction_id=prediction_id,
            symbol=symbol,
            prediction_timestamp=entry_timestamp,
            outcome_timestamp=resolved_outcome.exit_timestamp,
            action=action,
            predicted_probability=predicted_probability,
            predicted_confidence=predicted_confidence,
            expected_return=expected_return,
            actual_return=resolved_outcome.return_percentage,
            is_successful=trade_outcome.is_successful,
            outcome_type=resolved_outcome.outcome_type,
            evidence_ids=json.dumps([f"{e.source}_{e.factor_name}" for e in evidence_list]),
            weights=json.dumps(current_weights) if current_weights else "{}",
            market_regime=market_regime,
            calibration_error=None,
            drift_metrics=None,
            factor_performance=json.dumps(attribution.to_dict()) if attribution else "{}",
            lessons_learned=failure_analysis.lessons_learned if failure_analysis else None,
        )
        self.knowledge_database.store_record(knowledge_record)
        knowledge_stored = True
        
        return FeedbackLoopResult(
            trade_outcome=trade_outcome,
            attribution=attribution,
            failure_analysis=failure_analysis,
            regime_updated=regime_updated,
            factor_evolution_updated=factor_evolution_updated,
            drift_detected=drift_detected,
            calibration_checked=calibration_checked,
            weight_recommendations=weight_recommendations,
            retraining_decision=retraining_decision,
            knowledge_stored=knowledge_stored,
        )
    
    def generate_research_dashboard(self) -> dict:
        """
        Generate research dashboard from current state.
        
        Returns:
            Dashboard data dictionary
        """
        # This would require accumulated data from all components
        # For now, return a placeholder
        return {
            "timestamp": datetime.now().isoformat(),
            "top_improving_factors": [],
            "top_declining_factors": [],
            "current_market_regime": "unknown",
            "calibration_quality": "UNKNOWN",
            "recent_win_rate": 0.0,
            "expected_return": 0.0,
            "weight_recommendations": {},
            "retraining_recommendation": {},
            "drift_alerts": [],
        }
    
    def get_learning_summary(self) -> Dict:
        """
        Get summary of learning progress.
        
        Returns:
            Dictionary with learning summary
        """
        # Get knowledge database statistics
        knowledge_stats = self.knowledge_database.get_statistics()
        
        # Get regime statistics
        all_regimes = self.regime_stats.get_all_regimes()
        
        # Get factor evolution
        all_factors = self.factor_evolution.get_all_factors()
        
        # Detect factor decay
        decay_results = self.factor_evolution.detect_all_decay()
        
        return {
            "total_predictions": knowledge_stats["total_predictions"],
            "completed_predictions": knowledge_stats["completed_predictions"],
            "successful_predictions": knowledge_stats["successful_predictions"],
            "average_return": knowledge_stats["average_return"],
            "tracked_regimes": all_regimes,
            "tracked_factors": all_factors,
            "decayed_factors": [f for f, d in decay_results.items() if d.has_decayed],
        }


# Global instance
_global_feedback_loop = None


def get_feedback_loop() -> FeedbackLoopOrchestrator:
    """
    Get global feedback loop orchestrator instance.
    
    Returns:
        FeedbackLoopOrchestrator instance
    """
    global _global_feedback_loop
    if _global_feedback_loop is None:
        _global_feedback_loop = FeedbackLoopOrchestrator()
    return _global_feedback_loop
