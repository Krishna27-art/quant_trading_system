"""
Retraining Decision Engine

Decides whether retraining should happen based on multiple criteria.
Evaluates drift, accuracy, calibration, and data availability.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

from continuous_learning.drift_detection.feature_drift import DriftResult
from continuous_learning.drift_detection.prediction_drift import PredictionDriftResult
from continuous_learning.drift_detection.data_drift import DataDriftResult
from continuous_learning.calibration.calibration_monitor import CalibrationMetrics
from utils.logger import get_logger

logger = get_logger("continuous_learning.retraining")


@dataclass
class RetrainingDecision:
    """Decision on whether to retrain."""
    should_retrain: bool
    confidence: float  # 0-1
    reasons: List[str]
    priority: str  # "LOW", "MEDIUM", "HIGH", "URGENT"
    recommended_actions: List[str]
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate retraining decision.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check confidence is between 0 and 1
        if not (0.0 <= self.confidence <= 1.0):
            errors.append(f"Confidence must be between 0 and 1, got {self.confidence}")
        
        # Check priority is valid
        valid_priorities = ["LOW", "MEDIUM", "HIGH", "URGENT"]
        if self.priority not in valid_priorities:
            errors.append(f"Invalid priority: {self.priority}")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "should_retrain": self.should_retrain,
            "confidence": round(self.confidence, 4),
            "reasons": self.reasons,
            "priority": self.priority,
            "recommended_actions": self.recommended_actions,
        }


class RetrainingDecisionEngine:
    """
    Decides whether retraining should happen.
    
    Criteria:
    - Drift severity
    - Accuracy degradation
    - Calibration quality
    - Data availability
    - Time since last retraining
    """
    
    def __init__(
        self,
        drift_threshold: float = 0.5,
        accuracy_threshold: float = 0.5,
        calibration_threshold: float = 0.15,
        min_data_points: int = 1000,
        min_days_since_retrain: int = 7,
    ):
        """
        Initialize retraining decision engine.
        
        Args:
            drift_threshold: Threshold for drift to trigger retraining
            accuracy_threshold: Minimum accuracy threshold
            calibration_threshold: Maximum calibration error threshold
            min_data_points: Minimum data points for retraining
            min_days_since_retrain: Minimum days since last retraining
        """
        self.drift_threshold = drift_threshold
        self.accuracy_threshold = accuracy_threshold
        self.calibration_threshold = calibration_threshold
        self.min_data_points = min_data_points
        self.min_days_since_retrain = min_days_since_retrain
        self._logger = get_logger("continuous_learning.retraining")
    
    def should_retrain(
        self,
        feature_drift: Dict[str, DriftResult],
        prediction_drift: Dict[str, PredictionDriftResult],
        data_drift: Dict[str, DataDriftResult],
        calibration_metrics: CalibrationMetrics,
        current_accuracy: float,
        data_points_available: int,
        last_retrain_date: Optional[datetime] = None,
    ) -> RetrainingDecision:
        """
        Decide whether to retrain.
        
        Args:
            feature_drift: Feature drift results
            prediction_drift: Prediction drift results
            data_drift: Data drift results
            calibration_metrics: Calibration metrics
            current_accuracy: Current model accuracy
            data_points_available: Number of data points available
            last_retrain_date: Date of last retraining
            
        Returns:
            RetrainingDecision
        """
        reasons = []
        recommended_actions = []
        retrain_score = 0.0
        
        # Check feature drift
        feature_drift_score = self._evaluate_feature_drift(feature_drift)
        if feature_drift_score > self.drift_threshold:
            reasons.append(f"High feature drift detected (score: {feature_drift_score:.2f})")
            recommended_actions.append("Investigate feature drift")
            retrain_score += 0.3
        
        # Check prediction drift
        prediction_drift_score = self._evaluate_prediction_drift(prediction_drift)
        if prediction_drift_score > self.drift_threshold:
            reasons.append(f"High prediction drift detected (score: {prediction_drift_score:.2f})")
            recommended_actions.append("Investigate prediction drift")
            retrain_score += 0.3
        
        # Check data drift
        data_drift_score = self._evaluate_data_drift(data_drift)
        if data_drift_score > self.drift_threshold:
            reasons.append(f"High data drift detected (score: {data_drift_score:.2f})")
            recommended_actions.append("Investigate data drift")
            retrain_score += 0.2
        
        # Check calibration
        if not calibration_metrics.is_calibrated:
            reasons.append(f"Poor calibration (ECE: {calibration_metrics.expected_calibration_error:.4f})")
            recommended_actions.append("Recalibrate probabilities")
            retrain_score += 0.2
        
        # Check accuracy
        if current_accuracy < self.accuracy_threshold:
            reasons.append(f"Low accuracy ({current_accuracy:.2%})")
            recommended_actions.append("Review model performance")
            retrain_score += 0.2
        
        # Check data availability
        if data_points_available < self.min_data_points:
            reasons.append(f"Insufficient data for retraining ({data_points_available} < {self.min_data_points})")
            recommended_actions.append("Wait for more data")
            retrain_score -= 0.5  # Penalize if not enough data
        
        # Check time since last retrain
        if last_retrain_date:
            days_since_retrain = (datetime.now() - last_retrain_date).days
            if days_since_retrain < self.min_days_since_retrain:
                reasons.append(f"Too soon since last retraining ({days_since_retrain} days)")
                recommended_actions.append("Wait for minimum retraining interval")
                retrain_score -= 0.3
        
        # Normalize score
        retrain_score = max(0.0, min(1.0, retrain_score))
        
        # Make decision
        should_retrain = retrain_score > 0.5
        
        # Determine priority
        if retrain_score > 0.8:
            priority = "URGENT"
        elif retrain_score > 0.6:
            priority = "HIGH"
        elif retrain_score > 0.5:
            priority = "MEDIUM"
        else:
            priority = "LOW"
        
        # If not retraining, add reason
        if not should_retrain:
            if not reasons:
                reasons.append("Model performing within acceptable parameters")
            recommended_actions.append("Continue monitoring")
        
        return RetrainingDecision(
            should_retrain=should_retrain,
            confidence=retrain_score,
            reasons=reasons,
            priority=priority,
            recommended_actions=recommended_actions,
        )
    
    def _evaluate_feature_drift(self, feature_drift: Dict[str, DriftResult]) -> float:
        """Evaluate overall feature drift."""
        if not feature_drift:
            return 0.0
        
        high_drift_count = sum(1 for r in feature_drift.values() if r.alert_level == "HIGH")
        medium_drift_count = sum(1 for r in feature_drift.values() if r.alert_level == "MEDIUM")
        
        total = len(feature_drift)
        if total == 0:
            return 0.0
        
        # Weighted score
        score = (high_drift_count * 1.0 + medium_drift_count * 0.5) / total
        return score
    
    def _evaluate_prediction_drift(self, prediction_drift: Dict[str, PredictionDriftResult]) -> float:
        """Evaluate overall prediction drift."""
        if not prediction_drift:
            return 0.0
        
        high_drift_count = sum(1 for r in prediction_drift.values() if r.alert_level == "HIGH")
        medium_drift_count = sum(1 for r in prediction_drift.values() if r.alert_level == "MEDIUM")
        
        total = len(prediction_drift)
        if total == 0:
            return 0.0
        
        # Weighted score
        score = (high_drift_count * 1.0 + medium_drift_count * 0.5) / total
        return score
    
    def _evaluate_data_drift(self, data_drift: Dict[str, DataDriftResult]) -> float:
        """Evaluate overall data drift."""
        if not data_drift:
            return 0.0
        
        high_drift_count = sum(1 for r in data_drift.values() if r.alert_level == "HIGH")
        medium_drift_count = sum(1 for r in data_drift.values() if r.alert_level == "MEDIUM")
        
        total = len(data_drift)
        if total == 0:
            return 0.0
        
        # Weighted score
        score = (high_drift_count * 1.0 + medium_drift_count * 0.5) / total
        return score
    
    def generate_report(self, decision: RetrainingDecision) -> str:
        """
        Generate human-readable report.
        
        Args:
            decision: RetrainingDecision
            
        Returns:
            Formatted report string
        """
        lines = []
        
        lines.append("RETRAINING DECISION REPORT")
        lines.append("=" * 50)
        lines.append(f"Should Retrain: {decision.should_retrain}")
        lines.append(f"Confidence: {decision.confidence:.2%}")
        lines.append(f"Priority: {decision.priority}")
        
        lines.append("\nREASONS")
        lines.append("-" * 40)
        for reason in decision.reasons:
            lines.append(f"  - {reason}")
        
        lines.append("\nRECOMMENDED ACTIONS")
        lines.append("-" * 40)
        for action in decision.recommended_actions:
            lines.append(f"  - {action}")
        
        return "\n".join(lines)


def should_retrain(
    feature_drift: Dict[str, DriftResult],
    prediction_drift: Dict[str, PredictionDriftResult],
    data_drift: Dict[str, DataDriftResult],
    calibration_metrics: CalibrationMetrics,
    current_accuracy: float,
    data_points_available: int,
) -> RetrainingDecision:
    """
    Convenience function to decide whether to retrain.
    
    Args:
        feature_drift: Feature drift results
        prediction_drift: Prediction drift results
        data_drift: Data drift results
        calibration_metrics: Calibration metrics
        current_accuracy: Current model accuracy
        data_points_available: Number of data points available
        
    Returns:
        RetrainingDecision
    """
    engine = RetrainingDecisionEngine()
    return engine.should_retrain(
        feature_drift,
        prediction_drift,
        data_drift,
        calibration_metrics,
        current_accuracy,
        data_points_available,
    )
