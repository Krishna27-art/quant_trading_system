"""
Prediction Drift

Detects drift in prediction distributions and characteristics.
Monitors changes in prediction probabilities, confidence levels, and action distributions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from scipy import stats

from utils.logger import get_logger

logger = get_logger("continuous_learning.drift_detection")


@dataclass
class PredictionDriftResult:
    """Result of prediction drift detection."""
    metric_name: str
    drift_score: float  # 0-1, higher means more drift
    alert_level: str  # "NONE", "LOW", "MEDIUM", "HIGH"
    p_value: float
    statistic: float
    recommended_action: str
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate prediction drift result.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check drift score is between 0 and 1
        if not (0.0 <= self.drift_score <= 1.0):
            errors.append(f"Drift score must be between 0 and 1, got {self.drift_score}")
        
        # Check alert level is valid
        valid_levels = ["NONE", "LOW", "MEDIUM", "HIGH"]
        if self.alert_level not in valid_levels:
            errors.append(f"Invalid alert level: {self.alert_level}")
        
        # Check p-value is between 0 and 1
        if not (0.0 <= self.p_value <= 1.0):
            errors.append(f"P-value must be between 0 and 1, got {self.p_value}")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "metric_name": self.metric_name,
            "drift_score": round(self.drift_score, 4),
            "alert_level": self.alert_level,
            "p_value": round(self.p_value, 4),
            "statistic": round(self.statistic, 4),
            "recommended_action": self.recommended_action,
        }


class PredictionDriftDetector:
    """
    Detects drift in prediction distributions.
    
    Monitors:
    - Prediction probabilities
    - Confidence levels
    - Action distributions (BUY/SELL/HOLD)
    """
    
    def __init__(
        self,
        low_threshold: float = 0.1,
        medium_threshold: float = 0.3,
        high_threshold: float = 0.5,
    ):
        """
        Initialize prediction drift detector.
        
        Args:
            low_threshold: Threshold for LOW alert
            medium_threshold: Threshold for MEDIUM alert
            high_threshold: Threshold for HIGH alert
        """
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        self._logger = get_logger("continuous_learning.drift_detection")
    
    def detect_probability_drift(
        self,
        training_probabilities: np.ndarray,
        current_probabilities: np.ndarray,
    ) -> PredictionDriftResult:
        """
        Detect drift in prediction probabilities.
        
        Args:
            training_probabilities: Training prediction probabilities
            current_probabilities: Current prediction probabilities
            
        Returns:
            PredictionDriftResult
        """
        # Remove NaN values
        training_clean = training_probabilities[~np.isnan(training_probabilities)]
        current_clean = current_probabilities[~np.isnan(current_probabilities)]
        
        if len(training_clean) < 10 or len(current_clean) < 10:
            return self._insufficient_data_result("probability")
        
        # Calculate KS test
        ks_statistic, ks_p_value = stats.ks_2samp(training_clean, current_clean)
        
        # Calculate mean shift
        training_mean = np.mean(training_clean)
        current_mean = np.mean(current_clean)
        mean_shift = abs(current_mean - training_mean)
        
        # Calculate drift score
        drift_score = self._calculate_drift_score(ks_p_value, mean_shift)
        
        # Determine alert level
        alert_level = self._determine_alert_level(drift_score)
        
        # Recommend action
        recommended_action = self._recommend_action(alert_level)
        
        return PredictionDriftResult(
            metric_name="probability",
            drift_score=drift_score,
            alert_level=alert_level,
            p_value=ks_p_value,
            statistic=ks_statistic,
            recommended_action=recommended_action,
        )
    
    def detect_confidence_drift(
        self,
        training_confidences: List[str],
        current_confidences: List[str],
    ) -> PredictionDriftResult:
        """
        Detect drift in confidence level distributions.
        
        Args:
            training_confidences: List of training confidence levels
            current_confidences: List of current confidence levels
            
        Returns:
            PredictionDriftResult
        """
        # Convert to numeric
        confidence_map = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        
        training_numeric = np.array([confidence_map.get(c, 2) for c in training_confidences])
        current_numeric = np.array([confidence_map.get(c, 2) for c in current_confidences])
        
        if len(training_numeric) < 10 or len(current_numeric) < 10:
            return self._insufficient_data_result("confidence")
        
        # Calculate KS test
        ks_statistic, ks_p_value = stats.ks_2samp(training_numeric, current_numeric)
        
        # Calculate drift score
        drift_score = 1.0 - ks_p_value
        
        # Determine alert level
        alert_level = self._determine_alert_level(drift_score)
        
        # Recommend action
        recommended_action = self._recommend_action(alert_level)
        
        return PredictionDriftResult(
            metric_name="confidence",
            drift_score=drift_score,
            alert_level=alert_level,
            p_value=ks_p_value,
            statistic=ks_statistic,
            recommended_action=recommended_action,
        )
    
    def detect_action_drift(
        self,
        training_actions: List[str],
        current_actions: List[str],
    ) -> PredictionDriftResult:
        """
        Detect drift in action distributions.
        
        Args:
            training_actions: List of training actions
            current_actions: List of current actions
            
        Returns:
            PredictionDriftResult
        """
        # Calculate distributions
        training_dist = self._calculate_action_distribution(training_actions)
        current_dist = self._calculate_action_distribution(current_actions)
        
        # Calculate chi-squared test
        actions = list(set(training_actions) | set(current_actions))
        
        training_counts = [training_dist.get(a, 0) for a in actions]
        current_counts = [current_dist.get(a, 0) for a in actions]
        
        # Add small constant to avoid division by zero
        training_counts = [c + 1 for c in training_counts]
        current_counts = [c + 1 for c in current_counts]
        
        try:
            chi2_statistic, chi2_p_value = stats.chisquare(current_counts, training_counts)
        except:
            chi2_statistic, chi2_p_value = 0.0, 1.0
        
        # Calculate drift score
        drift_score = 1.0 - chi2_p_value
        
        # Determine alert level
        alert_level = self._determine_alert_level(drift_score)
        
        # Recommend action
        recommended_action = self._recommend_action(alert_level)
        
        return PredictionDriftResult(
            metric_name="action",
            drift_score=drift_score,
            alert_level=alert_level,
            p_value=chi2_p_value,
            statistic=chi2_statistic,
            recommended_action=recommended_action,
        )
    
    def _calculate_action_distribution(self, actions: List[str]) -> Dict[str, float]:
        """Calculate action distribution."""
        total = len(actions)
        if total == 0:
            return {}
        
        dist = {}
        for action in actions:
            dist[action] = dist.get(action, 0) + 1
        
        return {k: v / total for k, v in dist.items()}
    
    def _calculate_drift_score(self, p_value: float, mean_shift: float) -> float:
        """Calculate combined drift score."""
        # Lower p-value = more drift
        p_score = 1.0 - p_value
        
        # Normalize mean shift (assuming max shift is 0.5)
        shift_score = min(mean_shift / 0.5, 1.0)
        
        # Weighted average
        combined = p_score * 0.7 + shift_score * 0.3
        
        return combined
    
    def _determine_alert_level(self, drift_score: float) -> str:
        """Determine alert level from drift score."""
        if drift_score < self.low_threshold:
            return "NONE"
        elif drift_score < self.medium_threshold:
            return "LOW"
        elif drift_score < self.high_threshold:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _recommend_action(self, alert_level: str) -> str:
        """Recommend action based on alert level."""
        if alert_level == "NONE":
            return "MONITOR"
        elif alert_level == "LOW":
            return "MONITOR"
        elif alert_level == "MEDIUM":
            return "INVESTIGATE"
        else:  # HIGH
            return "RECALIBRATE"
    
    def _insufficient_data_result(self, metric_name: str) -> PredictionDriftResult:
        """Create result for insufficient data."""
        return PredictionDriftResult(
            metric_name=metric_name,
            drift_score=0.0,
            alert_level="NONE",
            p_value=1.0,
            statistic=0.0,
            recommended_action="INSUFFICIENT_DATA",
        )
    
    def detect_all_drift(
        self,
        training_probabilities: np.ndarray,
        current_probabilities: np.ndarray,
        training_confidences: Optional[List[str]] = None,
        current_confidences: Optional[List[str]] = None,
        training_actions: Optional[List[str]] = None,
        current_actions: Optional[List[str]] = None,
    ) -> Dict[str, PredictionDriftResult]:
        """
        Detect drift for all prediction metrics.
        
        Args:
            training_probabilities: Training prediction probabilities
            current_probabilities: Current prediction probabilities
            training_confidences: Optional training confidence levels
            current_confidences: Optional current confidence levels
            training_actions: Optional training actions
            current_actions: Optional current actions
            
        Returns:
            Dictionary mapping metric names to PredictionDriftResult
        """
        results = {}
        
        # Probability drift
        results["probability"] = self.detect_probability_drift(
            training_probabilities,
            current_probabilities,
        )
        
        # Confidence drift
        if training_confidences and current_confidences:
            results["confidence"] = self.detect_confidence_drift(
                training_confidences,
                current_confidences,
            )
        
        # Action drift
        if training_actions and current_actions:
            results["action"] = self.detect_action_drift(
                training_actions,
                current_actions,
            )
        
        return results


def detect_prediction_drift(
    training_probabilities: np.ndarray,
    current_probabilities: np.ndarray,
) -> PredictionDriftResult:
    """
    Convenience function to detect prediction drift.
    
    Args:
        training_probabilities: Training prediction probabilities
        current_probabilities: Current prediction probabilities
        
    Returns:
        PredictionDriftResult
    """
    detector = PredictionDriftDetector()
    return detector.detect_probability_drift(training_probabilities, current_probabilities)
