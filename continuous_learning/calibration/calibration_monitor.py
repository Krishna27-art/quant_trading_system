"""
Calibration Monitor

Monitors calibration quality of probability predictions.
Detects when predicted probabilities don't match actual outcomes.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from scipy import stats

from utils.logger import get_logger

logger = get_logger("continuous_learning.calibration")


@dataclass
class CalibrationResult:
    """Result of calibration monitoring."""
    prediction_id: str
    predicted_probability: float
    actual_outcome: float
    calibration_error: float
    is_well_calibrated: bool
    bin: str  # "LOW", "MEDIUM", "HIGH"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "prediction_id": self.prediction_id,
            "predicted_probability": round(self.predicted_probability, 4),
            "actual_outcome": round(self.actual_outcome, 4),
            "calibration_error": round(self.calibration_error, 4),
            "is_well_calibrated": self.is_well_calibrated,
            "bin": self.bin,
        }


@dataclass
class CalibrationMetrics:
    """Overall calibration metrics."""
    total_predictions: int
    calibration_error: float
    expected_calibration_error: float
    brier_score: float
    reliability_score: float
    is_calibrated: bool
    recommended_action: str
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate calibration metrics.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check calibration error is between 0 and 1
        if not (0.0 <= self.calibration_error <= 1.0):
            errors.append(f"Calibration error must be between 0 and 1, got {self.calibration_error}")
        
        # Check ECE is between 0 and 1
        if not (0.0 <= self.expected_calibration_error <= 1.0):
            errors.append(f"ECE must be between 0 and 1, got {self.expected_calibration_error}")
        
        # Check Brier score is between 0 and 1
        if not (0.0 <= self.brier_score <= 1.0):
            errors.append(f"Brier score must be between 0 and 1, got {self.brier_score}")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "total_predictions": self.total_predictions,
            "calibration_error": round(self.calibration_error, 4),
            "expected_calibration_error": round(self.expected_calibration_error, 4),
            "brier_score": round(self.brier_score, 4),
            "reliability_score": round(self.reliability_score, 4),
            "is_calibrated": self.is_calibrated,
            "recommended_action": self.recommended_action,
        }


class CalibrationMonitor:
    """
    Monitors calibration quality of probability predictions.
    
    Detects:
    - Calibration error
    - Expected Calibration Error (ECE)
    - Brier score
    - Reliability diagram
    """
    
    def __init__(
        self,
        calibration_threshold: float = 0.1,
        ece_threshold: float = 0.15,
        min_samples: int = 30,
    ):
        """
        Initialize calibration monitor.
        
        Args:
            calibration_threshold: Threshold for calibration error
            ece_threshold: Threshold for ECE
            min_samples: Minimum samples for calibration check
        """
        self.calibration_threshold = calibration_threshold
        self.ece_threshold = ece_threshold
        self.min_samples = min_samples
        self._logger = get_logger("continuous_learning.calibration")
    
    def calculate_calibration(
        self,
        predicted_probabilities: np.ndarray,
        actual_outcomes: np.ndarray,
    ) -> CalibrationMetrics:
        """
        Calculate overall calibration metrics.
        
        Args:
            predicted_probabilities: Predicted probabilities
            actual_outcomes: Actual outcomes (0 or 1)
            
        Returns:
            CalibrationMetrics
        """
        if len(predicted_probabilities) < self.min_samples:
            return self._insufficient_data_metrics()
        
        # Calculate calibration error (mean absolute difference)
        calibration_error = np.mean(np.abs(predicted_probabilities - actual_outcomes))
        
        # Calculate Expected Calibration Error (ECE)
        ece = self._calculate_ece(predicted_probabilities, actual_outcomes)
        
        # Calculate Brier score
        brier_score = np.mean((predicted_probabilities - actual_outcomes) ** 2)
        
        # Calculate reliability score
        reliability_score = 1.0 - ece
        
        # Determine if calibrated
        is_calibrated = (
            calibration_error < self.calibration_threshold and
            ece < self.ece_threshold
        )
        
        # Recommend action
        recommended_action = self._recommend_action(is_calibrated, calibration_error, ece)
        
        return CalibrationMetrics(
            total_predictions=len(predicted_probabilities),
            calibration_error=calibration_error,
            expected_calibration_error=ece,
            brier_score=brier_score,
            reliability_score=reliability_score,
            is_calibrated=is_calibrated,
            recommended_action=recommended_action,
        )
    
    def _calculate_ece(
        self,
        predicted_probabilities: np.ndarray,
        actual_outcomes: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Calculate Expected Calibration Error.
        
        Args:
            predicted_probabilities: Predicted probabilities
            actual_outcomes: Actual outcomes
            n_bins: Number of bins
            
        Returns:
            ECE value
        """
        # Create bins
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        ece = 0.0
        total_samples = len(predicted_probabilities)
        
        for i in range(n_bins):
            # Get samples in this bin
            mask = (predicted_probabilities >= bin_edges[i]) & (predicted_probabilities < bin_edges[i + 1])
            
            if np.sum(mask) == 0:
                continue
            
            bin_probs = predicted_probabilities[mask]
            bin_outcomes = actual_outcomes[mask]
            
            # Calculate average predicted and actual in this bin
            avg_predicted = np.mean(bin_probs)
            avg_actual = np.mean(bin_outcomes)
            
            # Weight by bin size
            bin_weight = len(bin_probs) / total_samples
            
            # Add to ECE
            ece += bin_weight * abs(avg_predicted - avg_actual)
        
        return ece
    
    def _recommend_action(self, is_calibrated: bool, calibration_error: float, ece: float) -> str:
        """Recommend action based on calibration."""
        if is_calibrated:
            return "MONITOR"
        elif calibration_error > 0.3 or ece > 0.4:
            return "RECALIBRATE"
        else:
            return "INVESTIGATE"
    
    def _insufficient_data_metrics(self) -> CalibrationMetrics:
        """Create metrics for insufficient data."""
        return CalibrationMetrics(
            total_predictions=0,
            calibration_error=0.0,
            expected_calibration_error=0.0,
            brier_score=0.0,
            reliability_score=0.0,
            is_calibrated=True,
            recommended_action="INSUFFICIENT_DATA",
        )
    
    def check_single_prediction(
        self,
        predicted_probability: float,
        actual_outcome: float,
    ) -> CalibrationResult:
        """
        Check calibration for a single prediction.
        
        Args:
            predicted_probability: Predicted probability
            actual_outcome: Actual outcome
            
        Returns:
            CalibrationResult
        """
        # Calculate calibration error
        calibration_error = abs(predicted_probability - actual_outcome)
        
        # Determine bin
        if predicted_probability < 0.33:
            bin_label = "LOW"
        elif predicted_probability < 0.67:
            bin_label = "MEDIUM"
        else:
            bin_label = "HIGH"
        
        # Check if well calibrated
        is_well_calibrated = calibration_error < 0.2
        
        return CalibrationResult(
            prediction_id="single",
            predicted_probability=predicted_probability,
            actual_outcome=actual_outcome,
            calibration_error=calibration_error,
            is_well_calibrated=is_well_calibrated,
            bin=bin_label,
        )
    
    def generate_reliability_diagram(
        self,
        predicted_probabilities: np.ndarray,
        actual_outcomes: np.ndarray,
        n_bins: int = 10,
    ) -> Dict:
        """
        Generate reliability diagram data.
        
        Args:
            predicted_probabilities: Predicted probabilities
            actual_outcomes: Actual outcomes
            n_bins: Number of bins
            
        Returns:
            Dictionary with reliability diagram data
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        bin_data = []
        
        for i in range(n_bins):
            mask = (predicted_probabilities >= bin_edges[i]) & (predicted_probabilities < bin_edges[i + 1])
            
            if np.sum(mask) == 0:
                continue
            
            bin_probs = predicted_probabilities[mask]
            bin_outcomes = actual_outcomes[mask]
            
            avg_predicted = np.mean(bin_probs)
            avg_actual = np.mean(bin_outcomes)
            count = len(bin_probs)
            
            bin_data.append({
                "bin_center": (bin_edges[i] + bin_edges[i + 1]) / 2,
                "predicted": avg_predicted,
                "actual": avg_actual,
                "count": count,
            })
        
        return {
            "bins": bin_data,
            "perfect_calibration": [(b["bin_center"], b["bin_center"]) for b in bin_data],
        }
    
    def generate_report(self, metrics: CalibrationMetrics) -> str:
        """
        Generate human-readable report.
        
        Args:
            metrics: CalibrationMetrics
            
        Returns:
            Formatted report string
        """
        lines = []
        
        lines.append("CALIBRATION MONITOR REPORT")
        lines.append("=" * 50)
        lines.append(f"Total Predictions: {metrics.total_predictions}")
        lines.append(f"Calibration Error: {metrics.calibration_error:.4f}")
        lines.append(f"Expected Calibration Error: {metrics.expected_calibration_error:.4f}")
        lines.append(f"Brier Score: {metrics.brier_score:.4f}")
        lines.append(f"Reliability Score: {metrics.reliability_score:.4f}")
        lines.append(f"Is Calibrated: {metrics.is_calibrated}")
        lines.append(f"Recommended Action: {metrics.recommended_action}")
        
        return "\n".join(lines)


def monitor_calibration(
    predicted_probabilities: np.ndarray,
    actual_outcomes: np.ndarray,
) -> CalibrationMetrics:
    """
    Convenience function to monitor calibration.
    
    Args:
        predicted_probabilities: Predicted probabilities
        actual_outcomes: Actual outcomes
        
    Returns:
        CalibrationMetrics
    """
    monitor = CalibrationMonitor()
    return monitor.calculate_calibration(predicted_probabilities, actual_outcomes)
