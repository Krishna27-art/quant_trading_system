"""
Feature Drift

Detects drift in feature distributions between training and current data.
Compares statistical properties and alerts on significant changes.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from utils.logger import get_logger

logger = get_logger("continuous_learning.drift_detection")


@dataclass
class DriftResult:
    """Result of drift detection."""
    feature_name: str
    drift_score: float  # 0-1, higher means more drift
    alert_level: str  # "NONE", "LOW", "MEDIUM", "HIGH"
    p_value: float
    statistic: float
    recommended_action: str
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate drift result.
        
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
            "feature_name": self.feature_name,
            "drift_score": round(self.drift_score, 4),
            "alert_level": self.alert_level,
            "p_value": round(self.p_value, 4),
            "statistic": round(self.statistic, 4),
            "recommended_action": self.recommended_action,
        }


class FeatureDriftDetector:
    """
    Detects drift in feature distributions.
    
    Methods:
    - Kolmogorov-Smirnov test
    - Wasserstein distance
    - Population Stability Index (PSI)
    """
    
    def __init__(
        self,
        low_threshold: float = 0.1,
        medium_threshold: float = 0.3,
        high_threshold: float = 0.5,
    ):
        """
        Initialize feature drift detector.
        
        Args:
            low_threshold: Threshold for LOW alert
            medium_threshold: Threshold for MEDIUM alert
            high_threshold: Threshold for HIGH alert
        """
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        self._logger = get_logger("continuous_learning.drift_detection")
    
    def detect_drift(
        self,
        feature_name: str,
        training_data: np.ndarray,
        current_data: np.ndarray,
    ) -> DriftResult:
        """
        Detect drift for a single feature.
        
        Args:
            feature_name: Name of the feature
            training_data: Training data distribution
            current_data: Current data distribution
            
        Returns:
            DriftResult
        """
        # Remove NaN values
        training_clean = training_data[~np.isnan(training_data)]
        current_clean = current_data[~np.isnan(current_data)]
        
        if len(training_clean) < 10 or len(current_clean) < 10:
            return self._insufficient_data_result(feature_name)
        
        # Calculate KS test
        ks_statistic, ks_p_value = stats.ks_2samp(training_clean, current_clean)
        
        # Calculate Wasserstein distance
        wasserstein_dist = stats.wasserstein_distance(training_clean, current_clean)
        
        # Calculate PSI
        psi = self._calculate_psi(training_clean, current_clean)
        
        # Combine into drift score
        drift_score = self._combine_metrics(ks_p_value, wasserstein_dist, psi)
        
        # Determine alert level
        alert_level = self._determine_alert_level(drift_score)
        
        # Recommend action
        recommended_action = self._recommend_action(alert_level, drift_score)
        
        return DriftResult(
            feature_name=feature_name,
            drift_score=drift_score,
            alert_level=alert_level,
            p_value=ks_p_value,
            statistic=ks_statistic,
            recommended_action=recommended_action,
        )
    
    def _calculate_psi(
        self,
        training_data: np.ndarray,
        current_data: np.ndarray,
        bins: int = 10,
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        Args:
            training_data: Training data
            current_data: Current data
            bins: Number of bins
            
        Returns:
            PSI value
        """
        # Create bins based on training data
        min_val = np.min(training_data)
        max_val = np.max(training_data)
        
        if min_val == max_val:
            return 0.0
        
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        # Calculate distributions
        train_hist, _ = np.histogram(training_data, bins=bin_edges)
        current_hist, _ = np.histogram(current_data, bins=bin_edges)
        
        # Normalize to percentages
        train_pct = train_hist / len(training_data) + 0.0001  # Avoid division by zero
        current_pct = current_hist / len(current_data) + 0.0001
        
        # Calculate PSI
        psi = np.sum((train_pct - current_pct) * np.log(train_pct / current_pct))
        
        return psi
    
    def _combine_metrics(
        self,
        ks_p_value: float,
        wasserstein_dist: float,
        psi: float,
    ) -> float:
        """
        Combine multiple drift metrics into single score.
        
        Args:
            ks_p_value: KS test p-value
            wasserstein_dist: Wasserstein distance
            psi: PSI value
            
        Returns:
            Combined drift score (0-1)
        """
        # Lower p-value = more drift
        ks_score = 1.0 - ks_p_value
        
        # Normalize Wasserstein distance
        # Assuming reasonable range is 0-2
        wasserstein_score = min(wasserstein_dist / 2.0, 1.0)
        
        # Normalize PSI
        # Assuming reasonable range is 0-2
        psi_score = min(psi / 2.0, 1.0)
        
        # Weighted average
        combined = (
            ks_score * 0.4 +
            wasserstein_score * 0.3 +
            psi_score * 0.3
        )
        
        return combined
    
    def _determine_alert_level(self, drift_score: float) -> str:
        """
        Determine alert level from drift score.
        
        Args:
            drift_score: Combined drift score
            
        Returns:
            Alert level
        """
        if drift_score < self.low_threshold:
            return "NONE"
        elif drift_score < self.medium_threshold:
            return "LOW"
        elif drift_score < self.high_threshold:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _recommend_action(self, alert_level: str, drift_score: float) -> str:
        """
        Recommend action based on alert level.
        
        Args:
            alert_level: Alert level
            drift_score: Drift score
            
        Returns:
            Recommended action
        """
        if alert_level == "NONE":
            return "MONITOR"
        elif alert_level == "LOW":
            return "MONITOR"
        elif alert_level == "MEDIUM":
            return "INVESTIGATE"
        else:  # HIGH
            return "RETRAIN"
    
    def _insufficient_data_result(self, feature_name: str) -> DriftResult:
        """Create result for insufficient data."""
        return DriftResult(
            feature_name=feature_name,
            drift_score=0.0,
            alert_level="NONE",
            p_value=1.0,
            statistic=0.0,
            recommended_action="INSUFFICIENT_DATA",
        )
    
    def batch_detect_drift(
        self,
        training_data: Dict[str, np.ndarray],
        current_data: Dict[str, np.ndarray],
    ) -> Dict[str, DriftResult]:
        """
        Detect drift for multiple features.
        
        Args:
            training_data: Dictionary mapping feature names to training data
            current_data: Dictionary mapping feature names to current data
            
        Returns:
            Dictionary mapping feature names to DriftResult
        """
        results = {}
        
        for feature_name in training_data.keys():
            if feature_name in current_data:
                try:
                    result = self.detect_drift(
                        feature_name,
                        training_data[feature_name],
                        current_data[feature_name],
                    )
                    results[feature_name] = result
                except Exception as e:
                    self._logger.error(f"Failed to detect drift for {feature_name}: {e}")
        
        return results
    
    def generate_report(self, drift_results: Dict[str, DriftResult]) -> str:
        """
        Generate human-readable report.
        
        Args:
            drift_results: Dictionary of drift results
            
        Returns:
            Formatted report string
        """
        lines = []
        
        lines.append("FEATURE DRIFT REPORT")
        lines.append("=" * 50)
        
        # Count by alert level
        alert_counts = {"NONE": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0}
        
        for result in drift_results.values():
            alert_counts[result.alert_level] += 1
        
        lines.append(f"Total Features: {len(drift_results)}")
        lines.append(f"High Drift: {alert_counts['HIGH']}")
        lines.append(f"Medium Drift: {alert_counts['MEDIUM']}")
        lines.append(f"Low Drift: {alert_counts['LOW']}")
        lines.append(f"No Drift: {alert_counts['NONE']}")
        
        lines.append("\nDETAILED RESULTS")
        lines.append("-" * 40)
        
        for feature_name, result in drift_results.items():
            lines.append(f"{feature_name}:")
            lines.append(f"  Drift Score: {result.drift_score:.2%}")
            lines.append(f"  Alert Level: {result.alert_level}")
            lines.append(f"  P-Value: {result.p_value:.4f}")
            lines.append(f"  Action: {result.recommended_action}")
        
        return "\n".join(lines)


def detect_feature_drift(
    feature_name: str,
    training_data: np.ndarray,
    current_data: np.ndarray,
) -> DriftResult:
    """
    Convenience function to detect feature drift.
    
    Args:
        feature_name: Name of the feature
        training_data: Training data distribution
        current_data: Current data distribution
        
    Returns:
        DriftResult
    """
    detector = FeatureDriftDetector()
    return detector.detect_drift(feature_name, training_data, current_data)
