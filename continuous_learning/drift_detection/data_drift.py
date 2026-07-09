"""
Data Drift

Detects drift in overall data characteristics and market behavior.
Monitors changes in volatility, volume, and market structure.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from utils.logger import get_logger

logger = get_logger("continuous_learning.drift_detection")


@dataclass
class DataDriftResult:
    """Result of data drift detection."""
    metric_name: str
    drift_score: float  # 0-1, higher means more drift
    alert_level: str  # "NONE", "LOW", "MEDIUM", "HIGH"
    p_value: float
    statistic: float
    recommended_action: str
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate data drift result.
        
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


class DataDriftDetector:
    """
    Detects drift in overall data characteristics.
    
    Monitors:
    - Volatility
    - Volume
    - Returns distribution
    - Market structure
    """
    
    def __init__(
        self,
        low_threshold: float = 0.1,
        medium_threshold: float = 0.3,
        high_threshold: float = 0.5,
    ):
        """
        Initialize data drift detector.
        
        Args:
            low_threshold: Threshold for LOW alert
            medium_threshold: Threshold for MEDIUM alert
            high_threshold: Threshold for HIGH alert
        """
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        self._logger = get_logger("continuous_learning.drift_detection")
    
    def detect_volatility_drift(
        self,
        training_volatility: np.ndarray,
        current_volatility: np.ndarray,
    ) -> DataDriftResult:
        """
        Detect drift in volatility.
        
        Args:
            training_volatility: Training volatility values
            current_volatility: Current volatility values
            
        Returns:
            DataDriftResult
        """
        # Remove NaN values
        training_clean = training_volatility[~np.isnan(training_volatility)]
        current_clean = current_volatility[~np.isnan(current_volatility)]
        
        if len(training_clean) < 10 or len(current_clean) < 10:
            return self._insufficient_data_result("volatility")
        
        # Calculate KS test
        ks_statistic, ks_p_value = stats.ks_2samp(training_clean, current_clean)
        
        # Calculate mean shift
        training_mean = np.mean(training_clean)
        current_mean = np.mean(current_clean)
        mean_shift = abs(current_mean - training_mean) / (training_mean + 0.01)
        
        # Calculate drift score
        drift_score = self._calculate_drift_score(ks_p_value, mean_shift)
        
        # Determine alert level
        alert_level = self._determine_alert_level(drift_score)
        
        # Recommend action
        recommended_action = self._recommend_action(alert_level)
        
        return DataDriftResult(
            metric_name="volatility",
            drift_score=drift_score,
            alert_level=alert_level,
            p_value=ks_p_value,
            statistic=ks_statistic,
            recommended_action=recommended_action,
        )
    
    def detect_volume_drift(
        self,
        training_volume: np.ndarray,
        current_volume: np.ndarray,
    ) -> DataDriftResult:
        """
        Detect drift in volume.
        
        Args:
            training_volume: Training volume values
            current_volume: Current volume values
            
        Returns:
            DataDriftResult
        """
        # Remove NaN values
        training_clean = training_volume[~np.isnan(training_volume)]
        current_clean = current_volume[~np.isnan(current_volume)]
        
        if len(training_clean) < 10 or len(current_clean) < 10:
            return self._insufficient_data_result("volume")
        
        # Calculate KS test
        ks_statistic, ks_p_value = stats.ks_2samp(training_clean, current_clean)
        
        # Calculate drift score
        drift_score = 1.0 - ks_p_value
        
        # Determine alert level
        alert_level = self._determine_alert_level(drift_score)
        
        # Recommend action
        recommended_action = self._recommend_action(alert_level)
        
        return DataDriftResult(
            metric_name="volume",
            drift_score=drift_score,
            alert_level=alert_level,
            p_value=ks_p_value,
            statistic=ks_statistic,
            recommended_action=recommended_action,
        )
    
    def detect_return_drift(
        self,
        training_returns: np.ndarray,
        current_returns: np.ndarray,
    ) -> DataDriftResult:
        """
        Detect drift in return distribution.
        
        Args:
            training_returns: Training return values
            current_returns: Current return values
            
        Returns:
            DataDriftResult
        """
        # Remove NaN values
        training_clean = training_returns[~np.isnan(training_returns)]
        current_clean = current_returns[~np.isnan(current_returns)]
        
        if len(training_clean) < 10 or len(current_clean) < 10:
            return self._insufficient_data_result("returns")
        
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
        
        return DataDriftResult(
            metric_name="returns",
            drift_score=drift_score,
            alert_level=alert_level,
            p_value=ks_p_value,
            statistic=ks_statistic,
            recommended_action=recommended_action,
        )
    
    def detect_market_structure_drift(
        self,
        training_data: pd.DataFrame,
        current_data: pd.DataFrame,
    ) -> DataDriftResult:
        """
        Detect drift in market structure (correlations between features).
        
        Args:
            training_data: Training data DataFrame
            current_data: Current data DataFrame
            
        Returns:
            DataDriftResult
        """
        # Calculate correlation matrices
        numeric_cols = training_data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return self._insufficient_data_result("market_structure")
        
        training_corr = training_data[numeric_cols].corr()
        current_corr = current_data[numeric_cols].corr()
        
        # Calculate correlation difference
        corr_diff = np.abs(training_corr - current_corr).mean()
        
        # Calculate drift score
        drift_score = min(corr_diff, 1.0)
        
        # Determine alert level
        alert_level = self._determine_alert_level(drift_score)
        
        # Recommend action
        recommended_action = self._recommend_action(alert_level)
        
        return DataDriftResult(
            metric_name="market_structure",
            drift_score=drift_score,
            alert_level=alert_level,
            p_value=1.0 - drift_score,
            statistic=corr_diff,
            recommended_action=recommended_action,
        )
    
    def _calculate_drift_score(self, p_value: float, mean_shift: float) -> float:
        """Calculate combined drift score."""
        # Lower p-value = more drift
        p_score = 1.0 - p_value
        
        # Normalize mean shift
        shift_score = min(mean_shift, 1.0)
        
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
    
    def _insufficient_data_result(self, metric_name: str) -> DataDriftResult:
        """Create result for insufficient data."""
        return DataDriftResult(
            metric_name=metric_name,
            drift_score=0.0,
            alert_level="NONE",
            p_value=1.0,
            statistic=0.0,
            recommended_action="INSUFFICIENT_DATA",
        )
    
    def detect_all_drift(
        self,
        training_data: pd.DataFrame,
        current_data: pd.DataFrame,
    ) -> Dict[str, DataDriftResult]:
        """
        Detect drift for all data metrics.
        
        Args:
            training_data: Training data DataFrame
            current_data: Current data DataFrame
            
        Returns:
            Dictionary mapping metric names to DataDriftResult
        """
        results = {}
        
        # Volatility drift
        if "volatility" in training_data.columns and "volatility" in current_data.columns:
            results["volatility"] = self.detect_volatility_drift(
                training_data["volatility"].values,
                current_data["volatility"].values,
            )
        
        # Volume drift
        if "volume" in training_data.columns and "volume" in current_data.columns:
            results["volume"] = self.detect_volume_drift(
                training_data["volume"].values,
                current_data["volume"].values,
            )
        
        # Return drift
        if "returns" in training_data.columns and "returns" in current_data.columns:
            results["returns"] = self.detect_return_drift(
                training_data["returns"].values,
                current_data["returns"].values,
            )
        
        # Market structure drift
        results["market_structure"] = self.detect_market_structure_drift(
            training_data,
            current_data,
        )
        
        return results


def detect_data_drift(
    training_data: pd.DataFrame,
    current_data: pd.DataFrame,
) -> Dict[str, DataDriftResult]:
    """
    Convenience function to detect data drift.
    
    Args:
        training_data: Training data DataFrame
        current_data: Current data DataFrame
        
    Returns:
        Dictionary mapping metric names to DataDriftResult
    """
    detector = DataDriftDetector()
    return detector.detect_all_drift(training_data, current_data)
