"""
Calibration Engine

Calibrates probability estimates to match historical outcomes.
Ensures model probabilities are well-calibrated.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from utils.logger import get_logger

logger = get_logger("signal_engine.calibration")


@dataclass
class CalibrationResult:
    """Result of probability calibration."""
    calibrated_probability: float
    original_probability: float
    calibration_method: str
    calibration_score: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "calibrated_probability": round(self.calibrated_probability, 4),
            "original_probability": round(self.original_probability, 4),
            "calibration_method": self.calibration_method,
            "calibration_score": round(self.calibration_score, 4),
        }


class CalibrationEngine:
    """
    Calibrates probability estimates to match historical outcomes.
    
    Methods:
    - Platt Scaling (Logistic Regression)
    - Isotonic Regression
    - Temperature Scaling
    """
    
    def __init__(
        self,
        calibration_method: str = "isotonic",
    ):
        """
        Initialize calibration engine.
        
        Args:
            calibration_method: Method for calibration (platt, isotonic, temperature)
        """
        self.calibration_method = calibration_method
        self.calibrator = None
        self.is_fitted = False
        self._logger = get_logger("signal_engine.calibration")
    
    def fit(
        self,
        probabilities: List[float],
        outcomes: List[int],
    ) -> None:
        """
        Fit calibration model on historical data.
        
        Args:
            probabilities: List of predicted probabilities
            outcomes: List of actual outcomes (0 or 1)
        """
        if len(probabilities) != len(outcomes):
            raise ValueError("Probabilities and outcomes must have same length")
        
        X = np.array(probabilities).reshape(-1, 1)
        y = np.array(outcomes)
        
        if self.calibration_method == "platt":
            self.calibrator = LogisticRegression(solver='lbfgs')
            self.calibrator.fit(X, y)
        elif self.calibration_method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(X.ravel(), y)
        elif self.calibration_method == "temperature":
            # Temperature scaling (simplified)
            self.calibrator = self._fit_temperature_scaling(probabilities, outcomes)
        else:
            raise ValueError(f"Unknown calibration method: {self.calibration_method}")
        
        self.is_fitted = True
        self._logger.info(f"Fitted calibration model using {self.calibration_method}")
    
    def _fit_temperature_scaling(self, probabilities: List[float], outcomes: List[int]) -> float:
        """Fit temperature scaling parameter."""
        # Simplified temperature scaling
        # In practice, this would use optimization
        return 1.0
    
    def calibrate(self, probability: float) -> CalibrationResult:
        """
        Calibrate a single probability.
        
        Args:
            probability: Probability to calibrate
            
        Returns:
            CalibrationResult
        """
        if not self.is_fitted:
            self._logger.warning("Calibrator not fitted, returning original probability")
            return CalibrationResult(
                calibrated_probability=probability,
                original_probability=probability,
                calibration_method="none",
                calibration_score=0.0,
            )
        
        if self.calibration_method == "platt":
            calibrated = self._calibrate_platt(probability)
        elif self.calibration_method == "isotonic":
            calibrated = self._calibrate_isotonic(probability)
        elif self.calibration_method == "temperature":
            calibrated = self._calibrate_temperature(probability)
        else:
            calibrated = probability
        
        # Calculate calibration score
        calibration_score = self._calculate_calibration_score(probability, calibrated)
        
        return CalibrationResult(
            calibrated_probability=calibrated,
            original_probability=probability,
            calibration_method=self.calibration_method,
            calibration_score=calibration_score,
        )
    
    def _calibrate_platt(self, probability: float) -> float:
        """Calibrate using Platt scaling."""
        X = np.array([[probability]])
        calibrated = self.calibrator.predict_proba(X)[0, 1]
        return calibrated
    
    def _calibrate_isotonic(self, probability: float) -> float:
        """Calibrate using isotonic regression."""
        calibrated = self.calibrator.predict([probability])[0]
        return calibrated
    
    def _calibrate_temperature(self, probability: float) -> float:
        """Calibrate using temperature scaling."""
        # Simplified temperature scaling
        temperature = self.calibrator
        calibrated = 1 / (1 + np.exp(-(np.log(probability / (1 - probability + 1e-10)) / temperature)))
        return calibrated
    
    def _calculate_calibration_score(self, original: float, calibrated: float) -> float:
        """Calculate calibration score."""
        # Score based on how much calibration changed the probability
        change = abs(calibrated - original)
        score = 1.0 - min(change, 1.0)
        return score
    
    def batch_calibrate(self, probabilities: List[float]) -> List[CalibrationResult]:
        """
        Calibrate multiple probabilities.
        
        Args:
            probabilities: List of probabilities to calibrate
            
        Returns:
            List of CalibrationResult
        """
        results = []
        
        for prob in probabilities:
            result = self.calibrate(prob)
            results.append(result)
        
        return results
    
    def evaluate_calibration(
        self,
        probabilities: List[float],
        outcomes: List[int],
        n_bins: int = 10,
    ) -> Dict:
        """
        Evaluate calibration quality using reliability diagram.
        
        Args:
            probabilities: List of predicted probabilities
            outcomes: List of actual outcomes
            n_bins: Number of bins for reliability diagram
            
        Returns:
            Dictionary with calibration metrics
        """
        if not self.is_fitted:
            self._logger.warning("Calibrator not fitted, cannot evaluate")
            return {}
        
        # Bin predictions
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probabilities, bins) - 1
        
        # Calculate observed vs expected for each bin
        bin_observed = []
        bin_expected = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_observed.append(np.mean(np.array(outcomes)[mask]))
                bin_expected.append(np.mean(np.array(probabilities)[mask]))
            else:
                bin_observed.append(0.0)
                bin_expected.append(0.0)
        
        # Calculate Expected Calibration Error (ECE)
        ece = 0.0
        total_samples = len(probabilities)
        
        for i in range(n_bins):
            mask = bin_indices == i
            bin_size = mask.sum()
            if bin_size > 0:
                ece += (bin_size / total_samples) * abs(bin_observed[i] - bin_expected[i])
        
        return {
            "ece": ece,
            "bin_observed": bin_observed,
            "bin_expected": bin_expected,
        }


def calibrate_probability(
    probability: float,
    historical_probabilities: Optional[List[float]] = None,
    historical_outcomes: Optional[List[int]] = None,
    calibration_method: str = "isotonic",
) -> CalibrationResult:
    """
    Convenience function to calibrate a probability.
    
    Args:
        probability: Probability to calibrate
        historical_probabilities: Optional historical probabilities for fitting
        historical_outcomes: Optional historical outcomes for fitting
        calibration_method: Calibration method to use
        
    Returns:
        CalibrationResult
    """
    engine = CalibrationEngine(calibration_method=calibration_method)
    
    if historical_probabilities and historical_outcomes:
        engine.fit(historical_probabilities, historical_outcomes)
    
    return engine.calibrate(probability)
