"""
Probability Calibration

Calibrates model probabilities using isotonic regression.
Ensures predicted probabilities match actual frequencies.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
import joblib


class ProbabilityCalibrator:
    """
    Calibrates model probabilities using isotonic regression.
    
    Calibration ensures that when the model predicts 70% probability,
    the event actually occurs ~70% of the time.
    """
    
    def __init__(self, method: str = "isotonic"):
        """
        Initialize calibrator.
        
        Args:
            method: Calibration method ('isotonic' or 'sigmoid')
        """
        self.method = method
        self.calibrator = None
        self.is_fitted = False
    
    def fit(self, y_true: np.ndarray, y_proba: np.ndarray) -> None:
        """
        Fit the calibrator on true labels and predicted probabilities.
        
        Args:
            y_true: True labels (0 or 1)
            y_proba: Predicted probabilities
        """
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        else:
            self.calibrator = CalibratedClassifierCV(method='sigmoid', cv='prefit')
        
        self.calibrator.fit(y_proba, y_true)
        self.is_fitted = True
    
    def calibrate(self, y_proba: np.ndarray) -> np.ndarray:
        """
        Calibrate predicted probabilities.
        
        Args:
            y_proba: Predicted probabilities
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before calibration")
        
        return self.calibrator.predict(y_proba)
    
    def save(self, path: str) -> None:
        """Save calibrator to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted calibrator")
        
        joblib.dump(self.calibrator, path)
    
    def load(self, path: str) -> None:
        """Load calibrator from disk."""
        self.calibrator = joblib.load(path)
        self.is_fitted = True
