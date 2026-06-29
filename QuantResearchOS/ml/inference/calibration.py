import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV


class ProbabilityCalibrator:
    """
    Calibrates raw probabilities from primary ML models using Isotonic Regression or Platt Scaling (Sigmoid).
    Ensures that a predicted confidence of 80% translates to a historical 80% win rate.
    """

    def __init__(self, method: str = "isotonic"):
        # method can be 'sigmoid' (Platt) or 'isotonic'
        self.method = method
        self.calibrator = None

    def fit(self, base_estimator, X_val, y_val):
        """
        Fit the calibrator using a held-out validation set.
        base_estimator should already be pre-fit (cv='prefit').
        """
        self.calibrator = CalibratedClassifierCV(
            estimator=base_estimator, method=self.method, cv="prefit"
        )
        self.calibrator.fit(X_val, y_val)

    def predict_proba(self, X) -> np.ndarray:
        if self.calibrator is None:
            raise RuntimeError("Calibrator is not fitted.")
        # Returns calibrated probabilities
        return self.calibrator.predict_proba(X)

    # --- Persistence ---

    def save(self, path: str) -> None:
        """Save the fitted calibrator to *path* using joblib."""
        if self.calibrator is None:
            raise RuntimeError("No fitted calibrator to save.")
        joblib.dump(self.calibrator, path)

    def load(self, path: str) -> None:
        """Load a previously saved calibrator from *path*."""
        self.calibrator = joblib.load(path)
