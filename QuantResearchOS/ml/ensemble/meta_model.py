import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

logger = logging.getLogger(__name__)


class MetaEnsemble:
    def __init__(self):
        # The meta-model is usually a simple, highly regularized model
        # to prevent overfitting on the base models' predictions
        self.meta_model = LogisticRegression(class_weight="balanced")

    def train(self, base_predictions: pd.DataFrame, y_true: pd.Series):
        """
        Train the meta ensemble on the out-of-fold predictions of the base models.
        base_predictions: DataFrame where each column is a base model's prediction
        """
        self.meta_model.fit(base_predictions, y_true)

        # Calculate training metrics
        preds = self.meta_model.predict(base_predictions)
        probs = self.meta_model.predict_proba(base_predictions)

        acc = accuracy_score(y_true, preds)
        loss = log_loss(y_true, probs)
        logger.info("Meta-Ensemble trained. Accuracy: %.4f, Log Loss: %.4f", acc, loss)

    def predict_proba(self, base_predictions: pd.DataFrame) -> np.ndarray:
        """
        Given new predictions from the base models, output the full probability array.
        Shape: (n_samples, n_classes)
        """
        return self.meta_model.predict_proba(base_predictions)

    # --- Model persistence ---

    def save_model(self, path: str) -> None:
        """Persist the meta-model to *path* using joblib."""
        joblib.dump(self.meta_model, path)

    def load_model(self, path: str) -> None:
        """Load a previously saved meta-model from *path*."""
        self.meta_model = joblib.load(path)
