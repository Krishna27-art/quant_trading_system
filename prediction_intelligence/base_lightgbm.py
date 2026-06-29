import lightgbm as lgb
import numpy as np
import pandas as pd


class BaseLightGBM:
    """
    Layer 2: Base LightGBM Model
    Aggressive L1 regularization, optimized for asymmetric intraday binary targets.
    """

    def __init__(self):
        self.model = None
        self.feature_names = []

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, feature_cols: list[str]):
        self.feature_names = feature_cols

        # Binary classification target (1 if hit +1.5% before -0.75%, else 0)
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "feature_fraction": 0.7,
            "lambda_l1": 1.5,  # Aggressive L1 regularization as requested
            "lambda_l2": 0.5,
            "min_child_samples": 40,
            "verbose": -1,
            "n_jobs": -1,
            "random_state": 42,
        }

        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(X_train[self.feature_names], y_train)

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained")
        # Returns probability of class 1 (win)
        return self.model.predict_proba(X_test[self.feature_names])[:, 1]
