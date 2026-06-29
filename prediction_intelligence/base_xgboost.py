import numpy as np
import pandas as pd
import xgboost as xgb


class BaseXGBoost:
    """
    Layer 2: Base XGBoost Model
    Specifically trained on a subset of features emphasizing Order Flow and Volume (RVOL).
    """

    def __init__(self):
        self.model = None
        self.feature_names = []

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, feature_cols: list[str]):
        # In a real pipeline, we would filter feature_cols to only Volume/Flow features
        self.feature_names = feature_cols

        self.model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.6,
            objective="binary:logistic",
            n_jobs=-1,
            random_state=42,
        )
        self.model.fit(X_train[self.feature_names], y_train)

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict_proba(X_test[self.feature_names])[:, 1]
