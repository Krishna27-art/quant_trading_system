from typing import Any

import joblib
import lightgbm as lgb
import pandas as pd
import xgboost as xgb


class BaseTreeTrainer:
    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type.lower()
        self.model = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: dict[str, Any],
        task: str = "regression",
    ):
        """
        Train the tree model.

        Parameters
        ----------
        task : str
            'regression' or 'classification'. Explicitly controls model choice
            instead of guessing from dtype.
        """
        if task not in ("regression", "classification"):
            raise ValueError(f"task must be 'regression' or 'classification', got '{task}'")

        is_regression = task == "regression"

        if self.model_type == "xgboost":
            if is_regression:
                self.model = xgb.XGBRegressor(**params)
            else:
                self.model = xgb.XGBClassifier(**params)
            self.model.fit(X_train, y_train)

        elif self.model_type == "lightgbm":
            if is_regression:
                self.model = lgb.LGBMRegressor(**params)
            else:
                self.model = lgb.LGBMClassifier(**params)
            self.model.fit(X_train, y_train)

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def predict(self, X: pd.DataFrame) -> pd.Series | pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model is not trained yet.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series | pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model is not trained yet.")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise AttributeError(f"{self.model_type} does not support predict_proba")

    # --- Model persistence ---

    def save_model(self, path: str) -> None:
        """Persist the trained model to *path* using joblib."""
        if self.model is None:
            raise RuntimeError("No trained model to save.")
        joblib.dump(self.model, path)

    def load_model(self, path: str) -> None:
        """Load a previously saved model from *path*."""
        self.model = joblib.load(path)
