import os

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class BaseLogistic:
    """
    Layer 2: Base Logistic Regression Model.
    Prevents lookahead bias via scikit-learn Pipeline and TimeSeriesSplit.
    """

    def __init__(self, n_splits: int = 5, use_pca: bool = False, n_components: int = 5):
        self.n_splits = n_splits
        self.use_pca = use_pca
        self.n_components = n_components
        self.feature_names: list[str] = []
        self.pipeline: Pipeline | None = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_cols: list[str],
        save_path: str | None = None,
    ):
        self.feature_names = feature_cols
        X_clean = X_train[self.feature_names]

        steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]

        if self.use_pca:
            steps.append(("pca", PCA(n_components=self.n_components)))

        steps.append(
            ("classifier", LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42))
        )

        pipeline = Pipeline(steps)

        # TimeSeriesSplit cross-validation
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        for _fold, (train_idx, _val_idx) in enumerate(tscv.split(X_clean)):
            X_fold_train = X_clean.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            pipeline.fit(X_fold_train, y_fold_train)

        # Final fit on all training data
        pipeline.fit(X_clean, y_train)
        self.pipeline = pipeline

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(self.pipeline, save_path)

    def load(self, load_path: str):
        self.pipeline = joblib.load(load_path)

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        if not self.pipeline:
            raise RuntimeError("Model has not been trained or loaded yet.")
        X_clean = X_test[self.feature_names]
        return self.pipeline.predict_proba(X_clean)[:, 1]


class EnsembleModel:
    """
    Ensemble Model for Research OS v2.
    Aggregates predictions from multiple model pipelines:
    - Linear: Logistic Regression
    - Structural Breaks: Random Forest
    - Sequential/Gradient: Gradient Boosting
    """

    def __init__(self, feature_cols: list[str], weights: dict[str, float] | None = None):
        self.feature_cols = feature_cols
        self.weights = weights or {"logistic": 0.4, "random_forest": 0.3, "gradient_boosting": 0.3}

        self.models = {
            "logistic": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42),
                    ),
                ]
            ),
            "random_forest": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
                    ),
                ]
            ),
            "gradient_boosting": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
                    ),
                ]
            ),
        }

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        X_clean = X_train[self.feature_cols]
        for _name, model in self.models.items():
            model.fit(X_clean, y_train)

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        X_clean = X_test[self.feature_cols]
        preds = []
        weight_list = []

        for name, model in self.models.items():
            pred = model.predict_proba(X_clean)[:, 1]
            weight = self.weights.get(name, 0.0)
            preds.append(pred * weight)
            weight_list.append(weight)

        summed_weights = sum(weight_list)
        if summed_weights <= 0:
            summed_weights = 1.0

        return np.sum(preds, axis=0) / summed_weights

    def save(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(dir_path, f"{name}.joblib"))

    def load(self, dir_path: str):
        for name in self.models:
            self.models[name] = joblib.load(os.path.join(dir_path, f"{name}.joblib"))
