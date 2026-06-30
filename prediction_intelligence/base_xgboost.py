"""
Base XGBoost Classifier

Second model in the ensemble alongside BaseLightGBM. Same feature contract,
same save/load contract, so it can be swapped in as {timeframe}_xgb.pkl
if generate_live_predictions.py is extended to ensemble multiple models.

Not currently loaded by generate_live_predictions.py (that script loads
{timeframe}_lgbm.pkl). To use this model in production, either:
  (a) rename its output to match a model_file entry in TIMEFRAME_CONFIG, or
  (b) extend generate_live_predictions.py to average BaseLightGBM and
      BaseXGBoost probabilities (recommended — ensembling reduces variance).
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from utils.logger import get_logger

from .base_lightgbm import FEATURE_COLS, _log_loss, _roc_auc

logger = get_logger("base_xgboost")

_PARAMS: dict[str, dict] = {
    "INTRADAY": {
        "n_estimators": 400,
        "learning_rate": 0.03,
        "max_depth": 4,
        "subsample": 0.7,
        "colsample_bytree": 0.6,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_jobs": -1,
        "random_state": 42,
    },
    "SWING": {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_jobs": -1,
        "random_state": 42,
    },
    "LONGTERM": {
        "n_estimators": 400,
        "learning_rate": 0.02,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.3,
        "reg_lambda": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_jobs": -1,
        "random_state": 42,
    },
}

_DEFAULT_MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/saved"))


class BaseXGBoost:
    """
    XGBoost binary classifier for one trading timeframe.
    Same external contract as BaseLightGBM (train / predict_proba / save / load).
    """

    def __init__(
        self,
        timeframe: str = "SWING",
        model_dir: Path | str | None = None,
        early_stopping_rounds: int = 30,
    ):
        if timeframe not in _PARAMS:
            raise ValueError(f"timeframe must be one of {list(_PARAMS)}; got '{timeframe}'")
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")

        self.timeframe = timeframe.upper()
        self.model_dir = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
        self.early_stopping_rounds = early_stopping_rounds
        self.feature_names: list[str] = FEATURE_COLS
        self.model: Optional["xgb.XGBClassifier"] = None
        self._train_metrics: dict = {}

    @property
    def save_path(self) -> Path:
        return self.model_dir / f"{self.timeframe.lower()}_xgb.pkl"

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict:
        self._validate_inputs(X_train, y_train, "train")
        self._validate_inputs(X_val, y_val, "val")

        X_tr = X_train[self.feature_names].values.astype(np.float32)
        X_vl = X_val[self.feature_names].values.astype(np.float32)
        y_tr = y_train.values.astype(np.int32)
        y_vl = y_val.values.astype(np.int32)

        pos_rate = y_tr.mean()
        scale_pos_weight = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

        params = dict(_PARAMS[self.timeframe])
        params["scale_pos_weight"] = round(float(scale_pos_weight), 3)
        params["early_stopping_rounds"] = self.early_stopping_rounds

        self.model = xgb.XGBClassifier(**params)
        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_vl, y_vl)],
            verbose=False,
        )

        val_proba = self.model.predict_proba(X_vl)[:, 1]
        val_logloss = _log_loss(y_vl, val_proba)
        val_auc = _roc_auc(y_vl, val_proba)

        importance = dict(zip(self.feature_names, self.model.feature_importances_.tolist()))
        top = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]

        self._train_metrics = {
            "timeframe": self.timeframe,
            "n_train": int(len(y_tr)),
            "n_val": int(len(y_vl)),
            "pos_rate_train": round(float(pos_rate), 4),
            "val_logloss": round(float(val_logloss), 5),
            "val_auc": round(float(val_auc), 4),
            "best_iteration": int(getattr(self.model, "best_iteration", params["n_estimators"])),
            "top_features": top,
        }

        logger.info(
            f"[{self.timeframe}] XGB done. "
            f"val_logloss={val_logloss:.5f} val_auc={val_auc:.4f}"
        )
        return self._train_metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"Missing columns in X: {missing}")
        arr = X[self.feature_names].values.astype(np.float32)
        return self.model.predict_proba(arr)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(np.int32)

    def save(self, path: Path | str | None = None) -> Path:
        self._require_trained()
        target = Path(path) if path else self.save_path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "feature_names": self.feature_names,
            "timeframe": self.timeframe,
            "train_metrics": self._train_metrics,
        }
        with open(target, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"[{self.timeframe}] XGB model saved → {target}")
        return target

    def load(self, path: Path | str | None = None) -> "BaseXGBoost":
        target = Path(path) if path else self.save_path
        if not target.exists():
            raise FileNotFoundError(f"No model artifact at {target}.")
        with open(target, "rb") as f:
            payload = pickle.load(f)
        self.model = payload["model"]
        self.feature_names = payload["feature_names"]
        self.timeframe = payload["timeframe"]
        self._train_metrics = payload.get("train_metrics", {})
        logger.info(f"[{self.timeframe}] XGB model loaded ← {target}")
        return self

    def feature_importance(self) -> pd.Series:
        self._require_trained()
        return pd.Series(self.model.feature_importances_, index=self.feature_names).sort_values(ascending=False)

    def metrics(self) -> dict:
        return dict(self._train_metrics)

    def _require_trained(self) -> None:
        if self.model is None:
            raise RuntimeError(f"[{self.timeframe}] Model is not trained. Call train() or load() first.")

    def _validate_inputs(self, X: pd.DataFrame, y: pd.Series, split_name: str) -> None:
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"{split_name}: missing feature columns: {missing}")
        if y.isna().any():
            raise ValueError(f"{split_name}: y contains NaN.")
        if set(y.unique()) - {0, 1}:
            raise ValueError(f"{split_name}: y must be binary (0/1).")
        if len(X) != len(y):
            raise ValueError(f"{split_name}: X/y length mismatch.")
        if len(X) < 50:
            raise ValueError(f"{split_name}: only {len(X)} samples — need at least 50.")