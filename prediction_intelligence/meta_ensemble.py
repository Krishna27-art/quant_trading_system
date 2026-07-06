"""
prediction_intelligence/meta_ensemble.py

Layer 2 Stacking Ensemble Model:
Combines BaseLightGBM, BaseXGBoost, and BaseLogistic into a meta-learner (stacking).
If the training sample size is small (small-n fallback), it automatically falls back
to soft-voting (probability averaging) to prevent overfitting or CV split errors.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

from prediction_intelligence.base_lightgbm import BaseLightGBM
from prediction_intelligence.base_xgboost import BaseXGBoost
from prediction_intelligence.base_logistic import BaseLogistic
from utils.logger import get_logger

logger = get_logger("meta_ensemble")


class MetaEnsemble:
    """
    MetaEnsemble combining LightGBM, XGBoost, and Logistic Regression.
    
    If n_samples >= 100, trains a LogisticRegression meta-learner using TimeSeriesSplit OOF.
    If n_samples < 100, falls back to a simple average of base model probabilities.
    """

    def __init__(
        self,
        timeframe: str = "SWING",
        model_dir: Path | str | None = None,
        feature_cols: list[str] | None = None,
    ):
        self.timeframe = timeframe.upper()
        self.model_dir = Path(model_dir) if model_dir else Path(os.getenv("MODEL_DIR", "models/saved"))
        
        # Determine features automatically if not provided
        from prediction_intelligence.base_logistic import INTRADAY_FEATURES, SWING_FEATURES, LONGTERM_FEATURES
        if feature_cols:
            self.feature_cols = feature_cols
        elif self.timeframe == "INTRADAY":
            self.feature_cols = INTRADAY_FEATURES
        elif self.timeframe == "SWING":
            self.feature_cols = SWING_FEATURES
        else:
            self.feature_cols = LONGTERM_FEATURES

        self.lgbm = BaseLightGBM(timeframe=self.timeframe, model_dir=self.model_dir)
        self.lgbm.feature_names = self.feature_cols
        self.xgb = BaseXGBoost(timeframe=self.timeframe, model_dir=self.model_dir)
        self.xgb.feature_names = self.feature_cols
        self.logistic = BaseLogistic()

        self.meta_learner = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
        self._is_fitted = False
        self.use_stacking_fallback = False
        self.train_metrics = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Fit all three base models and the stacking meta-learner.
        
        If len(X) < 100, activates soft-voting fallback.
        """
        n_samples = len(X)
        logger.info(f"[{self.timeframe}] Training MetaEnsemble with {n_samples} samples.")

        # Decide on stacking vs simple soft-voting fallback
        if n_samples < 250:
            self.use_stacking_fallback = True
            logger.info(f"[{self.timeframe}] Sample size {n_samples} < 250. Using soft-voting average fallback.")

            import lightgbm as lgb
            import xgboost as xgb
            from prediction_intelligence.base_lightgbm import _PARAMS as lgb_params_dict
            from prediction_intelligence.base_xgboost import _PARAMS as xgb_params_dict

            pos_rate = y.mean()
            scale_pos_weight = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

            lgb_params = dict(lgb_params_dict[self.timeframe])
            lgb_params["scale_pos_weight"] = round(scale_pos_weight, 3)
            self.lgbm.model = lgb.LGBMClassifier(**lgb_params)
            self.lgbm.model.fit(X[self.feature_cols].values.astype(np.float32), y.values.astype(np.int32))

            xgb_params = dict(xgb_params_dict[self.timeframe])
            xgb_params["scale_pos_weight"] = round(float(scale_pos_weight), 3)
            self.xgb.model = xgb.XGBClassifier(**xgb_params)
            self.xgb.model.fit(X[self.feature_cols].values.astype(np.float32), y.values.astype(np.int32))

            logger.info(f"[{self.timeframe}] Training BaseLogistic...")
            logistic_metrics = self.logistic.train(X, y, self.feature_cols)

            lgbm_metrics = {"val_auc": 0.0}
            xgb_metrics = {"val_auc": 0.0}
        else:
            self.use_stacking_fallback = False
            logger.info(f"[{self.timeframe}] Sample size {n_samples} >= 250. Training Logistic Regression meta-learner...")

            # Train base models
            # LGBM and XGB require validation sets for early stopping.
            # We split the training set 80/20 for this.
            split_idx = int(n_samples * 0.8)
            X_tr, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_tr, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

            logger.info(f"[{self.timeframe}] Training BaseLightGBM and BaseXGBoost...")
            lgbm_metrics = self.lgbm.train(X_tr, y_tr, X_val, y_val)
            xgb_metrics = self.xgb.train(X_tr, y_tr, X_val, y_val)

            logger.info(f"[{self.timeframe}] Training BaseLogistic...")
            logistic_metrics = self.logistic.train(X, y, self.feature_cols)

            # Determine purging horizon (V) based on timeframe
            if self.timeframe == "INTRADAY":
                v_barrier = 10
            elif self.timeframe == "SWING":
                v_barrier = 15
            else:
                v_barrier = 12

            # Generate out-of-fold predictions for stacking
            tscv = TimeSeriesSplit(n_splits=5)
            oof_preds = np.zeros((n_samples, 3))  # 3 columns for LGBM, XGB, LR probabilities
            
            dates = pd.to_datetime(X["__date__"]) if "__date__" in X.columns else None
            cols_to_keep = self.feature_cols + (["__date__"] if "__date__" in X.columns else [])
            X_clean = X[cols_to_keep].copy()

            for train_idx, val_idx in tscv.split(X_clean):
                if dates is not None and len(val_idx) > 0:
                    val_start_date = dates.iloc[val_idx[0]]
                    if self.timeframe == "INTRADAY":
                        delta = pd.Timedelta(minutes=v_barrier)
                    elif self.timeframe == "SWING":
                        delta = pd.Timedelta(days=v_barrier)
                    else:
                        delta = pd.Timedelta(weeks=v_barrier)
                    cutoff_date = val_start_date - delta
                    train_idx_purged = [i for i in train_idx if dates.iloc[i] < cutoff_date]
                else:
                    if len(train_idx) > v_barrier:
                        train_idx_purged = train_idx[:-v_barrier]
                    else:
                        train_idx_purged = train_idx

                X_train_fold, X_val_fold = X_clean.iloc[train_idx_purged], X_clean.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx_purged], y.iloc[val_idx]

                X_train_fold_clean = X_train_fold[self.feature_cols]
                X_val_fold_clean = X_val_fold[self.feature_cols]

                # Base models fitted on fold (using simplified fold fit)
                fold_lgbm = BaseLightGBM(timeframe=self.timeframe, model_dir=self.model_dir)
                fold_lgbm.feature_names = self.feature_cols
                fold_xgb = BaseXGBoost(timeframe=self.timeframe, model_dir=self.model_dir)
                fold_xgb.feature_names = self.feature_cols
                fold_logistic = BaseLogistic()

                # Direct fit on fold to avoid validation size errors
                import lightgbm as lgb
                import xgboost as xgb
                from prediction_intelligence.base_lightgbm import _PARAMS as lgb_p_d
                from prediction_intelligence.base_xgboost import _PARAMS as xgb_p_d

                pos_r = y_train_fold.mean()
                spw = (1 - pos_r) / pos_r if pos_r > 0 else 1.0

                lparams = dict(lgb_p_d[self.timeframe])
                lparams["scale_pos_weight"] = round(spw, 3)
                fold_lgbm.model = lgb.LGBMClassifier(**lparams)
                fold_lgbm.model.fit(X_train_fold_clean.values.astype(np.float32), y_train_fold.values.astype(np.int32))

                xparams = dict(xgb_p_d[self.timeframe])
                xparams["scale_pos_weight"] = round(float(spw), 3)
                fold_xgb.model = xgb.XGBClassifier(**xparams)
                fold_xgb.model.fit(X_train_fold_clean.values.astype(np.float32), y_train_fold.values.astype(np.int32))

                fold_logistic.train(X_train_fold, y_train_fold, self.feature_cols)

                # Predict on val fold
                if fold_lgbm.model is not None:
                    oof_preds[val_idx, 0] = fold_lgbm.predict_proba(X_val_fold_clean)[:, 1]
                else:
                    oof_preds[val_idx, 0] = 0.5
                
                if fold_xgb.model is not None:
                    oof_preds[val_idx, 1] = fold_xgb.predict_proba(X_val_fold_clean)[:, 1]
                else:
                    oof_preds[val_idx, 1] = 0.5
                
                if fold_logistic.is_ready():
                    oof_preds[val_idx, 2] = fold_logistic.predict_proba(X_val_fold_clean)
                else:
                    oof_preds[val_idx, 2] = 0.5

            # Train meta-learner on OOF predictions of the val_indices
            val_indices = []
            for _, val_idx in tscv.split(X):
                val_indices.extend(val_idx)
            val_indices = np.array(val_indices)

            X_meta = oof_preds[val_indices]
            y_meta = y.iloc[val_indices]

            self.meta_learner.fit(X_meta, y_meta)
            logger.info(f"[{self.timeframe}] Meta-learner coefficients: {self.meta_learner.coef_}")

        self._is_fitted = True
        self.train_metrics = {
            "timeframe": self.timeframe,
            "n_samples": n_samples,
            "use_stacking_fallback": self.use_stacking_fallback,
            "lgbm_val_auc": lgbm_metrics.get("val_auc", 0.0),
            "xgb_val_auc": xgb_metrics.get("val_auc", 0.0),
            "logistic_train_acc": logistic_metrics.get("final_train_acc", 0.0),
        }
        return self.train_metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of win.
        Returns shape (n_samples, 2) matching sklearn predict_proba convention.
        """
        if not self._is_fitted:
            raise RuntimeError("MetaEnsemble must be fitted before predict_proba.")

        # Ensure all feature columns are present and filled
        X_aligned = X[self.feature_cols].copy().fillna(0.0)

        # Get probabilities from all base models
        p_lgbm = self.lgbm.predict_proba(X_aligned)[:, 1]
        p_xgb = self.xgb.predict_proba(X_aligned)[:, 1]
        p_logistic = self.logistic.predict_proba(X_aligned)

        if self.use_stacking_fallback:
            # Soft voting average
            p_win = (p_lgbm + p_xgb + p_logistic) / 3.0
        else:
            # Feed into meta-learner
            X_meta = np.column_stack([p_lgbm, p_xgb, p_logistic])
            p_win = self.meta_learner.predict_proba(X_meta)[:, 1]

        return np.column_stack([1 - p_win, p_win])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(np.int32)

    def is_ready(self) -> bool:
        return self._is_fitted

    def save(self, path: Path | str | None = None) -> Path:
        """
        Save MetaEnsemble directory structure.
        """
        target_dir = Path(path) if path else Path(self.model_dir) / f"meta_ensemble_{self.timeframe.lower()}"
        target_dir.mkdir(parents=True, exist_ok=True)

        # Save sub-models
        self.lgbm.save(target_dir / "lgbm.pkl")
        self.xgb.save(target_dir / "xgb.pkl")
        self.logistic._save(str(target_dir / "logistic.joblib"))

        # Save meta learner
        if not self.use_stacking_fallback:
            joblib.dump(self.meta_learner, target_dir / "meta_learner.joblib")

        # Save metadata JSON
        meta = {
            "timeframe": self.timeframe,
            "feature_cols": self.feature_cols,
            "use_stacking_fallback": self.use_stacking_fallback,
            "train_metrics": self.train_metrics,
            "is_fitted": self._is_fitted
        }
        with open(target_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"[{self.timeframe}] MetaEnsemble saved → {target_dir}")
        return target_dir

    def load(self, path: Path | str | None = None) -> MetaEnsemble:
        """
        Load MetaEnsemble directory structure.
        """
        target_dir = Path(path) if path else Path(self.model_dir) / f"meta_ensemble_{self.timeframe.lower()}"
        if not target_dir.exists():
            raise FileNotFoundError(f"MetaEnsemble path not found: {target_dir}")

        with open(target_dir / "meta.json") as f:
            meta = json.load(f)

        self.timeframe = meta["timeframe"]
        self.feature_cols = meta["feature_cols"]
        self.use_stacking_fallback = meta["use_stacking_fallback"]
        self.train_metrics = meta.get("train_metrics", {})
        self._is_fitted = meta.get("is_fitted", True)

        self.lgbm.load(target_dir / "lgbm.pkl")
        self.xgb.load(target_dir / "xgb.pkl")
        self.logistic.load(str(target_dir / "logistic.joblib"))

        if not self.use_stacking_fallback:
            self.meta_learner = joblib.load(target_dir / "meta_learner.joblib")

        logger.info(f"[{self.timeframe}] MetaEnsemble loaded ← {target_dir}")
        return self
