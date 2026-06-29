"""
LightGBM Ranker Model

Transitions from simple linear combination to a gradient boosted ranker.
Formulates the problem as a learning-to-rank task (LambdaMART) across the cross-section of stocks.
"""

import os
import pickle
from typing import Any

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from utils.logger import get_logger

logger = get_logger("lightgbm_ranker")


class LightGBMRankerModel:
    """
    Wrapper around lightgbm.LGBMRanker for cross-sectional alpha ranking.
    Designed to ingest multi-horizon labels and sector-neutralized features.
    """

    def __init__(self, model_dir: str = "models/saved/"):
        self.model_dir = model_dir
        self.model = None
        self.feature_names = []

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)

        if lgb is None:
            logger.warning("lightgbm package is not installed. Model training/inference will fail.")

    def _prepare_query_groups(self, data: pd.DataFrame, date_column: str = "date") -> np.ndarray:
        """
        LightGBMRanker requires a 'group' array specifying the number of items per query (date).
        The data must be sorted by the date column.
        """
        # Ensure data is sorted by date
        counts = data.groupby(date_column).size().values
        return counts

    def train(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        feature_columns: list[str],
        n_splits: int = 5,
        params: dict[str, Any] | None = None,
    ):
        """
        LightGBM LambdaRank — directly optimizes NDCG / rank correlation.
        Critical: group parameter tells LGB which rows belong to same date.
        """
        if lgb is None:
            raise ImportError("lightgbm is required to train the ranker.")

        self.feature_names = feature_columns

        # We assume labels_df is a Series or a DataFrame with a single column of targets
        labels = labels_df.iloc[:, 0] if isinstance(labels_df, pd.DataFrame) else labels_df

        # Convert to ranking labels (1-10 decile, within each cross-section date)
        if "date" in labels.index.names:
            groupby_col = "date"
        else:
            groupby_col = labels.index.names[0]  # Fallback

        # Discretize to 1-10 decile
        rank_labels = (
            labels.groupby(level=groupby_col)
            .transform(lambda x: pd.qcut(x, 10, labels=False, duplicates="drop"))
            .fillna(0)
            .astype(int)
        )

        default_params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [5, 10],  # evaluate at top 5 and top 10 stocks
            "n_estimators": 300,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 30,  # critical: avoids overfit on small universe
            "verbose": -1,
            "n_jobs": -1,
            "random_state": 42,
        }

        if params:
            default_params.update(params)

        self.model = lgb.LGBMRanker(**default_params)

        # Walk-forward: train on past, validate on future
        all_dates = features_df.index.get_level_values(groupby_col).unique().sort_values()
        if len(all_dates) < n_splits:
            logger.warning(
                f"Not enough dates ({len(all_dates)}) for {n_splits} splits. Training on all data."
            )
            X_tr, y_tr = features_df[feature_columns], rank_labels
            g_tr = features_df.groupby(level=groupby_col).size().values
            self.model.fit(X_tr, y_tr, group=g_tr)
            return

        fold_size = len(all_dates) // n_splits
        ic_scores = []
        val_preds_all = []
        val_wins_all = []

        from scipy.stats import spearmanr

        for fold in range(n_splits - 1):
            train_end = all_dates[fold * fold_size + fold_size]
            val_start = all_dates[fold * fold_size + fold_size]
            val_end = all_dates[min((fold + 1) * fold_size + fold_size, len(all_dates) - 1)]

            train_mask = features_df.index.get_level_values(groupby_col) < train_end
            val_mask = (features_df.index.get_level_values(groupby_col) >= val_start) & (
                features_df.index.get_level_values(groupby_col) < val_end
            )

            X_tr, y_tr = features_df.loc[train_mask, feature_columns], rank_labels[train_mask]
            X_va, y_va = features_df.loc[val_mask, feature_columns], labels[val_mask]  # raw for IC
            g_tr = features_df[train_mask].groupby(level=groupby_col).size().values

            if len(X_tr) == 0 or len(X_va) == 0:
                continue

            self.model.fit(X_tr, y_tr, group=g_tr)

            # Evaluate: IC per day on val set
            preds = self.model.predict(X_va)
            val_preds_all.extend(preds)
            val_wins_all.extend((y_va > 0).astype(int).values)
            daily_ics = []
            pred_series = pd.Series(preds, index=X_va.index)

            for date, grp in pred_series.groupby(level=groupby_col):
                if date in y_va.index.get_level_values(groupby_col):
                    actual = (
                        y_va.xs(date, level=groupby_col)
                        if isinstance(y_va.index, pd.MultiIndex)
                        else y_va.loc[date]
                    )
                    if len(grp) > 5 and len(actual) > 5:
                        if isinstance(grp.index, pd.MultiIndex):
                            grp_align = grp.droplevel(groupby_col)
                        else:
                            grp_align = grp

                        # Align by index within date
                        aligned = pd.concat([grp_align, actual], axis=1).dropna()
                        if len(aligned) > 5:
                            ic, _ = spearmanr(aligned.iloc[:, 0].values, aligned.iloc[:, 1].values)
                            if not np.isnan(ic):
                                daily_ics.append(ic)

            fold_ic = np.mean(daily_ics) if daily_ics else 0
            ic_scores.append(fold_ic)
            logger.info(f"  Fold {fold + 1}: IC={fold_ic:.4f}")

        mean_ic = np.mean(ic_scores) if ic_scores else 0
        logger.info(f"Mean walk-forward IC: {mean_ic:.4f}")

        # Final fit on all data
        logger.info("Fitting final model on all data...")
        X_all, y_all = features_df[feature_columns], rank_labels
        g_all = features_df.groupby(level=groupby_col).size().values
        self.model.fit(X_all, y_all, group=g_all)

        # Win Probability Calibration via Isotonic Regression (Out of Sample)
        logger.info(
            "Fitting Isotonic Regression for Win% Probability Calibration on Out-of-Sample predictions..."
        )
        from sklearn.isotonic import IsotonicRegression

        self.calibrator = IsotonicRegression(out_of_bounds="clip")

        if len(val_preds_all) > 10:
            self.calibrator.fit(np.array(val_preds_all), np.array(val_wins_all))
            logger.info(
                f"Calibration fitted successfully on {len(val_preds_all)} out-of-sample data points."
            )
        else:
            logger.warning(
                "Not enough out-of-sample data points for calibration. Falling back to in-sample."
            )
            raw_scores_all = self.model.predict(X_all)
            binary_wins = (labels > 0).astype(int)
            self.calibrator.fit(raw_scores_all, binary_wins)

        logger.info("Training completed.")

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate raw ranking scores and calibrated Win probabilities.
        """
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")

        # Check missing features
        missing = [f for f in self.feature_names if f not in features_df.columns]
        if missing:
            raise ValueError(f"Missing required features for prediction: {missing}")

        X = features_df[self.feature_names]
        scores = self.model.predict(X)

        win_probs = (
            self.calibrator.predict(scores)
            if hasattr(self, "calibrator")
            else np.zeros_like(scores)
        )

        return pd.DataFrame(
            {"alpha_score": scores, "win_probability": win_probs}, index=features_df.index
        )

    def save(self, model_name: str = "lgbm_ranker_v1.pkl"):
        """Serialize model to disk."""
        if self.model is None:
            raise ValueError("No model to save.")

        path = os.path.join(self.model_dir, model_name)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "feature_names": self.feature_names,
                    "calibrator": getattr(self, "calibrator", None),
                },
                f,
            )
        logger.info(f"Model saved to {path}")

    def load(self, model_name: str = "lgbm_ranker_v1.pkl"):
        """Load serialized model."""
        path = os.path.join(self.model_dir, model_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.feature_names = data["feature_names"]
            if "calibrator" in data:
                self.calibrator = data["calibrator"]
        logger.info(f"Model loaded from {path}")
