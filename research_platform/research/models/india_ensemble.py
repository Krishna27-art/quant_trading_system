"""
India Ensemble Model

3-model ensemble for NSE cross-sectional prediction.
Each model sees the same features but is tuned differently.
Final signal = rank-average of 3 predictions.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from utils.logger import get_logger

logger = get_logger("india_ensemble")

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. Ensemble will use fallback models.")

try:
    from sklearn.linear_model import Ridge
    from sklearn.utils.validation import check_is_fitted

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available. Ensemble will use fallback models.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_lgb_fitted(model) -> bool:
    """Return True if a LightGBM model has been fitted."""
    if model is None:
        return False
    return hasattr(model, "booster_") and model.booster_ is not None


def _is_ridge_fitted(model) -> bool:
    """Return True if a Ridge model has been fitted."""
    if model is None:
        return False
    try:
        check_is_fitted(model)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class EnsembleMetrics:
    """Metrics for ensemble model performance."""

    ensemble_ic: float
    model1_ic: float
    model2_ic: float
    model3_ic: float
    ensemble_sharpe: float = 0.0
    model1_sharpe: float = 0.0
    model2_sharpe: float = 0.0
    model3_sharpe: float = 0.0
    ensemble_turnover: float = 0.0
    model1_turnover: float = 0.0
    model2_turnover: float = 0.0
    model3_turnover: float = 0.0
    correlation_matrix: dict[str, dict[str, float]] | None = None
    ensemble_beats_best: bool = False
    best_component_ic: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ensemble_ic": self.ensemble_ic,
            "model1_ic": self.model1_ic,
            "model2_ic": self.model2_ic,
            "model3_ic": self.model3_ic,
            "ensemble_sharpe": self.ensemble_sharpe,
            "model1_sharpe": self.model1_sharpe,
            "model2_sharpe": self.model2_sharpe,
            "model3_sharpe": self.model3_sharpe,
            "ensemble_turnover": self.ensemble_turnover,
            "model1_turnover": self.model1_turnover,
            "model2_turnover": self.model2_turnover,
            "model3_turnover": self.model3_turnover,
            "correlation_matrix": self.correlation_matrix,
            "ensemble_beats_best": self.ensemble_beats_best,
            "best_component_ic": self.best_component_ic,
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class IndiaEnsembleModel:
    """
    3-model ensemble for NSE cross-sectional prediction.

    Each model sees the same features but is tuned differently.
    Final signal = rank-average of 3 predictions.

    Models:
    - Model 1: LightGBM — captures non-linear interactions
    - Model 2: Ridge — captures linear factor exposures
    - Model 3: LightGBM tuned for longer horizon — different bias
    """

    def __init__(self, min_model_ic: float = 0.015):
        self.min_model_ic = min_model_ic
        self.logger = logger

        # Model 1: LightGBM — non-linear interactions
        if LIGHTGBM_AVAILABLE:
            self.lgb = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.03,
                num_leaves=31,        # shallow — prevents overfit
                feature_fraction=0.6, # 60% features per tree
                bagging_fraction=0.8,
                bagging_freq=5,
                min_child_samples=30, # critical on small NSE universe
                reg_alpha=0.1,
                reg_lambda=0.1,
                verbose=-1,
            )
        else:
            self.lgb = None

        # Model 2: Ridge — stable linear factor exposures (beta, size, value)
        if SKLEARN_AVAILABLE:
            self.ridge = Ridge(alpha=10.0)
        else:
            self.ridge = None

        # Model 3: LightGBM tuned for longer horizon (shallower, fewer trees)
        if LIGHTGBM_AVAILABLE:
            self.lgb2 = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=15,        # even shallower
                feature_fraction=0.5,
                min_child_samples=50,
                verbose=-1,
            )
        else:
            self.lgb2 = None

        # Weights — set to 0 for pruned models
        self.weights: list[float] = [1.0, 1.0, 1.0]
        self.model_names: list[str] = ["LGB1", "Ridge", "LGB2"]

        self.logger.info("IndiaEnsembleModel initialized")

    # -----------------------------------------------------------------------
    # Fit
    # -----------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        """Fit all models. Early-stopping failures per model are isolated."""

        # Model 1: LightGBM
        if self.lgb is not None:
            try:
                self.lgb.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(30, verbose=False)],
                )
                self.logger.info("LightGBM model 1 fitted")
            except Exception as exc:
                self.logger.error(f"LightGBM model 1 fit failed: {exc} — fitting without early stopping")
                try:
                    self.lgb.set_params(n_estimators=300)
                    self.lgb.fit(X_train, y_train)
                    self.logger.info("LightGBM model 1 fitted (no early stopping)")
                except Exception as exc2:
                    self.logger.error(f"LightGBM model 1 fallback fit also failed: {exc2}")
                    self.lgb = None

        # Model 2: Ridge
        if self.ridge is not None:
            try:
                self.ridge.fit(X_train, y_train)
                self.logger.info("Ridge model fitted")
            except Exception as exc:
                self.logger.error(f"Ridge fit failed: {exc}")
                self.ridge = None
        else:
            self.logger.warning("Ridge model not available")

        # Model 3: LightGBM (longer horizon)
        if self.lgb2 is not None:
            try:
                self.lgb2.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(30, verbose=False)],
                )
                self.logger.info("LightGBM model 2 fitted")
            except Exception as exc:
                self.logger.error(f"LightGBM model 2 fit failed: {exc} — fitting without early stopping")
                try:
                    self.lgb2.set_params(n_estimators=200)
                    self.lgb2.fit(X_train, y_train)
                    self.logger.info("LightGBM model 2 fitted (no early stopping)")
                except Exception as exc2:
                    self.logger.error(f"LightGBM model 2 fallback fit also failed: {exc2}")
                    self.lgb2 = None

        self.logger.info("Ensemble fit complete")

    # -----------------------------------------------------------------------
    # Predict
    # -----------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Rank-average ensemble prediction.

        Only models that are (a) fitted and (b) have weight > 0 contribute.
        If no model is available, returns uniform ranks (0.5 for all rows).

        Args:
            X: Feature DataFrame, same schema used during fit.

        Returns:
            1-D array of ensemble scores in [0, 1]; higher = more bullish.
        """
        n = len(X)
        ranked_predictions: list[np.ndarray] = []
        active_weights: list[float] = []

        model_slots = [
            (self.lgb,   _is_lgb_fitted,   self.weights[0], "LGB1"),
            (self.ridge, _is_ridge_fitted,  self.weights[1], "Ridge"),
            (self.lgb2,  _is_lgb_fitted,    self.weights[2], "LGB2"),
        ]

        for model, is_fitted_fn, weight, name in model_slots:
            # Skip pruned models
            if weight == 0.0:
                continue
            # Skip unfitted or None models
            if not is_fitted_fn(model):
                self.logger.warning(f"{name} is not fitted — excluded from prediction")
                continue

            try:
                raw = model.predict(X)
                # Rank to [0, 1] so models with different score scales are comparable
                ranked = rankdata(raw) / n
                ranked_predictions.append(ranked)
                active_weights.append(weight)
            except Exception as exc:
                self.logger.error(f"{name} prediction failed: {exc} — excluded")

        if not ranked_predictions:
            self.logger.error("No fitted models available — returning uniform ranks (0.5)")
            return np.full(n, 0.5)

        # Weighted average of rank-normalised predictions
        ensemble_pred = np.average(ranked_predictions, axis=0, weights=active_weights)
        return ensemble_pred

    # -----------------------------------------------------------------------
    # IC / diagnostics
    # -----------------------------------------------------------------------

    def ic_per_model(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        returns: pd.Series | None = None,
    ) -> EnsembleMetrics:
        """
        Per-model IC, turnover, Sharpe, and correlation matrix on test set.

        Args:
            X_test: Test features
            y_test: Test targets (forward returns / labels)
            returns: Optional realised returns for Sharpe calculation

        Returns:
            EnsembleMetrics dataclass
        """
        n = len(X_test)
        print("\n--- IC PER MODEL ---")

        raw_preds: list[np.ndarray] = []
        model_names_out: list[str] = []

        model_slots = [
            (self.lgb,   _is_lgb_fitted,   "LGB1"),
            (self.ridge, _is_ridge_fitted,  "Ridge"),
            (self.lgb2,  _is_lgb_fitted,    "LGB2"),
        ]

        for model, is_fitted_fn, name in model_slots:
            if is_fitted_fn(model):
                try:
                    raw_preds.append(model.predict(X_test))
                except Exception as exc:
                    self.logger.error(f"{name} predict failed in ic_per_model: {exc}")
                    raw_preds.append(np.zeros(n))
            else:
                raw_preds.append(np.zeros(n))
            model_names_out.append(name)

        ics: list[float] = []
        for name, pred in zip(model_names_out, raw_preds):
            ic, _ = spearmanr(pred, y_test)
            ics.append(float(ic))
            print(f"  {name:15}: IC = {ic:.4f}")

        ensemble_pred = self.predict(X_test)
        ensemble_ic, _ = spearmanr(ensemble_pred, y_test)
        print(f"  {'Ensemble':15}: IC = {ensemble_ic:.4f}")

        # Turnover
        turnovers = self._calculate_turnover(raw_preds)
        for name, turnover in zip(model_names_out, turnovers):
            print(f"  {name:15}: Turnover = {turnover:.4f}")
        ensemble_turnover = self._calculate_turnover([ensemble_pred])[0]
        print(f"  {'Ensemble':15}: Turnover = {ensemble_turnover:.4f}")

        # Sharpe (optional)
        sharpes = [0.0, 0.0, 0.0]
        ensemble_sharpe = 0.0
        if returns is not None and len(returns) == n:
            returns_arr = np.asarray(returns, dtype=float)
            for i, pred in enumerate(raw_preds):
                ranked_pred = rankdata(pred) / n
                long_mask  = ranked_pred >= 0.8
                short_mask = ranked_pred <= 0.2
                strat = np.where(long_mask, returns_arr, np.where(short_mask, -returns_arr, 0.0))
                std = np.std(strat)
                sharpes[i] = float(np.mean(strat) / std * np.sqrt(252)) if std > 0 else 0.0
                print(f"  {model_names_out[i]:15}: Sharpe = {sharpes[i]:.4f}")

            ranked_ens = rankdata(ensemble_pred) / n
            long_mask  = ranked_ens >= 0.8
            short_mask = ranked_ens <= 0.2
            ens_strat = np.where(long_mask, returns_arr, np.where(short_mask, -returns_arr, 0.0))
            std = np.std(ens_strat)
            ensemble_sharpe = float(np.mean(ens_strat) / std * np.sqrt(252)) if std > 0 else 0.0
            print(f"  {'Ensemble':15}: Sharpe = {ensemble_sharpe:.4f}")

        # Correlation matrix
        correlation_matrix = self._calculate_correlation_matrix(raw_preds, model_names_out)
        print("\n--- CORRELATION MATRIX ---")
        for i, name1 in enumerate(model_names_out):
            for j, name2 in enumerate(model_names_out):
                if i < j:
                    print(f"  {name1:15} vs {name2:15}: {correlation_matrix[name1][name2]:.4f}")

        # Contribution check
        print("\n--- MODEL CONTRIBUTION ---")
        for name, ic in zip(model_names_out, ics):
            symbol = "✓" if ic > self.min_model_ic else "✗"
            action = "KEEP" if ic > self.min_model_ic else "CONSIDER REMOVING"
            print(f"  {symbol} {name}: IC {ic:.4f} {'>' if ic > self.min_model_ic else '<'} {self.min_model_ic} - {action}")

        best_component_ic = max(ics)
        ensemble_beats_best = float(ensemble_ic) > best_component_ic
        print("\n--- ENSEMBLE VALIDATION ---")
        print(f"  Best component IC : {best_component_ic:.4f}")
        print(f"  Ensemble IC       : {ensemble_ic:.4f}")
        print(f"  Ensemble beats best: {ensemble_beats_best}")
        if not ensemble_beats_best:
            print("  ⚠ WARNING: Ensemble does NOT beat best component. Consider simplifying.")

        return EnsembleMetrics(
            ensemble_ic=float(ensemble_ic),
            model1_ic=ics[0],
            model2_ic=ics[1],
            model3_ic=ics[2],
            ensemble_sharpe=ensemble_sharpe,
            model1_sharpe=sharpes[0],
            model2_sharpe=sharpes[1],
            model3_sharpe=sharpes[2],
            ensemble_turnover=ensemble_turnover,
            model1_turnover=turnovers[0],
            model2_turnover=turnovers[1],
            model3_turnover=turnovers[2],
            correlation_matrix=correlation_matrix,
            ensemble_beats_best=ensemble_beats_best,
            best_component_ic=best_component_ic,
        )

    # -----------------------------------------------------------------------
    # Pruning
    # -----------------------------------------------------------------------

    def prune_ensemble(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        returns: pd.Series | None = None,
    ) -> tuple[EnsembleMetrics, bool]:
        """
        Set weights=0 for sub-threshold models.  Returns (metrics, should_delete_ensemble).

        A True second return means the ensemble is worse than its best single component
        and you should use that component directly.
        """
        metrics = self.ic_per_model(X_test, y_test, returns)

        if not metrics.ensemble_beats_best:
            print("\n--- ENSEMBLE PRUNING DECISION ---")
            print(
                f"  Ensemble IC ({metrics.ensemble_ic:.4f}) does NOT exceed "
                f"best component IC ({metrics.best_component_ic:.4f})"
            )
            print("  RECOMMENDATION: DELETE ENSEMBLE, USE SINGLE BEST MODEL")
            return metrics, True

        print("\n--- INDIVIDUAL MODEL PRUNING ---")
        component_ics = [metrics.model1_ic, metrics.model2_ic, metrics.model3_ic]
        models_removed = 0
        for idx, (name, ic) in enumerate(zip(self.model_names, component_ics)):
            if ic < self.min_model_ic:
                print(f"  Model {idx + 1} ({name}): IC {ic:.4f} < {self.min_model_ic} - REMOVING")
                self.weights[idx] = 0.0
                models_removed += 1
            else:
                print(f"  Model {idx + 1} ({name}): IC {ic:.4f} >= {self.min_model_ic} - KEEP")

        if models_removed:
            # Guard: if all weights zeroed, restore equal weights
            if sum(self.weights) == 0.0:
                self.logger.warning("All models pruned — restoring equal weights to avoid ZeroDivisionError")
                self.weights = [1.0, 1.0, 1.0]
            else:
                print(f"  Updated weights: {self.weights}")
        else:
            print("  All models meet minimum IC threshold - KEEP ALL")

        return metrics, False

    # -----------------------------------------------------------------------
    # Weight optimisation
    # -----------------------------------------------------------------------

    def optimize_weights(self, X_val: pd.DataFrame, y_val: pd.Series) -> list[float]:
        """
        Grid-search over weight combinations on validation set.
        Preserves zero-weights for models that were pruned.
        """
        best_weights = list(self.weights)
        best_ic = -np.inf

        candidates = [0.0, 0.5, 1.0, 1.5, 2.0]

        for w1 in candidates:
            for w2 in candidates:
                for w3 in candidates:
                    # Respect pruned models (don't restore zeroed weights)
                    trial = [
                        w1 if self.weights[0] > 0 else 0.0,
                        w2 if self.weights[1] > 0 else 0.0,
                        w3 if self.weights[2] > 0 else 0.0,
                    ]
                    if sum(trial) == 0:
                        continue

                    self.weights = trial
                    pred = self.predict(X_val)
                    ic, _ = spearmanr(pred, y_val)
                    if ic > best_ic:
                        best_ic = ic
                        best_weights = list(trial)

        self.weights = best_weights
        self.logger.info(f"Optimized weights: {best_weights}, IC: {best_ic:.4f}")
        return best_weights

    # -----------------------------------------------------------------------
    # Feature importance
    # -----------------------------------------------------------------------

    def get_feature_importance(self) -> dict[str, dict[str, float]]:
        """Feature importance from each fitted model."""
        importance: dict[str, dict[str, float]] = {}

        if _is_lgb_fitted(self.lgb):
            try:
                importance["LGB1"] = dict(
                    zip(self.lgb.feature_name_, self.lgb.feature_importances_, strict=False)
                )
            except Exception as exc:
                self.logger.warning(f"Could not extract LGB1 feature importance: {exc}")
                importance["LGB1"] = {}
        else:
            importance["LGB1"] = {}

        if _is_ridge_fitted(self.ridge):
            try:
                importance["Ridge"] = dict(
                    zip(self.ridge.feature_names_in_, np.abs(self.ridge.coef_), strict=False)
                )
            except Exception as exc:
                self.logger.warning(f"Could not extract Ridge feature importance: {exc}")
                importance["Ridge"] = {}
        else:
            importance["Ridge"] = {}

        if _is_lgb_fitted(self.lgb2):
            try:
                importance["LGB2"] = dict(
                    zip(self.lgb2.feature_name_, self.lgb2.feature_importances_, strict=False)
                )
            except Exception as exc:
                self.logger.warning(f"Could not extract LGB2 feature importance: {exc}")
                importance["LGB2"] = {}
        else:
            importance["LGB2"] = {}

        return importance

    # -----------------------------------------------------------------------
    # Model info
    # -----------------------------------------------------------------------

    def get_model_info(self) -> dict[str, Any]:
        """Serialisable summary of ensemble configuration."""
        info: dict[str, Any] = {
            "n_models": 3,
            "model_names": self.model_names,
            "weights": self.weights,
            "min_model_ic": self.min_model_ic,
            "lgb1_fitted": _is_lgb_fitted(self.lgb),
            "ridge_fitted": _is_ridge_fitted(self.ridge),
            "lgb2_fitted": _is_lgb_fitted(self.lgb2),
        }
        if _is_lgb_fitted(self.lgb):
            info["lgb1_params"] = self.lgb.get_params()
        if _is_ridge_fitted(self.ridge):
            info["ridge_params"] = self.ridge.get_params()
        if _is_lgb_fitted(self.lgb2):
            info["lgb2_params"] = self.lgb2.get_params()
        return info

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Pickle the full ensemble (models + weights + config) to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info(f"Ensemble saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "IndiaEnsembleModel":
        """Load a previously saved ensemble from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Ensemble model file not found: {path}")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is {type(obj)}, expected IndiaEnsembleModel")
        logger.info(f"Ensemble loaded from {path}")
        return obj

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _calculate_turnover(self, predictions: list[np.ndarray]) -> list[float]:
        """
        Rank-variance proxy for single-snapshot turnover potential.
        (True turnover requires consecutive time periods.)
        """
        turnovers: list[float] = []
        for pred in predictions:
            ranks = rankdata(pred)
            rank_variance = float(np.var(ranks))
            max_variance = (len(pred) ** 2) / 12
            turnovers.append(rank_variance / max_variance if max_variance > 0 else 0.0)
        return turnovers

    def _calculate_correlation_matrix(
        self, predictions: list[np.ndarray], model_names: list[str]
    ) -> dict[str, dict[str, float]]:
        """Spearman rank correlation between model predictions."""
        matrix: dict[str, dict[str, float]] = {}
        for i, name1 in enumerate(model_names):
            matrix[name1] = {}
            for j, name2 in enumerate(model_names):
                if i == j:
                    matrix[name1][name2] = 1.0
                else:
                    corr, _ = spearmanr(predictions[i], predictions[j])
                    matrix[name1][name2] = float(corr)
        return matrix