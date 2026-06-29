"""
India Ensemble Model

3-model ensemble for NSE cross-sectional prediction.
Each model sees the same features but is tuned differently.
Final signal = rank-average of 3 predictions.
"""

from dataclasses import dataclass
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

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available. Ensemble will use fallback models.")


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
        """Convert to dictionary."""
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
        """
        Initialize ensemble model.

        Args:
            min_model_ic: Minimum IC required for a model to contribute
        """
        self.min_model_ic = min_model_ic
        self.logger = logger

        # Model 1: LightGBM — captures non-linear interactions
        if LIGHTGBM_AVAILABLE:
            self.lgb = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.03,
                num_leaves=31,  # keep shallow — prevents overfit
                feature_fraction=0.6,  # use 60% features per tree
                bagging_fraction=0.8,
                bagging_freq=5,
                min_child_samples=30,  # important: prevents overfit on small NSE universe
                reg_alpha=0.1,
                reg_lambda=0.1,
                verbose=-1,
            )
        else:
            self.lgb = None

        # Model 2: Ridge — captures linear factor exposures
        # Good at stable relationships like beta, size, value
        if SKLEARN_AVAILABLE:
            self.ridge = Ridge(alpha=10.0)
        else:
            self.ridge = None

        # Model 3: LightGBM tuned for longer horizon
        # Different n_estimators and leaf depth = different bias
        if LIGHTGBM_AVAILABLE:
            self.lgb2 = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=15,  # even shallower
                feature_fraction=0.5,
                min_child_samples=50,
                verbose=-1,
            )
        else:
            self.lgb2 = None

        # Model weights (can be optimized later)
        self.weights = [1.0, 1.0, 1.0]

        # Model names
        self.model_names = ["LGB1", "Ridge", "LGB2"]

        self.logger.info("IndiaEnsembleModel initialized")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        """
        Fit all models in the ensemble.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        """
        # Fit LightGBM model 1
        if self.lgb is not None:
            self.lgb.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            self.logger.info("LightGBM model 1 fitted")

        # Fit Ridge model
        if self.ridge is not None:
            self.ridge.fit(X_train, y_train)
            self.logger.info("Ridge model fitted")
        else:
            self.logger.warning("Ridge model not available")

        # Fit LightGBM model 2
        if self.lgb2 is not None:
            self.lgb2.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            self.logger.info("LightGBM model 2 fitted")

        self.logger.info("All ensemble models fitted")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using ensemble.

        Get raw predictions from each model, convert to ranks (0–1 scale),
        then average. Rank-averaging is more robust than score-averaging.

        Args:
            X: Features to predict

        Returns:
            Ensemble predictions
        """
        predictions = []

        # Get predictions from each model
        if self.lgb is not None:
            p1 = self.lgb.predict(X)
            predictions.append(p1)
        else:
            predictions.append(np.zeros(len(X)))

        if self.ridge is not None:
            p2 = self.ridge.predict(X)
            predictions.append(p2)
        else:
            predictions.append(np.zeros(len(X)))

        if self.lgb2 is not None:
            p3 = self.lgb2.predict(X)
            predictions.append(p3)
        else:
            predictions.append(np.zeros(len(X)))

        # Convert each to ranks (0–1 scale) before averaging
        ranked_predictions = []
        for pred in predictions:
            r = rankdata(pred) / len(pred)
            ranked_predictions.append(r)

        # Equal-weight ensemble (can optimize weights later)
        ensemble_pred = np.average(ranked_predictions, axis=0, weights=self.weights)

        return ensemble_pred

    def ic_per_model(
        self, X_test: pd.DataFrame, y_test: pd.Series, returns: pd.Series | None = None
    ) -> EnsembleMetrics:
        """
        Check which model contributes most — prune if one is useless.

        Only add a model to the ensemble if its standalone IC > 0.015.
        A model with IC near zero adds noise, not diversification.

        Args:
            X_test: Test features
            y_test: Test targets
            returns: Optional returns series for Sharpe calculation

        Returns:
            Ensemble metrics with IC, turnover, Sharpe, and correlation matrix
        """
        print("\n--- IC PER MODEL ---")

        # Get predictions from each model
        predictions = []
        model_names = []

        if self.lgb is not None:
            p1 = self.lgb.predict(X_test)
            predictions.append(p1)
            model_names.append("LGB1")
        else:
            predictions.append(np.zeros(len(X_test)))
            model_names.append("LGB1 (fallback)")

        if self.ridge is not None:
            p2 = self.ridge.predict(X_test)
            predictions.append(p2)
            model_names.append("Ridge")
        else:
            predictions.append(np.zeros(len(X_test)))
            model_names.append("Ridge (fallback)")

        if self.lgb2 is not None:
            p3 = self.lgb2.predict(X_test)
            predictions.append(p3)
            model_names.append("LGB2")
        else:
            predictions.append(np.zeros(len(X_test)))
            model_names.append("LGB2 (fallback)")

        # Calculate IC for each model
        ics = []
        for name, pred in zip(model_names, predictions, strict=False):
            ic, _ = spearmanr(pred, y_test)
            ics.append(ic)
            print(f"  {name:15}: IC = {ic:.4f}")

        # Calculate ensemble IC
        ensemble_pred = self.predict(X_test)
        ensemble_ic, _ = spearmanr(ensemble_pred, y_test)
        print(f"  {'Ensemble':15}: IC = {ensemble_ic:.4f}")

        # Calculate turnover for each model
        turnovers = self._calculate_turnover(predictions)
        for name, turnover in zip(model_names, turnovers, strict=False):
            print(f"  {name:15}: Turnover = {turnover:.4f}")

        ensemble_turnover = self._calculate_turnover([ensemble_pred])[0]
        print(f"  {'Ensemble':15}: Turnover = {ensemble_turnover:.4f}")

        # Calculate Sharpe ratio for each model (if returns provided)
        sharpes = [0.0, 0.0, 0.0]
        ensemble_sharpe = 0.0
        if returns is not None and len(returns) == len(X_test):
            for i, pred in enumerate(predictions):
                # Convert predictions to ranks and calculate strategy returns
                ranked_pred = rankdata(pred) / len(pred)
                # Top quintile long, bottom quintile short
                long_mask = ranked_pred >= 0.8
                short_mask = ranked_pred <= 0.2
                strategy_returns = np.where(long_mask, returns, np.where(short_mask, -returns, 0))
                sharpes[i] = (
                    np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
                    if np.std(strategy_returns) > 0
                    else 0.0
                )
                print(f"  {model_names[i]:15}: Sharpe = {sharpes[i]:.4f}")

            # Ensemble Sharpe
            ranked_ensemble = rankdata(ensemble_pred) / len(ensemble_pred)
            long_mask = ranked_ensemble >= 0.8
            short_mask = ranked_ensemble <= 0.2
            ensemble_strategy_returns = np.where(
                long_mask, returns, np.where(short_mask, -returns, 0)
            )
            ensemble_sharpe = (
                np.mean(ensemble_strategy_returns)
                / np.std(ensemble_strategy_returns)
                * np.sqrt(252)
                if np.std(ensemble_strategy_returns) > 0
                else 0.0
            )
            print(f"  {'Ensemble':15}: Sharpe = {ensemble_sharpe:.4f}")

        # Calculate correlation matrix
        correlation_matrix = self._calculate_correlation_matrix(predictions, model_names)
        print("\n--- CORRELATION MATRIX ---")
        for i, name1 in enumerate(model_names):
            for j, name2 in enumerate(model_names):
                if i < j:
                    corr = correlation_matrix[name1][name2]
                    print(f"  {name1:15} vs {name2:15}: {corr:.4f}")

        # Check which models contribute
        print("\n--- MODEL CONTRIBUTION ---")
        for name, ic in zip(model_names, ics, strict=False):
            if ic > self.min_model_ic:
                print(f"  ✓ {name}: IC {ic:.4f} > {self.min_model_ic} - KEEP")
            else:
                print(f"  ✗ {name}: IC {ic:.4f} < {self.min_model_ic} - CONSIDER REMOVING")

        # Validate ensemble beats best component
        best_component_ic = max(ics)
        ensemble_beats_best = ensemble_ic > best_component_ic
        print("\n--- ENSEMBLE VALIDATION ---")
        print(f"  Best component IC: {best_component_ic:.4f}")
        print(f"  Ensemble IC: {ensemble_ic:.4f}")
        print(f"  Ensemble beats best: {ensemble_beats_best}")

        if not ensemble_beats_best:
            print(
                "  ⚠ WARNING: Ensemble does NOT beat best component. Consider simplifying to single best model."
            )

        return EnsembleMetrics(
            ensemble_ic=ensemble_ic,
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

    def _calculate_turnover(self, predictions: list[np.ndarray]) -> list[float]:
        """
        Calculate turnover for each model's predictions.

        Turnover = average absolute change in rank position between consecutive predictions.
        For single-time-point prediction, this is a placeholder.

        Args:
            predictions: List of prediction arrays

        Returns:
            List of turnover values
        """
        turnovers = []
        for pred in predictions:
            # For single time point, use rank variance as proxy for turnover potential
            ranks = rankdata(pred)
            rank_variance = np.var(ranks)
            # Normalize by max possible variance
            max_variance = (len(pred) ** 2) / 12
            turnover = rank_variance / max_variance if max_variance > 0 else 0.0
            turnovers.append(turnover)
        return turnovers

    def _calculate_correlation_matrix(
        self, predictions: list[np.ndarray], model_names: list[str]
    ) -> dict[str, dict[str, float]]:
        """
        Calculate correlation matrix between model predictions.

        Args:
            predictions: List of prediction arrays
            model_names: List of model names

        Returns:
            Correlation matrix as nested dictionary
        """
        correlation_matrix = {}
        for i, name1 in enumerate(model_names):
            correlation_matrix[name1] = {}
            for j, name2 in enumerate(model_names):
                if i == j:
                    correlation_matrix[name1][name2] = 1.0
                else:
                    corr, _ = spearmanr(predictions[i], predictions[j])
                    correlation_matrix[name1][name2] = corr
        return correlation_matrix

    def prune_ensemble(
        self, X_test: pd.DataFrame, y_test: pd.Series, returns: pd.Series | None = None
    ) -> tuple[EnsembleMetrics, bool]:
        """
        Prune ensemble by removing underperforming models.

        If ensemble does not beat best component, recommend removing ensemble entirely.

        Args:
            X_test: Test features
            y_test: Test targets
            returns: Optional returns series for Sharpe calculation

        Returns:
            Tuple of (metrics, should_delete_ensemble)
        """
        metrics = self.ic_per_model(X_test, y_test, returns)

        # Check if ensemble beats best component
        if not metrics.ensemble_beats_best:
            print("\n--- ENSEMBLE PRUNING DECISION ---")
            print(
                f"  Ensemble IC ({metrics.ensemble_ic:.4f}) does NOT exceed best component IC ({metrics.best_component_ic:.4f})"
            )
            print("  RECOMMENDATION: DELETE ENSEMBLE, USE SINGLE BEST MODEL")
            return metrics, True

        # If ensemble beats best, check if any individual models should be removed
        print("\n--- INDIVIDUAL MODEL PRUNING ---")
        models_to_remove = []

        # Check model 1
        if metrics.model1_ic < self.min_model_ic:
            print(f"  Model 1 (LGB1): IC {metrics.model1_ic:.4f} < {self.min_model_ic} - REMOVE")
            models_to_remove.append(0)

        # Check model 2
        if metrics.model2_ic < self.min_model_ic:
            print(f"  Model 2 (Ridge): IC {metrics.model2_ic:.4f} < {self.min_model_ic} - REMOVE")
            models_to_remove.append(1)

        # Check model 3
        if metrics.model3_ic < self.min_model_ic:
            print(f"  Model 3 (LGB2): IC {metrics.model3_ic:.4f} < {self.min_model_ic} - REMOVE")
            models_to_remove.append(2)

        if models_to_remove:
            print(f"\n  Removing {len(models_to_remove)} underperforming models")
            # Set weights to 0 for removed models
            for idx in models_to_remove:
                self.weights[idx] = 0.0
            print(f"  Updated weights: {self.weights}")
        else:
            print("  All models meet minimum IC threshold - KEEP ALL")

        return metrics, False

    def optimize_weights(self, X_val: pd.DataFrame, y_val: pd.Series) -> list[float]:
        """
        Optimize ensemble weights based on validation performance.

        Args:
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Optimized weights
        """
        # Simple grid search for weights
        best_weights = self.weights
        best_ic = 0.0

        # Try different weight combinations
        for w1 in [0.0, 0.5, 1.0, 1.5, 2.0]:
            for w2 in [0.0, 0.5, 1.0, 1.5, 2.0]:
                for w3 in [0.0, 0.5, 1.0, 1.5, 2.0]:
                    if w1 + w2 + w3 == 0:
                        continue

                    self.weights = [w1, w2, w3]
                    pred = self.predict(X_val)
                    ic, _ = spearmanr(pred, y_val)

                    if ic > best_ic:
                        best_ic = ic
                        best_weights = [w1, w2, w3]

        self.weights = best_weights
        self.logger.info(f"Optimized weights: {best_weights}, IC: {best_ic:.4f}")

        return best_weights

    def get_feature_importance(self) -> dict[str, dict[str, float]]:
        """
        Get feature importance from each model.

        Returns:
            Dictionary mapping model names to feature importance
        """
        importance = {}

        if self.lgb is not None:
            importance["LGB1"] = dict(
                zip(self.lgb.feature_name_, self.lgb.feature_importances_, strict=False)
            )

        importance["Ridge"] = {}
        if self.ridge is not None and hasattr(self.ridge, "coef_"):
            importance["Ridge"] = dict(
                zip(self.ridge.feature_names_in_, np.abs(self.ridge.coef_), strict=False)
            )

        if self.lgb2 is not None:
            importance["LGB2"] = dict(
                zip(self.lgb2.feature_name_, self.lgb2.feature_importances_, strict=False)
            )

        return importance

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about ensemble models.

        Returns:
            Model information dictionary
        """
        info = {
            "n_models": 3,
            "model_names": self.model_names,
            "weights": self.weights,
            "min_model_ic": self.min_model_ic,
        }

        if self.lgb is not None:
            info["lgb1_params"] = self.lgb.get_params()

        info["ridge_params"] = self.ridge.get_params()

        if self.lgb2 is not None:
            info["lgb2_params"] = self.lgb2.get_params()

        return info
