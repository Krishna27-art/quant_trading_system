"""
prediction_intelligence/base_logistic.py

Layer 2: Base Logistic Regression + Ensemble Models.

CRITICAL FIX: Previously BaseLogistic was never trained or loaded in the live
prediction path. generate_live_predictions.py stamped model_version=
'LOGREG_LONGTERM_v1' on LONGTERM signals but computed win_prob via hardcoded
arithmetic, not from this model.

This file provides:
  - BaseLogistic      : single LR pipeline with TimeSeriesSplit CV + save/load
  - EnsembleModel     : LR + RandomForest + GradientBoosting weighted ensemble
  - ModelRegistry     : singleton that loads/caches models from MODEL_PATH env
  - build_features()  : canonical feature engineering shared by train + live paths
                        (no lookahead — all indicators use .shift(1) where needed)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

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

try:
    from utils.logger import get_logger
    logger = get_logger("base_logistic")
except Exception:
    import logging
    logger = logging.getLogger("base_logistic")

# ---------------------------------------------------------------------------
# Default model storage path — overridden by MODEL_PATH env var or explicit arg
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_DIR = os.environ.get(
    "MODEL_PATH",
    str(Path(__file__).parent.parent / "data" / "production" / "models"),
)

# ---------------------------------------------------------------------------
# Canonical feature names
# All three timeframes use this same schema so the live path always knows
# which columns to pass.
# ---------------------------------------------------------------------------
from data_platform.features.canonical_builder import (
    CanonicalFeatureBuilder,
    LONGTERM_FEATURES,
    SWING_FEATURES,
    INTRADAY_FEATURES,
)

FEATURE_SCHEMA_VERSION = "v1.1.0"



def build_features(
    df: pd.DataFrame,
    timeframe: str,
    extra: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Build a feature DataFrame from an OHLCV candle DataFrame.
    Delegates to CanonicalFeatureBuilder (the single source of truth).
    """
    return CanonicalFeatureBuilder.build_features(df=df, timeframe=timeframe, extra=extra)


def build_label(
    df: pd.DataFrame,
    timeframe: str,
    target_pct: float | None = None,
    sl_pct: float | None = None,
    side: str = "long",
) -> pd.Series:
    """
    Triple-barrier binary label using TripleBarrierLabeler.
    Returns a pd.Series of 0 and 1, aligned with df's index.

    Horizons (in bars):
      INTRADAY: 10 bars (60m bars proxy)
      SWING: 15 bars (daily)
      LONGTERM: 12 bars (weekly)
    """
    from prediction_intelligence.triple_barrier import TripleBarrierLabeler

    tf = timeframe.upper()
    _defaults = {
        "INTRADAY": (0.015, -0.0075, 10),
        "SWING":    (0.030, -0.015, 15),
        "LONGTERM": (0.200, -0.100, 12),
    }
    default_tp, default_sl, default_horizon = _defaults.get(tf, (0.05, -0.025, 10))

    tp = target_pct if target_pct is not None else default_tp
    sl = -abs(sl_pct) if sl_pct is not None else default_sl  # Ensure negative
    horizon = default_horizon

    # Calculate symbol-specific transaction costs
    symbol = "DEFAULT"
    if "symbol" in df.columns:
        symbol = str(df["symbol"].iloc[0]).upper()
    elif "__symbol__" in df.columns:
        symbol = str(df["__symbol__"].iloc[0]).upper()

    SYMBOL_SLIPPAGE_MULTIPLIERS = {
        "RELIANCE": 0.8,
        "TCS": 0.8,
        "HDFCBANK": 0.8,
        "ICICIBANK": 0.8,
        "INFY": 0.9,
        "SBIN": 0.9,
        "ITC": 1.0,
        "HINDUNILVR": 1.0,
        "KOTAKBANK": 0.9,
        "AXISBANK": 1.0,
        "LT": 1.2,
        "ASIANPAINT": 1.0,
        "MARUTI": 1.2,
        "BAJFINANCE": 1.5,
        "WIPRO": 1.5,
    }
    BASE_ROUNDTRIP_COST_PCT = 0.004
    slippage_multiplier = SYMBOL_SLIPPAGE_MULTIPLIERS.get(symbol, 1.0)
    total_cost_pct = BASE_ROUNDTRIP_COST_PCT + (0.0005 * slippage_multiplier)

    # Cost-adjusted target barrier configuration
    if side.lower() == "long":
        upper_barrier_pct = tp + total_cost_pct
        lower_barrier_pct = sl
    else:
        # For a short trade:
        # Win (lower barrier) is hit when price falls by tp (adjusted for cost)
        # Loss (upper barrier) is hit when price rises by sl (absolute value)
        upper_barrier_pct = abs(sl)
        lower_barrier_pct = -(tp + total_cost_pct)

    # Instantiate labeler
    labeler = TripleBarrierLabeler(
        upper_barrier_pct=upper_barrier_pct,
        lower_barrier_pct=lower_barrier_pct,
        vertical_barrier_days=horizon,
        validate_labels=False,  # Skip hard raise to handle training skew robustly
    )

    # Resolve timestamps
    if isinstance(df.index, pd.DatetimeIndex):
        timestamps = df.index
    elif "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"])
    else:
        timestamps = pd.date_range("1970-01-01", periods=len(df), freq="D")

    labels_list = labeler.compute_labels(
        prices=df,
        timestamps=timestamps,
        symbol="TRAIN_DF",
    )

    target_value = 1 if side.lower() == "long" else -1

    # Map generated labels back to df index. Default is 0.
    # Map label_value of 1 to 1, and -1 / 0 to 0.
    out = pd.Series(0, index=df.index, dtype=int)
    if isinstance(df.index, pd.DatetimeIndex):
        for lbl in labels_list:
            if lbl.event_time in out.index:
                out.loc[lbl.event_time] = 1 if lbl.label_value == target_value else 0
    elif "timestamp" in df.columns:
        time_to_idx = {pd.to_datetime(t): idx for idx, t in enumerate(df["timestamp"])}
        for lbl in labels_list:
            ts = pd.to_datetime(lbl.event_time)
            if ts in time_to_idx:
                out.iloc[time_to_idx[ts]] = 1 if lbl.label_value == target_value else 0
    else:
        for i, lbl in enumerate(labels_list):
            if i < len(out):
                out.iloc[i] = 1 if lbl.label_value == target_value else 0

    return out


# ---------------------------------------------------------------------------
# BaseLogistic
# ---------------------------------------------------------------------------

class BaseLogistic:
    """
    Layer 2: Logistic Regression model for any timeframe.

    Training
    --------
    model = BaseLogistic()
    model.train(X_train, y_train, feature_cols, save_path="models/logreg_longterm_v1.joblib")

    Live inference
    --------------
    model = BaseLogistic()
    model.load("models/logreg_longterm_v1.joblib")
    prob = model.predict_proba(X_live)   # shape (n,)
    """

    def __init__(
        self,
        n_splits: int = 5,
        use_pca: bool = False,
        n_components: int = 5,
    ):
        self.n_splits     = n_splits
        self.use_pca      = use_pca
        self.n_components = n_components
        self.feature_names: list[str] = []
        self.pipeline: Pipeline | None = None
        self._is_fitted: bool = False

    # ── training ────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_cols: list[str],
        save_path: str | None = None,
    ) -> dict[str, float]:
        """
        Fit on X_train / y_train with TimeSeriesSplit CV.

        Returns:
            dict with "mean_cv_acc" and "final_train_acc" keys.
        """
        self.feature_names = feature_cols
        cols_to_keep = self.feature_names + (["__date__"] if "__date__" in X_train.columns else [])
        X = X_train[cols_to_keep].copy()

        steps: list[tuple] = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]
        if self.use_pca:
            n = min(self.n_components, len(feature_cols))
            steps.append(("pca", PCA(n_components=n)))

        steps.append((
            "classifier",
            LogisticRegression(
                l1_ratio=0,      # equivalent to penalty='l2'; avoids FutureWarning in sklearn ≥1.8
                C=1.0,
                max_iter=2000,
                solver="lbfgs",
                random_state=42,
                class_weight="balanced",  # handles class imbalance in win/loss labels
            ),
        ))

        pipeline = Pipeline(steps)

        # Determine purging horizon (V) based on the features/timeframe
        from data_platform.features.canonical_builder import (
            INTRADAY_FEATURES,
            SWING_FEATURES,
        )
        if set(self.feature_names).issubset(set(INTRADAY_FEATURES)):
            v_barrier = 10
        elif set(self.feature_names).issubset(set(SWING_FEATURES)):
            v_barrier = 15
        else:
            v_barrier = 12

        # TimeSeriesSplit cross-validation (no peeking at future folds with purging)
        tscv   = TimeSeriesSplit(n_splits=self.n_splits)
        cv_accs: list[float] = []

        dates = pd.to_datetime(X["__date__"]) if "__date__" in X.columns else None
        X_clean = X.drop(columns=["__date__"], errors="ignore")

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_clean)):
            if dates is not None and len(val_idx) > 0:
                val_start_date = dates.iloc[val_idx[0]]
                if set(self.feature_names).issubset(set(INTRADAY_FEATURES)):
                    delta = pd.Timedelta(minutes=v_barrier)
                elif set(self.feature_names).issubset(set(SWING_FEATURES)):
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

            X_tr, X_val = X_clean.iloc[train_idx_purged], X_clean.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx_purged], y_train.iloc[val_idx]

            # Check if training fold has at least 2 classes
            unique_classes = np.unique(y_tr)
            if len(unique_classes) <= 1:
                logger.warning(f"  Fold {fold_idx + 1}/{self.n_splits}: only 1 class {unique_classes} in training fold. Skipping fit, using prior.")
                val_unique = np.unique(y_val)
                acc = 1.0 if len(val_unique) == 1 and val_unique[0] == unique_classes[0] else 0.5
                cv_accs.append(acc)
                continue

            pipeline.fit(X_tr, y_tr)
            acc = float((pipeline.predict(X_val) == y_val).mean())
            cv_accs.append(acc)
            logger.info(f"  Fold {fold_idx + 1}/{self.n_splits} (purged {len(train_idx) - len(train_idx_purged)}): val_acc={acc:.4f}")

        mean_cv_acc = float(np.mean(cv_accs)) if cv_accs else 0.5
        logger.info(f"Mean CV accuracy: {mean_cv_acc:.4f}")

        # Final fit on full training data
        unique_classes = np.unique(y_train)
        if len(unique_classes) <= 1:
            logger.warning(f"Only 1 class {unique_classes} present in final training data. Using DummyClassifier fallback.")
            from sklearn.dummy import DummyClassifier
            dummy_steps = [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
            ]
            if self.use_pca:
                n = min(self.n_components, len(self.feature_names))
                dummy_steps.append(("pca", PCA(n_components=n)))
            dummy_steps.append(("classifier", DummyClassifier(strategy="prior")))
            pipeline = Pipeline(dummy_steps)

        pipeline.fit(X_clean, y_train)
        train_acc = float((pipeline.predict(X_clean) == y_train).mean())
        logger.info(f"Final train accuracy: {train_acc:.4f}")

        self.pipeline    = pipeline
        self._is_fitted  = True

        if save_path:
            self._save(save_path)

        return {"mean_cv_acc": mean_cv_acc, "final_train_acc": train_acc}

    # ── persistence ─────────────────────────────────────────────────────────

    def _save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "pipeline":      self.pipeline,
            "feature_names": self.feature_names,
            "n_splits":      self.n_splits,
            "use_pca":       self.use_pca,
            "n_components":  self.n_components,
        }
        joblib.dump(payload, path)
        logger.info(f"BaseLogistic saved → {path}")

    def load(self, load_path: str) -> None:
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
        payload = joblib.load(load_path)
        if isinstance(payload, Pipeline):
            # Legacy: older saves stored just the pipeline
            self.pipeline = payload
            logger.warning("Loaded legacy pipeline-only format. feature_names not restored.")
        else:
            self.pipeline      = payload["pipeline"]
            self.feature_names = payload.get("feature_names", [])
            self.n_splits      = payload.get("n_splits", self.n_splits)
            self.use_pca       = payload.get("use_pca", self.use_pca)
            self.n_components  = payload.get("n_components", self.n_components)
        self._is_fitted = True
        logger.info(f"BaseLogistic loaded from {load_path}")

    # ── inference ───────────────────────────────────────────────────────────

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Returns P(win) for each row — shape (n,), values in [0, 1].
        Raises RuntimeError if called before train() or load().
        """
        if not self._is_fitted or self.pipeline is None:
            raise RuntimeError(
                "BaseLogistic has not been trained or loaded. "
                "Call .train() or .load() before .predict_proba()."
            )
        cols = self.feature_names if self.feature_names else list(X_test.columns)
        return self.pipeline.predict_proba(X_test[cols])[:, 1]

    def is_ready(self) -> bool:
        return self._is_fitted and self.pipeline is not None


# ---------------------------------------------------------------------------
# EnsembleModel
# ---------------------------------------------------------------------------

class EnsembleModel:
    """
    Ensemble of three classifiers for Research OS v2:
      - logistic        (LR  — linear baselines, fast to update)
      - random_forest   (RF  — structural breaks, non-linear)
      - gradient_boosting (GB — sequential interactions)

    Weighted average of predict_proba outputs.
    """

    def __init__(
        self,
        feature_cols: list[str],
        weights: dict[str, float] | None = None,
    ):
        self.feature_cols = feature_cols
        self.weights = weights or {
            "logistic":          0.4,
            "random_forest":     0.3,
            "gradient_boosting": 0.3,
        }
        self._is_fitted: bool = False

        self.models: dict[str, Pipeline] = {
            "logistic": Pipeline([
                ("imputer",    SimpleImputer(strategy="median")),
                ("scaler",     StandardScaler()),
                ("classifier", LogisticRegression(
                    l1_ratio=0, C=1.0, max_iter=2000,
                    solver="lbfgs", random_state=42, class_weight="balanced",
                )),
            ]),
            "random_forest": Pipeline([
                ("imputer",    SimpleImputer(strategy="median")),
                ("scaler",     StandardScaler()),
                ("classifier", RandomForestClassifier(
                    n_estimators=200, max_depth=6,
                    min_samples_leaf=20, random_state=42,
                    class_weight="balanced", n_jobs=-1,
                )),
            ]),
            "gradient_boosting": Pipeline([
                ("imputer",    SimpleImputer(strategy="median")),
                ("scaler",     StandardScaler()),
                ("classifier", GradientBoostingClassifier(
                    n_estimators=150, max_depth=4,
                    learning_rate=0.05, subsample=0.8,
                    random_state=42,
                )),
            ]),
        }

    # ── training ────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> dict[str, float]:
        """Fit all sub-models. Returns per-model train accuracy."""
        X = X_train[self.feature_cols]
        accs: dict[str, float] = {}
        for name, model in self.models.items():
            try:
                model.fit(X, y_train)
                acc = float((model.predict(X) == y_train).mean())
                accs[name] = acc
                logger.info(f"  {name}: train_acc={acc:.4f}")
            except Exception as exc:
                logger.error(f"EnsembleModel: {name} fit failed: {exc}")
                accs[name] = 0.0
        self._is_fitted = True
        return accs

    # ── inference ───────────────────────────────────────────────────────────

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(
                "EnsembleModel has not been trained. Call .train() or .load() first."
            )
        X       = X_test[self.feature_cols]
        preds   = []
        wts     = []
        for name, model in self.models.items():
            try:
                p  = model.predict_proba(X)[:, 1]
                w  = self.weights.get(name, 0.0)
                preds.append(p * w)
                wts.append(w)
            except Exception as exc:
                logger.error(f"EnsembleModel: {name} predict_proba failed: {exc}")

        if not preds:
            logger.error("All ensemble sub-models failed — returning 0.5")
            return np.full(len(X_test), 0.5)

        total_w = sum(wts) or 1.0
        return np.sum(preds, axis=0) / total_w

    def is_ready(self) -> bool:
        return self._is_fitted

    # ── persistence ─────────────────────────────────────────────────────────

    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(dir_path, f"{name}.joblib"))
        meta = {
            "feature_cols": self.feature_cols,
            "weights":      self.weights,
        }
        with open(os.path.join(dir_path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"EnsembleModel saved → {dir_path}")

    def load(self, dir_path: str) -> None:
        meta_path = os.path.join(dir_path, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self.feature_cols = meta.get("feature_cols", self.feature_cols)
            self.weights      = meta.get("weights",      self.weights)

        for name in list(self.models.keys()):
            p = os.path.join(dir_path, f"{name}.joblib")
            if os.path.exists(p):
                self.models[name] = joblib.load(p)
            else:
                logger.warning(f"EnsembleModel: sub-model file missing: {p}")

        self._is_fitted = True
        logger.info(f"EnsembleModel loaded from {dir_path}")


# ---------------------------------------------------------------------------
# ModelRegistry  — singleton that wires trained models into the live path
# ---------------------------------------------------------------------------

class ModelRegistry:
    """
    Singleton model cache.  Loads each model exactly once per process.

    Usage in generate_live_predictions.py
    --------------------------------------
    from prediction_intelligence.base_logistic import ModelRegistry

    registry = ModelRegistry()          # or ModelRegistry(model_dir="/custom/path")

    # In generate_longterm_predictions():
    model   = registry.get("LOGREG_LONGTERM_v1", "LONGTERM")
    prob    = model.predict_proba(X_live_row)[0]
    """

    _instance: "ModelRegistry | None" = None

    # ── disk path aliases ────────────────────────────────────────────────────
    # Maps canonical model_version key → actual directory/file name on disk.
    # Needed when training scripts saved under a different name than the registry key.
    _DISK_ALIASES: dict[str, str] = {
        "ENSEMBLE_SWING_v1": "XGB_SWING_v1",
    }

    def __new__(cls, model_dir: str | None = None) -> "ModelRegistry":
        if cls._instance is None:
            obj             = super().__new__(cls)
            obj._model_dir  = model_dir or _DEFAULT_MODEL_DIR
            obj._cache: dict[str, BaseLogistic | EnsembleModel] = {}
            obj._imputers: dict[str, Any] = {}
            cls._instance   = obj
        return cls._instance

    # ── path conventions ────────────────────────────────────────────────────
    # LOGREG_*   → BaseLogistic  saved as  <model_dir>/<name>.joblib
    # ENSEMBLE_* → EnsembleModel saved as  <model_dir>/<name>/   (directory)

    def get(
        self,
        model_version: str,
        timeframe: str,
    ) -> BaseLogistic | EnsembleModel:
        """
        Return a fitted model for (model_version, timeframe).

        If the model file exists on disk it is loaded and cached.
        If not, a new BaseLogistic is returned unfitted — callers must
        check .is_ready() before calling .predict_proba().
        """
        key = f"{model_version}:{timeframe}"
        if key in self._cache:
            return self._cache[key]

        model: BaseLogistic | EnsembleModel | MetaEnsemble

        # Resolve disk name (handles legacy save paths)
        disk_name = self._DISK_ALIASES.get(model_version, model_version)

        if model_version.startswith("META"):
            from prediction_intelligence.meta_ensemble import MetaEnsemble
            model = MetaEnsemble(timeframe=timeframe, model_dir=self._model_dir)
            suffix = "_long" if "_long" in model_version else "_short" if "_short" in model_version else ""
            dir_p = os.path.join(self._model_dir, f"meta_ensemble_{timeframe.lower()}{suffix}")
            if os.path.isdir(dir_p):
                model.load(dir_p)
            else:
                logger.warning(
                    f"ModelRegistry: MetaEnsemble dir not found: {dir_p} — model NOT loaded"
                )
        elif model_version.startswith("ENSEMBLE"):
            model = EnsembleModel(feature_cols=self._feature_cols(timeframe))
            dir_p = os.path.join(self._model_dir, disk_name)
            if os.path.isdir(dir_p):
                model.load(dir_p)
            else:
                logger.warning(
                    f"ModelRegistry: EnsembleModel dir not found: {dir_p} — model NOT loaded"
                )
        else:
            model = BaseLogistic()
            file_p = os.path.join(self._model_dir, f"{disk_name}.joblib")
            if os.path.exists(file_p):
                model.load(file_p)
            else:
                logger.warning(
                    f"ModelRegistry: model file not found: {file_p} — model NOT loaded. "
                    "Run the training script first."
                )

        self._cache[key] = model
        return model

    def register(
        self,
        model_version: str,
        timeframe: str,
        model: BaseLogistic | EnsembleModel,
    ) -> None:
        """Manually register an already-fitted model (e.g. after inline training)."""
        self._cache[f"{model_version}:{timeframe}"] = model

    def is_ready(self, model_version: str, timeframe: str) -> bool:
        m = self._cache.get(f"{model_version}:{timeframe}")
        if m is None:
            return False
        return m.is_ready()

    @staticmethod
    def _feature_cols(timeframe: str) -> list[str]:
        tf = timeframe.upper()
        if tf == "INTRADAY":
            return INTRADAY_FEATURES
        if tf == "SWING":
            return SWING_FEATURES
        return LONGTERM_FEATURES

    def save_imputer(self, timeframe: str, imputer: Any) -> str:
        os.makedirs(self._model_dir, exist_ok=True)
        path = os.path.join(self._model_dir, f"imputer_{timeframe.lower()}.joblib")
        joblib.dump(imputer, path)
        self._imputers[timeframe.upper()] = imputer
        logger.info(f"Imputer for {timeframe} saved -> {path}")
        return path

    def get_imputer(self, timeframe: str) -> Any | None:
        tf = timeframe.upper()
        if tf in self._imputers:
            return self._imputers[tf]
        path = os.path.join(self._model_dir, f"imputer_{tf.lower()}.joblib")
        if os.path.exists(path):
            imputer = joblib.load(path)
            self._imputers[tf] = imputer
            logger.info(f"Imputer for {tf} loaded from {path}")
            return imputer
        return None

    def purge(self) -> None:
        """Clear cache — useful in tests."""
        self._cache.clear()
        if hasattr(self, "_imputers"):
            self._imputers.clear()
        ModelRegistry._instance = None