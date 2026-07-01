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
LONGTERM_FEATURES = [
    "pe_ratio",
    "debt_to_equity",
    "ma50_slope",        # 50-week MA slope (direction)
    "rsi_14w",           # 14-period weekly RSI
    "vol_ratio",         # recent vol / long-run vol
    "price_to_52w_high", # proximity to 52-week high
    "vix",
]

SWING_FEATURES = [
    "z_score_20d",
    "nifty_pcr",
    "rsi_14d",
    "ma20_slope",
    "atr_pct",           # ATR as % of price
    "volume_ratio",      # today vol / 20d avg vol
    "vix",
]

INTRADAY_FEATURES = [
    "vwap_dist",
    "rsi_14m",
    "vol_ratio_1m",
    "range_pct",         # (high-low)/open
    "momentum_5m",       # 5-bar close momentum
    "vix",
]


# ---------------------------------------------------------------------------
# Feature builder — MUST be the single source of truth for both training
# scripts and the live path. No lookahead: all rolling indicators use
# values available at bar close only.
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    timeframe: str,
    extra: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Build a feature DataFrame from an OHLCV candle DataFrame.

    Args:
        df      : OHLCV DataFrame with columns [timestamp, open, high, low,
                  close, volume]. Must already be sorted ascending.
        timeframe: "INTRADAY" | "SWING" | "LONGTERM"
        extra   : dict of scalar features not derivable from OHLCV
                  (e.g. {"vix": 14.2, "nifty_pcr": 1.05, "pe_ratio": 22.0}).
                  Values are broadcast to all rows.

    Returns:
        DataFrame with canonical feature columns for this timeframe.
        Rows with NaN (from rolling windows) are *not* dropped here —
        callers must dropna() before passing to fit/predict.
    """
    extra = extra or {}
    tf = timeframe.upper()
    out = pd.DataFrame(index=df.index)

    close  = df["close"]
    volume = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    high   = df["high"]
    low    = df["low"]
    open_  = df["open"]

    # ── common indicators ───────────────────────────────────────────────────
    def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(window).mean()
        loss  = (-delta.clip(upper=0)).rolling(window).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    if tf == "INTRADAY":
        # VWAP distance (cumulative within session — no lookahead)
        cum_vol = volume.cumsum()
        cum_val = (close * volume).cumsum()
        vwap    = cum_val / cum_vol.replace(0, np.nan)
        out["vwap_dist"]     = (close - vwap) / vwap.replace(0, np.nan)
        out["rsi_14m"]       = _rsi(close, 14)
        vol_avg              = volume.rolling(20).mean()
        out["vol_ratio_1m"]  = volume / vol_avg.replace(0, np.nan)
        out["range_pct"]     = (high - low) / open_.replace(0, np.nan)
        out["momentum_5m"]   = close.pct_change(5)
        out["vix"]           = float(extra.get("vix", 15.0))

    elif tf == "SWING":
        ma20                 = close.rolling(20).mean()
        std20                = close.rolling(20).std()
        out["z_score_20d"]   = (close - ma20) / std20.replace(0, np.nan)
        out["rsi_14d"]       = _rsi(close, 14)
        out["ma20_slope"]    = ma20.diff(3) / ma20.shift(3).replace(0, np.nan)
        atr                  = (high - low).rolling(14).mean()
        out["atr_pct"]       = atr / close.replace(0, np.nan)
        vol_avg              = volume.rolling(20).mean()
        out["volume_ratio"]  = volume / vol_avg.replace(0, np.nan)
        out["vix"]           = float(extra.get("vix", 15.0))
        out["nifty_pcr"]     = float(extra.get("nifty_pcr", 1.0))

    elif tf == "LONGTERM":
        ma50                     = close.rolling(50).mean()
        out["ma50_slope"]        = ma50.diff(5) / ma50.shift(5).replace(0, np.nan)
        out["rsi_14w"]           = _rsi(close, 14)
        short_vol                = close.pct_change().rolling(20).std()
        long_vol                 = close.pct_change().rolling(100).std()
        out["vol_ratio"]         = short_vol / long_vol.replace(0, np.nan)
        rolling_max              = close.rolling(52).max()
        out["price_to_52w_high"] = close / rolling_max.replace(0, np.nan)
        out["pe_ratio"]          = float(extra.get("pe_ratio", 20.0))
        out["debt_to_equity"]    = float(extra.get("debt_to_equity", 0.5))
        out["vix"]               = float(extra.get("vix", 15.0))

    else:
        raise ValueError(f"Unknown timeframe: {timeframe!r}. Expected INTRADAY, SWING, or LONGTERM.")

    return out


def build_label(
    df: pd.DataFrame,
    timeframe: str,
    target_pct: float | None = None,
    sl_pct: float | None = None,
) -> pd.Series:
    """
    Triple-barrier binary label: 1 if target hit before SL, else 0.
    Uses future prices — call only during training, never in live path.

    Default barriers per timeframe:
        INTRADAY  : target +1.5%, SL -0.75%
        SWING     : target +3.0%, SL -1.5%
        LONGTERM  : target +20%, SL -10%
    """
    _defaults = {
        "INTRADAY": (0.015, 0.0075),
        "SWING":    (0.030, 0.015),
        "LONGTERM": (0.200, 0.100),
    }
    tf = timeframe.upper()
    default_tp, default_sl = _defaults.get(tf, (0.05, 0.025))
    tp = target_pct if target_pct is not None else default_tp
    sl = sl_pct    if sl_pct    is not None else default_sl

    close = df["close"]
    labels = pd.Series(0, index=df.index, dtype=int)

    for i in range(len(df) - 1):
        entry = close.iloc[i]
        tp_price = entry * (1 + tp)
        sl_price = entry * (1 - sl)

        hit_tp = hit_sl = False
        for j in range(i + 1, len(df)):
            h = df["high"].iloc[j]
            l = df["low"].iloc[j]
            if h >= tp_price:
                hit_tp = True
                break
            if l <= sl_price:
                hit_sl = True
                break

        labels.iloc[i] = 1 if hit_tp and not hit_sl else 0

    return labels


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
        X = X_train[self.feature_names].copy()

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

        # TimeSeriesSplit cross-validation (no peeking at future folds)
        tscv   = TimeSeriesSplit(n_splits=self.n_splits)
        cv_accs: list[float] = []
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            pipeline.fit(X_tr, y_tr)
            acc = float((pipeline.predict(X_val) == y_val).mean())
            cv_accs.append(acc)
            logger.info(f"  Fold {fold_idx + 1}/{self.n_splits}: val_acc={acc:.4f}")

        mean_cv_acc = float(np.mean(cv_accs))
        logger.info(f"Mean CV accuracy: {mean_cv_acc:.4f}")

        # Final fit on full training data
        pipeline.fit(X, y_train)
        train_acc = float((pipeline.predict(X) == y_train).mean())
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

        model: BaseLogistic | EnsembleModel

        # Resolve disk name (handles legacy save paths)
        disk_name = self._DISK_ALIASES.get(model_version, model_version)

        if model_version.startswith("ENSEMBLE"):
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

    def purge(self) -> None:
        """Clear cache — useful in tests."""
        self._cache.clear()
        ModelRegistry._instance = None