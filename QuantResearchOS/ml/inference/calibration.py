"""
Probability Calibration

A raw model score (e.g. LightGBM predict_proba output) is not a real-world
probability unless it is calibrated. A model can claim 80% confidence and
actually be right 55% of the time — calibration corrects this gap using a
held-out validation set.

This module calibrates on RAW PROBABILITY ARRAYS, not on a model object.
This matters because BaseLightGBM / BaseXGBoost (prediction_intelligence/)
are custom wrapper classes, not raw sklearn estimators — sklearn's
CalibratedClassifierCV(cv="prefit") requires a true sklearn-compatible
estimator and will break against these wrappers. Calibrating on the output
probability array instead works with any model that exposes predict_proba.

Contract with generate_live_predictions.py:
    Saved to:   models/saved/{timeframe}_calibrator.pkl
    Loaded by:  ProbabilityCalibrator(timeframe=...).load()
    Applied as: calibrated_prob = calibrator.calibrate(raw_win_prob)

Usage (offline, after BaseLightGBM.train() on a separate validation set):
    raw_val_proba = model.predict_proba(X_val)[:, 1]
    calibrator = ProbabilityCalibrator(timeframe="INTRADAY", method="isotonic")
    calibrator.fit(raw_val_proba, y_val)
    calibrator.save()

Usage (live, inside generate_live_predictions.py):
    calibrator = ProbabilityCalibrator(timeframe="INTRADAY").load()
    win_prob = calibrator.calibrate(raw_win_prob)
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from utils.logger import get_logger

logger = get_logger("calibration")

_DEFAULT_MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/saved"))
_VALID_TIMEFRAMES = {"INTRADAY", "SWING", "LONGTERM"}
_VALID_METHODS = {"isotonic", "sigmoid"}


class ProbabilityCalibrator:
    """
    Calibrates raw model probabilities against actual historical outcomes.

    Two methods:
      isotonic - non-parametric, monotonic step function. Needs more data
                 (500+ resolved predictions) but fits arbitrary miscalibration
                 shapes. Use once enough resolved predictions exist.
      sigmoid  - Platt scaling, a single logistic fit. Works with less data
                 (100+ resolved predictions) but only corrects monotonic
                 over/under-confidence, not S-shaped miscalibration.
    """

    def __init__(
        self,
        timeframe: str = "SWING",
        method: str = "isotonic",
        model_dir: Path | str | None = None,
    ):
        timeframe = timeframe.upper()
        if timeframe not in _VALID_TIMEFRAMES:
            raise ValueError(f"timeframe must be one of {_VALID_TIMEFRAMES}; got '{timeframe}'")
        if method not in _VALID_METHODS:
            raise ValueError(f"method must be one of {_VALID_METHODS}; got '{method}'")
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for calibration. pip install scikit-learn")

        self.timeframe = timeframe
        self.method = method
        self.model_dir = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
        self._calibrator = None
        self._n_fit_samples = 0
        self._fit_metrics: dict = {}

    @property
    def save_path(self) -> Path:
        return self.model_dir / f"{self.timeframe.lower()}_calibrator.pkl"

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, raw_proba: np.ndarray, y_true: np.ndarray) -> dict:
        """
        Fit the calibration map: raw_proba -> calibrated_proba.

        Args:
            raw_proba: 1D array of raw P(win) scores from the model,
                       on a held-out set the model never trained on.
            y_true:    1D array of actual outcomes (1 = win, 0 = loss),
                       must correspond 1:1 with raw_proba.

        Returns:
            dict with brier_before, brier_after, n_samples.

        Raises:
            ValueError if inputs are malformed or too small to calibrate reliably.
        """
        raw_proba = np.asarray(raw_proba, dtype=np.float64).ravel()
        y_true = np.asarray(y_true, dtype=np.int32).ravel()

        if len(raw_proba) != len(y_true):
            raise ValueError(f"raw_proba ({len(raw_proba)}) and y_true ({len(y_true)}) length mismatch.")
        if set(np.unique(y_true)) - {0, 1}:
            raise ValueError("y_true must be binary (0/1).")

        min_required = 500 if self.method == "isotonic" else 100
        if len(raw_proba) < min_required:
            raise ValueError(
                f"Only {len(raw_proba)} resolved samples available; "
                f"{self.method} calibration needs at least {min_required}. "
                f"Use method='sigmoid' if you have fewer than 500 resolved predictions, "
                f"or wait for more predictions to resolve via resolve_outcomes.py."
            )

        brier_before = float(np.mean((raw_proba - y_true) ** 2))

        if self.method == "isotonic":
            self._calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self._calibrator.fit(raw_proba, y_true)
            calibrated = self._calibrator.predict(raw_proba)
        else:  # sigmoid
            self._calibrator = LogisticRegression()
            self._calibrator.fit(raw_proba.reshape(-1, 1), y_true)
            calibrated = self._calibrator.predict_proba(raw_proba.reshape(-1, 1))[:, 1]

        brier_after = float(np.mean((calibrated - y_true) ** 2))
        self._n_fit_samples = len(raw_proba)

        self._fit_metrics = {
            "timeframe": self.timeframe,
            "method": self.method,
            "n_samples": self._n_fit_samples,
            "brier_before": round(brier_before, 5),
            "brier_after": round(brier_after, 5),
            "improved": brier_after < brier_before,
        }

        if not self._fit_metrics["improved"]:
            logger.warning(
                f"[{self.timeframe}] Calibration made Brier score WORSE "
                f"({brier_before:.5f} -> {brier_after:.5f}). "
                f"Model may already be well-calibrated, or sample is too small/noisy. "
                f"Inspect before deploying this calibrator."
            )
        else:
            logger.info(
                f"[{self.timeframe}] Calibrated on {self._n_fit_samples} samples. "
                f"Brier {brier_before:.5f} -> {brier_after:.5f}"
            )

        return self._fit_metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def calibrate(self, raw_proba: float | np.ndarray) -> float | np.ndarray:
        """
        Map raw model probability to calibrated probability.
        Accepts a scalar or array; returns the same shape.
        """
        if self._calibrator is None:
            raise RuntimeError(
                f"[{self.timeframe}] Calibrator not fitted/loaded. "
                "Call fit() then save(), or load() an existing calibrator."
            )

        scalar_input = np.isscalar(raw_proba)
        arr = np.atleast_1d(np.asarray(raw_proba, dtype=np.float64))

        if self.method == "isotonic":
            out = self._calibrator.predict(arr)
        else:
            out = self._calibrator.predict_proba(arr.reshape(-1, 1))[:, 1]

        out = np.clip(out, 0.0, 1.0)
        return float(out[0]) if scalar_input else out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str | None = None) -> Path:
        if self._calibrator is None:
            raise RuntimeError("No fitted calibrator to save. Call fit() first.")
        target = Path(path) if path else self.save_path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "calibrator": self._calibrator,
            "method": self.method,
            "timeframe": self.timeframe,
            "n_fit_samples": self._n_fit_samples,
            "fit_metrics": self._fit_metrics,
        }
        with open(target, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"[{self.timeframe}] Calibrator saved → {target}")
        return target

    def load(self, path: Path | str | None = None) -> "ProbabilityCalibrator":
        """
        Load a previously fitted calibrator.
        Raises FileNotFoundError if missing — caller (generate_live_predictions.py)
        should fall back to uncalibrated raw probability and log a warning,
        never crash the prediction run over a missing calibrator.
        """
        target = Path(path) if path else self.save_path
        if not target.exists():
            raise FileNotFoundError(
                f"No calibrator artifact at {target}. "
                f"Run training pipeline calibration step, or use raw probabilities "
                f"until at least 100-500 predictions have resolved."
            )
        with open(target, "rb") as f:
            payload = pickle.load(f)
        self._calibrator = payload["calibrator"]
        self.method = payload["method"]
        self.timeframe = payload["timeframe"]
        self._n_fit_samples = payload.get("n_fit_samples", 0)
        self._fit_metrics = payload.get("fit_metrics", {})
        logger.info(
            f"[{self.timeframe}] Calibrator loaded ← {target} "
            f"(n_fit_samples={self._n_fit_samples})"
        )
        return self

    def metrics(self) -> dict:
        return dict(self._fit_metrics)


def calibrate_or_passthrough(
    raw_proba: float,
    timeframe: str,
    model_dir: Path | str | None = None,
) -> float:
    """
    Convenience wrapper for the live prediction path.

    Tries to load a fitted calibrator for this timeframe and apply it.
    If no calibrator artifact exists yet (cold start — not enough resolved
    predictions), returns raw_proba unchanged and logs once.

    This is the function generate_live_predictions.py should call:
        win_prob = calibrate_or_passthrough(raw_win_prob, timeframe="INTRADAY")
    """
    try:
        calibrator = ProbabilityCalibrator(timeframe=timeframe, model_dir=model_dir)
        calibrator.load()
        return calibrator.calibrate(raw_proba)
    except FileNotFoundError:
        logger.debug(
            f"[{timeframe}] No calibrator yet — using raw probability uncalibrated. "
            f"This is expected until enough predictions have resolved."
        )
        return raw_proba
    except Exception as e:
        logger.error(f"[{timeframe}] Calibration failed unexpectedly: {e}. Using raw probability.")
        return raw_proba