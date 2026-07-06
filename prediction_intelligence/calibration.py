"""
Probability Calibration

Provides calibration functions for model win probabilities.
Implements isotonic and Platt scaling calibration for binary outcomes.

This module replaces the deleted QuantResearchOS.ml.inference.calibration.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

from utils.logger import get_logger

logger = get_logger("calibration")


import os
import joblib
from pathlib import Path

MODEL_DIR = os.getenv("MODEL_DIR", "models/saved")
_CALIBRATOR_CACHE: dict[str, CalibratedClassifierCV | IsotonicRegression] = {}


def calibrate_or_passthrough(win_prob: float, timeframe: str, direction: str | None = None) -> float:
    """
    Calibrate win probability if a calibrator exists, otherwise pass through.

    For cold start (no calibrator fitted yet), returns raw win_prob.
    Calibrators are fitted after 100-500 resolved predictions via resolve_outcomes.py.

    Args:
        win_prob: Raw model win probability (0-1)
        timeframe: Timeframe key (INTRADAY, SWING, LONGTERM)
        direction: Trade direction (BUY, SELL)

    Returns:
        Calibrated win probability, or raw if no calibrator exists
    """
    direction_suffix = f"_{direction.lower()}" if direction else ""
    calibrator_key = f"{timeframe}{direction_suffix}_calibrator"
    calibrator = _CALIBRATOR_CACHE.get(calibrator_key)

    if calibrator is None:
        # Attempt to load from disk (direction-specific first, fallback to generic)
        path = Path(MODEL_DIR) / f"calibrator_{timeframe.lower()}{direction_suffix}.joblib"
        if not path.exists() and direction:
            path = Path(MODEL_DIR) / f"calibrator_{timeframe.lower()}.joblib"

        if path.exists():
            try:
                calibrator = joblib.load(path)
                _CALIBRATOR_CACHE[calibrator_key] = calibrator
                logger.info(f"Loaded calibrator for {timeframe} ({direction or 'generic'}) from disk: {path}")
            except Exception as e:
                logger.error(f"Failed to load calibrator from {path}: {e}")

    if calibrator is None:
        # Cold start: no calibrator fitted yet
        return win_prob

    try:
        # Calibrate using the fitted calibrator
        calibrated = calibrator.predict(np.array([[win_prob]]))[0]
        return float(np.clip(calibrated, 0.0, 1.0))
    except Exception as e:
        logger.error(f"Calibration failed for {timeframe} ({direction or 'generic'}): {e}, using raw probability")
        return win_prob


def fit_calibrator(
    raw_probs: list[float],
    outcomes: list[int],  # 1 for WIN, 0 for LOSS
    timeframe: str,
    direction: str | None = None,
    method: str = "isotonic",
) -> None:
    """
    Fit a probability calibrator using historical predictions and outcomes.

    Args:
        raw_probs: List of raw model win probabilities
        outcomes: List of actual outcomes (1=WIN, 0=LOSS)
        timeframe: Timeframe key (INTRADAY, SWING, LONGTERM)
        direction: Trade direction (BUY, SELL)
        method: Calibration method ('isotonic' or 'platt')
    """
    if len(raw_probs) < 50:
        logger.warning(f"Insufficient data for calibration fitting: {len(raw_probs)} < 50")
        return

    X = np.array(raw_probs).reshape(-1, 1)
    y = np.array(outcomes)

    try:
        if method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(X.ravel(), y)
        else:  # Platt (sigmoid)
            from sklearn.linear_model import LogisticRegression
            dummy_model = LogisticRegression()
            dummy_model.fit(X, y)
            calibrator = CalibratedClassifierCV(dummy_model, method="sigmoid", cv="prefit")
            calibrator.fit(X, y)

        direction_suffix = f"_{direction.lower()}" if direction else ""
        calibrator_key = f"{timeframe}{direction_suffix}_calibrator"
        _CALIBRATOR_CACHE[calibrator_key] = calibrator

        # Save to disk
        path = Path(MODEL_DIR) / f"calibrator_{timeframe.lower()}{direction_suffix}.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump(calibrator, path)
            logger.info(f"Saved calibrator for {timeframe} ({direction or 'generic'}) to disk: {path}")
        except Exception as e:
            logger.error(f"Failed to save calibrator to {path}: {e}")

        logger.info(f"Fitted {method} calibrator for {timeframe} ({direction or 'generic'}) on {len(raw_probs)} samples")

    except Exception as e:
        logger.error(f"Failed to fit calibrator for {timeframe} ({direction or 'generic'}): {e}")


def has_calibrator(timeframe: str, direction: str | None = None) -> bool:
    """Check if a calibrator exists for the given timeframe/direction."""
    direction_suffix = f"_{direction.lower()}" if direction else ""
    if f"{timeframe}{direction_suffix}_calibrator" in _CALIBRATOR_CACHE:
        return True
    path = Path(MODEL_DIR) / f"calibrator_{timeframe.lower()}{direction_suffix}.joblib"
    if path.exists():
        return True
    if direction:
        return (Path(MODEL_DIR) / f"calibrator_{timeframe.lower()}.joblib").exists()
    return False
