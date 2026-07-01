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


# Simple in-memory cache for calibrators (for production, use Redis/file storage)
_CALIBRATOR_CACHE: dict[str, CalibratedClassifierCV | IsotonicRegression] = {}


def calibrate_or_passthrough(win_prob: float, timeframe: str) -> float:
    """
    Calibrate win probability if a calibrator exists, otherwise pass through.

    For cold start (no calibrator fitted yet), returns raw win_prob.
    Calibrators are fitted after 100-500 resolved predictions via resolve_outcomes.py.

    Args:
        win_prob: Raw model win probability (0-1)
        timeframe: Timeframe key (INTRADAY, SWING, LONGTERM)

    Returns:
        Calibrated win probability, or raw if no calibrator exists
    """
    calibrator_key = f"{timeframe}_calibrator"
    calibrator = _CALIBRATOR_CACHE.get(calibrator_key)

    if calibrator is None:
        # Cold start: no calibrator fitted yet
        # This is expected for the first 100-500 predictions
        return win_prob

    try:
        # Calibrate using the fitted calibrator
        calibrated = calibrator.predict(np.array([[win_prob]]))[0]
        return float(np.clip(calibrated, 0.0, 1.0))
    except Exception as e:
        logger.error(f"Calibration failed for {timeframe}: {e}, using raw probability")
        return win_prob


def fit_calibrator(
    raw_probs: list[float],
    outcomes: list[int],  # 1 for WIN, 0 for LOSS
    timeframe: str,
    method: str = "isotonic",
) -> None:
    """
    Fit a probability calibrator using historical predictions and outcomes.

    Should be called by resolve_outcomes.py after accumulating sufficient data.

    Args:
        raw_probs: List of raw model win probabilities
        outcomes: List of actual outcomes (1=WIN, 0=LOSS)
        timeframe: Timeframe key (INTRADAY, SWING, LONGTERM)
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
            # For Platt, we need a base model - use a dummy model
            from sklearn.linear_model import LogisticRegression
            dummy_model = LogisticRegression()
            dummy_model.fit(X, y)
            calibrator = CalibratedClassifierCV(dummy_model, method="sigmoid", cv="prefit")
            calibrator.fit(X, y)

        calibrator_key = f"{timeframe}_calibrator"
        _CALIBRATOR_CACHE[calibrator_key] = calibrator

        logger.info(f"Fitted {method} calibrator for {timeframe} on {len(raw_probs)} samples")

    except Exception as e:
        logger.error(f"Failed to fit calibrator for {timeframe}: {e}")


def has_calibrator(timeframe: str) -> bool:
    """Check if a calibrator exists for the given timeframe."""
    return f"{timeframe}_calibrator" in _CALIBRATOR_CACHE
