"""
validation/prediction_store.py
================================
Single write path for all predictions.

Rules:
  1. Every write must pass through PredictionRecord validation first.
  2. Writes are idempotent: if (symbol, prediction_time, model_version) already
     exists the second write is a no-op (skip, not overwrite).
  3. Outcomes are written separately via resolve(), which only updates mutable
     columns (actual_outcome, exit_price, exit_time, actual_return, mfe, mae,
     hold_bars, target_hit, stop_hit, is_correct).

Usage
-----
    from validation.prediction_store import PredictionStore
    from validation.prediction_record import PredictionRecord

    store = PredictionStore()

    # Write a new prediction
    pred_id = store.store(record, db)

    # Later: resolve the outcome
    store.resolve(pred_id, outcome_fields={
        "actual_outcome": "WIN",
        "exit_price": 2450.0,
        "exit_time": now_ist(),
        "actual_return": 0.031,
        "mfe": 0.034,
        "mae": 0.008,
        "hold_bars": 5,
        "target_hit": True,
        "stop_hit": False,
        "is_correct": True,
    }, db=db)
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database.models import Prediction
from utils.logger import get_logger
from validation.prediction_record import Outcome, PredictionRecord

logger = get_logger("prediction_store")

# Columns that the resolver is allowed to update.
_RESOLVABLE_COLUMNS = {
    "actual_outcome",
    "exit_price",
    "exit_time",
    "actual_return",
    "mfe",
    "mae",
    "hold_bars",
    "target_hit",
    "stop_hit",
    "is_correct",
    "updated_at",
}


class PredictionStore:
    """
    The only class that writes predictions to the database.

    All public methods accept a SQLAlchemy Session so callers control
    transaction scope (important: caller must call db.commit()).
    """

    # ── write ─────────────────────────────────────────────────────────────────

    def store(self, record: PredictionRecord, db: Session) -> Optional[str]:
        """
        Persist a validated PredictionRecord.

        Returns:
            The prediction_id (UUID string) if written, None if it was a duplicate.

        Raises:
            ValueError: if the record fails validation (should be impossible if
                        PredictionRecord is constructed correctly, but defensive).
        """
        kwargs = record.to_orm_kwargs()

        # Guard: check for duplicate before attempting insert
        existing = (
            db.query(Prediction.id)
            .filter(
                Prediction.symbol == record.symbol,
                Prediction.prediction_time == record.prediction_time,
                Prediction.model_version == record.model_version,
            )
            .first()
        )
        if existing:
            logger.debug(
                "Skipping duplicate prediction: %s / %s / %s (id=%s already exists)",
                record.symbol, record.timeframe, record.model_version, existing[0],
            )
            return None

        pred = Prediction(**kwargs)
        try:
            db.add(pred)
            db.flush()   # write to DB but let caller commit
            logger.info(
                "Stored prediction %s | %s [%s] dir=%s entry=%.2f sl=%.2f tp=%.2f prob=%.3f",
                pred.id, record.symbol, record.timeframe, record.direction,
                record.entry_price, record.stop_loss, record.target_price,
                record.win_probability,
            )
            return pred.id

        except IntegrityError:
            db.rollback()
            logger.warning(
                "IntegrityError on insert for %s / %s / %s — treating as duplicate, skipping.",
                record.symbol, record.prediction_time, record.model_version,
            )
            return None

        except Exception as exc:
            db.rollback()
            logger.error("Failed to store prediction for %s: %s", record.symbol, exc, exc_info=True)
            raise

    # ── resolve ───────────────────────────────────────────────────────────────

    def resolve(
        self,
        prediction_id: str,
        outcome_fields: Dict[str, Any],
        db: Session,
    ) -> bool:
        """
        Write outcome fields back to a prediction row.

        Only columns in _RESOLVABLE_COLUMNS are allowed — this prevents
        accidental overwrite of immutable fields like entry_price or model_version.

        Returns True if the row was found and updated, False otherwise.
        """
        pred = db.query(Prediction).filter(Prediction.id == prediction_id).first()
        if pred is None:
            logger.error("resolve() called for unknown prediction_id=%s", prediction_id)
            return False

        if pred.actual_outcome != "OPEN":
            logger.warning(
                "Prediction %s is already resolved (%s). Skipping re-resolution.",
                prediction_id, pred.actual_outcome,
            )
            return False

        # Apply only allowed columns
        unknown = set(outcome_fields) - _RESOLVABLE_COLUMNS
        if unknown:
            raise ValueError(
                f"resolve() received unknown/immutable columns: {unknown}. "
                f"Only {_RESOLVABLE_COLUMNS} are permitted."
            )

        for col, val in outcome_fields.items():
            setattr(pred, col, val)

        # Auto-set updated_at if caller didn't include it
        if "updated_at" not in outcome_fields:
            from utils.time_utils import now_ist
            pred.updated_at = now_ist()

        logger.info(
            "Resolved prediction %s | %s [%s] outcome=%s return=%.3f%% MFE=%.3f%% MAE=%.3f%%",
            prediction_id,
            pred.symbol,
            pred.horizon,
            outcome_fields.get("actual_outcome", "?"),
            (outcome_fields.get("actual_return") or 0) * 100,
            (outcome_fields.get("mfe") or 0) * 100,
            (outcome_fields.get("mae") or 0) * 100,
        )
        return True

    # ── read ──────────────────────────────────────────────────────────────────

    def get_open(
        self,
        db: Session,
        timeframe: Optional[str] = None,
    ) -> List[Prediction]:
        """Return all OPEN (unresolved) predictions, optionally filtered by timeframe."""
        q = db.query(Prediction).filter(Prediction.actual_outcome == "OPEN")
        if timeframe:
            q = q.filter(Prediction.horizon == timeframe.upper())
        return q.order_by(Prediction.prediction_time).all()

    def get_resolved(
        self,
        db: Session,
        timeframe: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Prediction]:
        """Return resolved predictions (WIN/LOSS/TIMEOUT), newest first."""
        q = db.query(Prediction).filter(
            Prediction.actual_outcome.in_(["WIN", "LOSS", "TIMEOUT"])
        )
        if timeframe:
            q = q.filter(Prediction.horizon == timeframe.upper())
        return q.order_by(Prediction.prediction_time.desc()).limit(limit).all()

    def get_feature_snapshot(self, prediction_id: str, db: Session) -> Optional[Dict]:
        """Return the feature snapshot dict for a given prediction, or None if not stored."""
        pred = db.query(Prediction).filter(Prediction.id == prediction_id).first()
        if pred is None or not pred.feature_snapshot:
            return None
        try:
            return json.loads(pred.feature_snapshot)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Could not parse feature_snapshot for prediction %s", prediction_id)
            return None
