"""
validation/prediction_record.py
================================
Strict Pydantic data contract for predictions.

Every write path in the pipeline must construct a PredictionRecord before
touching the database.  This guarantees that:
  - Entry/SL/target are self-consistent for the declared direction
  - Probability is in [0, 1]
  - The feature snapshot is always populated (nothing gets lost)
  - Timestamps are always IST-aware

Outcome fields are all Optional and populated later by the resolver.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Direction / timeframe / outcome literals
# ---------------------------------------------------------------------------

Direction = Literal["BUY", "SELL"]
Timeframe = Literal["INTRADAY", "SWING", "LONGTERM"]
Outcome   = Literal["WIN", "LOSS", "TIMEOUT", "OPEN"]


# ---------------------------------------------------------------------------
# PredictionRecord
# ---------------------------------------------------------------------------

class PredictionRecord(BaseModel):
    """
    Immutable at-prediction snapshot.

    Required at creation:
        symbol, prediction_time, feature_snapshot, feature_schema_version,
        model_version, direction, timeframe, win_probability, confidence,
        entry_price, stop_loss, target_price

    Populated by the resolver (all Optional until resolved):
        actual_outcome, exit_price, exit_time, actual_return,
        mfe, mae, hold_bars, target_hit, stop_hit
    """

    # ── identity ─────────────────────────────────────────────────────────────
    prediction_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="UUID assigned at creation; becomes the database primary key.",
    )
    symbol: str
    prediction_time: datetime = Field(
        description="IST-aware timestamp of when the signal was emitted.",
    )

    # ── model provenance ─────────────────────────────────────────────────────
    model_version: str
    feature_schema_version: str = Field(
        description="Tag for the canonical feature set (e.g. 'v2.1'). "
                    "Lets you reconstruct which features a model used.",
    )

    # ── feature snapshot ─────────────────────────────────────────────────────
    feature_snapshot: Dict[str, Any] = Field(
        description="Every feature value seen by the model at prediction time. "
                    "Required — this is the audit trail.",
    )

    # ── signal ───────────────────────────────────────────────────────────────
    direction: Direction
    timeframe: Timeframe
    win_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    expected_return: Optional[float] = None
    regime: Optional[str] = None
    reason: Optional[str] = None
    latency_ms: Optional[int] = None

    # ── levels ───────────────────────────────────────────────────────────────
    entry_price: float = Field(gt=0)
    stop_loss: float   = Field(gt=0)
    target_price: float = Field(gt=0)
    expiry_time: Optional[datetime] = None

    # ── outcome (all None until resolved) ────────────────────────────────────
    actual_outcome: Outcome = "OPEN"
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    actual_return: Optional[float] = None   # fractional, e.g. 0.023 = +2.3%
    mfe: Optional[float] = None             # max favorable excursion (fraction)
    mae: Optional[float] = None             # max adverse excursion (fraction)
    hold_bars: Optional[int] = None         # number of bars from entry to exit
    target_hit: bool = False
    stop_hit: bool = False
    is_correct: Optional[bool] = None

    # ── validators ───────────────────────────────────────────────────────────

    @field_validator("prediction_time", "exit_time", "expiry_time", mode="before")
    @classmethod
    def _coerce_datetime(cls, v: Any) -> Optional[datetime]:
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        return datetime.fromisoformat(str(v))

    @field_validator("feature_snapshot")
    @classmethod
    def _snapshot_not_empty(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if not v:
            raise ValueError(
                "feature_snapshot must be a non-empty dict. "
                "Every feature value must be recorded with the prediction."
            )
        return v

    @field_validator("prediction_time")
    @classmethod
    def _prediction_time_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError(
                f"prediction_time must be timezone-aware (use now_ist()). "
                f"Got naive datetime: {v!r}"
            )
        return v

    @model_validator(mode="after")
    def _validate_levels(self) -> "PredictionRecord":
        """SL and target must be on the correct side of entry for the direction."""
        entry = self.entry_price
        sl    = self.stop_loss
        tp    = self.target_price

        if self.direction == "BUY":
            if sl >= entry:
                raise ValueError(
                    f"BUY prediction has stop_loss ({sl:.2f}) >= entry_price ({entry:.2f}). "
                    "Stop-loss must be below entry for a long trade."
                )
            if tp <= entry:
                raise ValueError(
                    f"BUY prediction has target_price ({tp:.2f}) <= entry_price ({entry:.2f}). "
                    "Target must be above entry for a long trade."
                )
        else:  # SELL
            if sl <= entry:
                raise ValueError(
                    f"SELL prediction has stop_loss ({sl:.2f}) <= entry_price ({entry:.2f}). "
                    "Stop-loss must be above entry for a short trade."
                )
            if tp >= entry:
                raise ValueError(
                    f"SELL prediction has target_price ({tp:.2f}) >= entry_price ({entry:.2f}). "
                    "Target must be below entry for a short trade."
                )
        return self

    # ── helpers ──────────────────────────────────────────────────────────────

    def feature_snapshot_json(self) -> str:
        """Return the feature snapshot serialised as a compact JSON string."""
        return json.dumps(self.feature_snapshot, separators=(",", ":"))

    def risk_reward_ratio(self) -> float:
        """Reward / Risk ratio from the declared levels."""
        risk   = abs(self.entry_price - self.stop_loss)
        reward = abs(self.target_price - self.entry_price)
        return reward / risk if risk > 0 else 0.0

    def is_resolved(self) -> bool:
        return self.actual_outcome != "OPEN"

    def to_orm_kwargs(self) -> dict:
        """
        Return a dict suitable for constructing / updating a Prediction ORM object.
        Maps PredictionRecord field names to ORM column names.
        """
        return {
            "id": self.prediction_id,
            "symbol": self.symbol,
            "model_version": self.model_version,
            "feature_schema_version": self.feature_schema_version,
            "feature_snapshot": self.feature_snapshot_json(),
            "features_used": self.feature_snapshot_json(),   # legacy compat
            "prediction": self.direction,
            "horizon": self.timeframe,
            "confidence": self.confidence,
            "expected_return": self.expected_return,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target_price": self.target_price,
            "prediction_time": self.prediction_time,
            "expiry_time": self.expiry_time,
            "regime": self.regime,
            "reason": self.reason,
            "latency_ms": self.latency_ms,
            "actual_outcome": self.actual_outcome,
            "target_hit": self.target_hit,
            "stop_hit": self.stop_hit,
            "actual_return": self.actual_return,
            "mfe": self.mfe,
            "mae": self.mae,
            "exit_time": self.exit_time,
            "hold_bars": self.hold_bars,
            "is_correct": self.is_correct,
        }
