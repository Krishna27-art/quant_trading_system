"""
Label Models

Defines the canonical `Label` object used across the platform for any
ML training target (triple-barrier direction labels, regression return
labels, etc).

This module exists specifically to satisfy strict Point-In-Time (PIT)
discipline: every Label carries the full causal timestamp chain
(event_time -> publication_time -> effective_time -> ingestion_time)
plus data lineage (source/version/ingestion_job/checksum) so that
utils.pit_validator and utils.label_validator can mechanically prove
a label could not have leaked future information into training.

Previously `utils/label_validator.py` imported `Label` from this module
path, but this module did not exist — meaning label validation could
never actually run (ImportError on first use). This file fixes that.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class LabelType(str, Enum):
    """Type of label being stored."""

    TRIPLE_BARRIER_DIRECTION = "triple_barrier_direction"  # -1 / 0 / 1
    FORWARD_RETURN = "forward_return"  # continuous return over horizon
    WIN_LOSS = "win_loss"  # binary win/loss vs a target/stoploss pair
    REGIME = "regime"  # market regime classification label


class Label(BaseModel):
    """
    A single point-in-time-safe training label.

    Field naming intentionally matches what utils.pit_validator.PITValidator
    and utils.label_validator.LabelValidator already check for, so both
    validators can be used against either a list[Label] or an equivalent
    pandas DataFrame built via `Label.to_frame_row()`.
    """

    # --- identity ---
    symbol: str = Field(..., description="Instrument symbol, e.g. 'RELIANCE.NS'")
    label_type: LabelType = Field(..., description="What kind of label this is")
    label_value: float = Field(..., description="Numeric label value (-1/0/1, return, etc.)")
    label_date: datetime = Field(..., description="Calendar date this label is attached to")
    horizon_days: int = Field(..., gt=0, description="Forward horizon used to compute the label")

    # --- causal / PIT timestamp chain (MANDATORY, in strict non-decreasing order) ---
    event_time: datetime = Field(
        ..., description="When the underlying market event actually occurred"
    )
    publication_time: datetime = Field(
        ..., description="When this event was publicly available/observable"
    )
    effective_time: datetime = Field(
        ..., description="When this label becomes usable as a training target"
    )
    ingestion_time: datetime = Field(
        ..., description="When our pipeline ingested/computed this label"
    )

    # --- data lineage (MANDATORY — 'where did this number come from') ---
    source: str = Field(..., description="e.g. 'nselib', 'triple_barrier_v1', 'manual_correction'")
    version: str = Field(..., description="Schema/labeling-logic version, e.g. 'tb_v1.2'")
    ingestion_job: str = Field(..., description="Identifier of the job/run that produced this row")
    checksum: str = Field(default="", description="Integrity checksum; auto-computed if blank")

    # --- optional triple-barrier context (useful, not required for validation) ---
    entry_price: float | None = None
    target_price: float | None = None
    stop_loss_price: float | None = None
    actual_return: float | None = None
    actual_mfe: float | None = None
    actual_mae: float | None = None
    actual_duration_bars: int | None = None

    @model_validator(mode="after")
    def _fill_checksum(self) -> "Label":
        if not self.checksum:
            self.checksum = self._compute_checksum()
        return self

    def _compute_checksum(self) -> str:
        """
        Deterministic checksum over the fields that define this label's
        identity and value — used to detect silent mutation/corruption
        between generation and storage.
        """
        payload = "|".join(
            [
                self.symbol,
                self.label_type.value,
                f"{self.label_value:.10f}",
                self.label_date.isoformat(),
                str(self.horizon_days),
                self.event_time.isoformat(),
                self.publication_time.isoformat(),
                self.source,
                self.version,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_frame_row(self) -> dict:
        """Flat dict representation, suitable for pd.DataFrame(list_of_these)."""
        row = self.model_dump()
        row["label_type"] = self.label_type.value
        return row

    class Config:
        use_enum_values = False