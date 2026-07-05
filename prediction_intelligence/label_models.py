"""
Label Models

Defines the canonical Label and LabelType enums used throughout the system.
These are the core data structures for ML training labels with Point-In-Time (PIT) controls.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class LabelType(str, Enum):
    """Label type enumeration for different labeling methodologies."""

    TRIPLE_BARRIER_DIRECTION = "triple_barrier_direction"
    TRIPLE_BARRIER_BINARY = "triple_barrier_binary"
    FIXED_HORIZON_RETURN = "fixed_horizon_return"
    REGIME_CLASSIFICATION = "regime_classification"
    VOLATILITY_REGIME = "volatility_regime"


class Label(BaseModel):
    """
    Canonical Label model with full Point-In-Time (PIT) audit chain.

    This is the single source of truth for all training labels in the system.
    Every label must have a complete audit trail to prevent lookahead bias.
    """

    # Core identification
    symbol: str = Field(..., description="Instrument symbol (e.g., RELIANCE)")
    label_type: LabelType = Field(..., description="Labeling methodology used")
    label_value: float = Field(..., description="Label value (direction, return, class, etc.)")

    # Temporal information
    label_date: datetime = Field(..., description="Date of the label (event time)")
    horizon_days: int = Field(..., description="Prediction horizon in days")

    # PIT audit chain - critical for preventing lookahead bias
    event_time: datetime = Field(..., description="When the event occurred")
    publication_time: datetime = Field(..., description="When the label was published")
    effective_time: datetime = Field(..., description="When the label becomes effective")
    ingestion_time: datetime = Field(..., description="When the label was ingested")

    # Provenance
    source: str = Field(..., description="Data source (e.g., yfinance, NSE)")
    version: str = Field(..., description="Labeling logic version")
    ingestion_job: str = Field(..., description="Job that ingested this label")
    # CRC-based audit checksum: sha256(symbol + label_type + label_date.isoformat() + version)
    # Computed by the labeling pipeline; validated by LabelValidator to detect tampering.
    checksum: str = Field("", description="Audit checksum for tamper detection")

    # Price information (for triple-barrier and return-based labels)
    entry_price: float | None = Field(None, description="Entry price for the trade")
    target_price: float | None = Field(None, description="Target price (take-profit)")
    stop_loss_price: float | None = Field(None, description="Stop-loss price")

    # Actual outcome (filled after resolution)
    actual_return: float | None = Field(None, description="Actual realized return")
    actual_mfe: float | None = Field(None, description="Maximum favorable excursion")
    actual_mae: float | None = Field(None, description="Maximum adverse excursion")
    actual_duration_bars: int | None = Field(None, description="Actual duration in bars")

    # Additional metadata
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional label context")

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

    def __init__(self, **data: Any):
        super().__init__(**data)
        if not self.checksum:
            self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        """Compute SHA256 checksum for PIT integrity audit."""
        import hashlib
        date_str = self.label_date.isoformat() if isinstance(self.label_date, datetime) else str(self.label_date)
        raw = f"{self.symbol}{self.label_type}{date_str}{self.version}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def is_valid(self) -> bool:
        """
        Basic validation check for the label.

        Returns:
            True if label passes basic validation
        """
        if self.label_value is None:
            return False
        if self.horizon_days <= 0:
            return False
        if self.entry_price is not None and self.entry_price <= 0:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert label to dictionary for serialization."""
        return self.model_dump()
