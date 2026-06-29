"""
Base Validator Framework

Provides the abstract base classes and data structures used by all
domain-specific validators (equity, options, corporate actions).

Classes exported:
  - ValidationSeverity   (enum)
  - ValidationResult     (single rule outcome)
  - ValidationReport     (collection of results with scoring)
  - BaseValidator        (abstract validator with shared helpers)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from utils.logger import get_logger

logger = get_logger("base_validator")


class ValidationSeverity(str, Enum):
    """How severe a validation failure is."""

    CRITICAL = "critical"  # must fix before pipeline continues
    WARNING = "warning"  # data usable but suspicious
    INFO = "info"  # informational only


@dataclass
class ValidationResult:
    """Outcome of a single validation rule."""

    rule_name: str
    passed: bool
    severity: ValidationSeverity = ValidationSeverity.WARNING
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "passed": self.passed,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class ValidationReport:
    """Aggregated report across multiple rules for one dataset."""

    dataset_name: str = ""
    total_records: int = 0
    results: list[ValidationResult] = field(default_factory=list)

    def add_result(self, result: ValidationResult) -> None:
        self.results.append(result)

    # ── queries ─────────────────────────────────────────────

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def critical_failures(self) -> list[ValidationResult]:
        return [
            r for r in self.results if not r.passed and r.severity == ValidationSeverity.CRITICAL
        ]

    def is_acceptable(self) -> bool:
        """True when there are no CRITICAL failures."""
        return len(self.critical_failures) == 0

    def calculate_score(self) -> float:
        """0.0–1.0 pass ratio across all rules."""
        if not self.results:
            return 1.0
        return self.passed_count / len(self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "total_records": self.total_records,
            "passed": self.passed_count,
            "failed": self.failed_count,
            "score": round(self.calculate_score(), 4),
            "acceptable": self.is_acceptable(),
            "results": [r.to_dict() for r in self.results],
        }


class BaseValidator(ABC):
    """
    Abstract base for all domain validators.

    Subclasses implement ``validate(df)`` and can reuse the shared
    helpers below for common checks.
    """

    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """Run all validation rules and return a report."""
        ...

    # ── reusable check helpers ──────────────────────────────

    def _canonicalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lowercase and strip column names so checks are case-insensitive."""
        df = df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        return df

    def _check_not_null(
        self,
        df: pd.DataFrame,
        columns: list[str],
    ) -> ValidationResult:
        """Verify that *columns* have no null values."""
        null_counts: dict[str, int] = {}
        for col in columns:
            if col in df.columns:
                cnt = int(df[col].isnull().sum())
                if cnt > 0:
                    null_counts[col] = cnt

        return ValidationResult(
            rule_name="not_null_check",
            passed=len(null_counts) == 0,
            severity=ValidationSeverity.CRITICAL,
            message=(
                "No null values in key columns"
                if not null_counts
                else f"Null values found: {null_counts}"
            ),
            details={"null_counts": null_counts},
        )

    def _check_no_duplicates(
        self,
        df: pd.DataFrame,
        subset: list[str],
    ) -> ValidationResult:
        """Check for duplicate rows on *subset* columns."""
        existing = [c for c in subset if c in df.columns]
        if not existing:
            return ValidationResult(
                rule_name="duplicate_check",
                passed=True,
                severity=ValidationSeverity.WARNING,
                message="No columns to check duplicates on",
            )
        dup_count = int(df.duplicated(subset=existing).sum())
        return ValidationResult(
            rule_name="duplicate_check",
            passed=dup_count == 0,
            severity=ValidationSeverity.WARNING,
            message=(
                "No duplicate rows"
                if dup_count == 0
                else f"{dup_count} duplicate rows on {existing}"
            ),
            details={"duplicate_count": dup_count, "subset": existing},
        )

    def _check_price_positive(self, df: pd.DataFrame) -> ValidationResult:
        """Ensure price columns (open/high/low/close) are > 0."""
        price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        bad: dict[str, int] = {}
        for col in price_cols:
            neg_count = int((df[col] <= 0).sum())
            if neg_count:
                bad[col] = neg_count
        return ValidationResult(
            rule_name="price_positive_check",
            passed=len(bad) == 0,
            severity=ValidationSeverity.CRITICAL,
            message=("All prices positive" if not bad else f"Non-positive prices: {bad}"),
            details={"non_positive": bad},
        )

    def _check_volume_positive(self, df: pd.DataFrame) -> ValidationResult:
        """Ensure volume is positive (greater than zero)."""
        if "volume" not in df.columns:
            return ValidationResult(
                rule_name="volume_positive_check",
                passed=True,
                severity=ValidationSeverity.INFO,
                message="No volume column present",
            )
        neg = int((df["volume"] <= 0).sum())
        return ValidationResult(
            rule_name="volume_positive_check",
            passed=neg == 0,
            severity=ValidationSeverity.WARNING,
            message=(
                "All volumes positive" if neg == 0 else f"{neg} rows with non-positive volume"
            ),
            details={"non_positive_volume_rows": neg},
        )

    def _check_date_order(
        self,
        df: pd.DataFrame,
        date_col: str,
    ) -> ValidationResult:
        """Verify that dates are in ascending order."""
        if date_col not in df.columns or df.empty:
            return ValidationResult(
                rule_name="date_order_check",
                passed=True,
                severity=ValidationSeverity.INFO,
                message=f"Column '{date_col}' not present or empty DataFrame",
            )
        dates = pd.to_datetime(df[date_col], errors="coerce")
        sorted_ok = bool(dates.is_monotonic_increasing)
        return ValidationResult(
            rule_name="date_order_check",
            passed=sorted_ok,
            severity=ValidationSeverity.WARNING,
            message=(
                "Dates in ascending order"
                if sorted_ok
                else "Date column is not monotonically increasing"
            ),
        )

    def _check_ohlc_consistency(self, df: pd.DataFrame) -> ValidationResult:
        """Check that low <= open,close <= high for each row."""
        required = {"open", "high", "low", "close"}
        if not required.issubset(set(df.columns)):
            return ValidationResult(
                rule_name="ohlc_consistency",
                passed=True,
                severity=ValidationSeverity.INFO,
                message="OHLC columns not all present, skipping",
            )
        violations = int(
            (
                (df["low"] > df["open"])
                | (df["low"] > df["close"])
                | (df["high"] < df["open"])
                | (df["high"] < df["close"])
            ).sum()
        )
        return ValidationResult(
            rule_name="ohlc_consistency",
            passed=violations == 0,
            severity=ValidationSeverity.CRITICAL,
            message=(
                "OHLC relationships valid"
                if violations == 0
                else f"{violations} rows violate low<=open/close<=high"
            ),
            details={"violation_count": violations},
        )
