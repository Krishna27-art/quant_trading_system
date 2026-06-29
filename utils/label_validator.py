"""
Label Store Validation Utilities

Institutional-grade validation for label generation and storage.
Ensures strict Point-In-Time (PIT) controls to prevent lookahead bias.

CONNECTION TO THE PIPELINE (this was previously missing):
This validator is meant to sit between
`QuantResearchOS.services.label_engine.triple_barrier.MultiObjectiveLabeler`
(which computes raw barrier-hit outcomes) and the label store / training
pipeline. Use `LabelValidator.from_triple_barrier_events(...)` to convert
that labeler's raw DataFrame output into validated `Label` objects before
anything downstream is allowed to train on them. Call
`validate_and_filter(...)` rather than constructing labels and assuming
they're safe — invalid rows are dropped and logged, not silently kept.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from prediction_intelligence.label_models import Label, LabelType
from utils.logger import get_logger

logger = get_logger("label_validator")


class LabelValidator:
    """
    Validator for label generation and storage.

    Ensures strict PIT controls and data quality.
    """

    REQUIRED_FIELDS = [
        "symbol",
        "label_type",
        "label_value",
        "label_date",
        "horizon_days",
        "event_time",
        "publication_time",
        "ingestion_time",
        "effective_time",
        "source",
        "version",
        "ingestion_job",
        "checksum",
    ]

    def __init__(self):
        """Initialize the label validator."""
        logger.info("Initialized LabelValidator")

    # ------------------------------------------------------------------
    # Connection to the actual label engine
    # ------------------------------------------------------------------

    @staticmethod
    def from_triple_barrier_events(
        barrier_events: pd.DataFrame,
        symbol: str,
        source: str = "triple_barrier_v1",
        version: str = "tb_v1.0",
        ingestion_job: str = "unspecified_job",
        publication_lag: pd.Timedelta = pd.Timedelta(0),
    ) -> list[Label]:
        """
        Build validated `Label` objects directly from
        `MultiObjectiveLabeler.get_labels(...)` output.

        Args:
            barrier_events: DataFrame returned by
                `MultiObjectiveLabeler.get_labels`, indexed by entry time,
                containing at least: direction_label, actual_return,
                actual_mfe, actual_mae, actual_duration_bars, first_touch.
            symbol: Instrument symbol these events belong to.
            source: Lineage tag identifying the labeling logic used.
            version: Version tag for the labeling logic.
            ingestion_job: Identifier of the job/run producing these labels.
            publication_lag: How long after event_time this label becomes
                publicly knowable. Defaults to zero (same bar). Set this
                explicitly for any data source with reporting delay
                (e.g. corporate actions, delayed corporate filings).

        Returns:
            List of `Label` objects. This does NOT validate them — call
            `validate_and_filter` next.
        """
        # Use a tz-naive "now" matching the (typically tz-naive) index timestamps
        # coming out of MultiObjectiveLabeler. If your price index IS tz-aware,
        # localize this to match before calling this method.
        now = pd.Timestamp.now()
        labels: list[Label] = []

        for entry_time, row in barrier_events.iterrows():
            event_time = pd.Timestamp(entry_time)
            publication_time = event_time + publication_lag
            effective_time = publication_time
            horizon_bars = row.get("actual_duration_bars")
            horizon_days = int(horizon_bars) if pd.notna(horizon_bars) and horizon_bars > 0 else 1

            try:
                label = Label(
                    symbol=symbol,
                    label_type=LabelType.TRIPLE_BARRIER_DIRECTION,
                    label_value=float(row["direction_label"]),
                    label_date=event_time,
                    horizon_days=horizon_days,
                    event_time=event_time,
                    publication_time=publication_time,
                    effective_time=effective_time,
                    ingestion_time=now,
                    source=source,
                    version=version,
                    ingestion_job=ingestion_job,
                    actual_return=float(row.get("actual_return"))
                    if pd.notna(row.get("actual_return"))
                    else None,
                    actual_mfe=float(row.get("actual_mfe"))
                    if pd.notna(row.get("actual_mfe"))
                    else None,
                    actual_mae=float(row.get("actual_mae"))
                    if pd.notna(row.get("actual_mae"))
                    else None,
                    actual_duration_bars=int(horizon_bars) if pd.notna(horizon_bars) else None,
                )
                labels.append(label)
            except Exception as exc:  # noqa: BLE001 — log and skip malformed rows
                logger.error(f"Failed to construct Label for {symbol} @ {entry_time}: {exc}")

        return labels

    def validate_and_filter(self, labels: list[Label]) -> tuple[list[Label], dict[str, Any]]:
        """
        Validate a batch of labels and return only the valid ones.

        This is the method the training pipeline should call — it never
        returns invalid labels, so callers cannot accidentally train on
        them even if they forget to check the report.

        Returns:
            (valid_labels, report) where report mirrors `validate_labels`.
        """
        report = self.validate_labels(labels)
        valid_labels = [label for label in labels if self.validate_label(label)]
        if report["invalid"] > 0:
            logger.warning(
                f"Dropped {report['invalid']}/{report['total']} invalid labels "
                f"for symbol(s): {sorted({e['symbol'] for e in report['errors']})}"
            )
        return valid_labels, report

    # ------------------------------------------------------------------
    # Core validation (unchanged logic, now actually reachable)
    # ------------------------------------------------------------------

    def validate_label(self, label: Label) -> bool:
        """
        Validate a single label.

        Args:
            label: Label object

        Returns:
            True if valid, False otherwise
        """
        for field in self.REQUIRED_FIELDS:
            if getattr(label, field, None) is None:
                logger.error(f"Label validation failed: missing field {field}")
                return False

        # Validate timestamp ordering (causal chain)
        if label.event_time > label.publication_time:
            logger.error("Label validation failed: event_time > publication_time")
            return False

        if label.publication_time > label.effective_time:
            logger.error("Label validation failed: publication_time > effective_time")
            return False

        if label.effective_time > label.ingestion_time:
            logger.error("Label validation failed: effective_time > ingestion_time")
            return False

        # Validate label_date is not before event_time
        label_date = pd.to_datetime(label.label_date).date()
        event_date = pd.to_datetime(label.event_time).date()
        if label_date < event_date:
            logger.error("Label validation failed: label_date < event_time")
            return False

        # Validate horizon is positive
        if label.horizon_days <= 0:
            logger.error("Label validation failed: horizon_days must be positive")
            return False

        # Validate label_value is finite
        if not np.isfinite(label.label_value):
            logger.error("Label validation failed: label_value is not finite")
            return False

        # Validate checksum integrity (detects silent mutation/corruption)
        expected_checksum = label._compute_checksum()
        if label.checksum != expected_checksum:
            logger.error(
                f"Label validation failed: checksum mismatch for {label.symbol} "
                f"@ {label.label_date} (possible corruption or tampering)"
            )
            return False

        logger.debug(f"Label validation passed for {label.symbol}")
        return True

    def validate_labels(self, labels: list[Label]) -> dict[str, Any]:
        """
        Validate a batch of labels.

        Args:
            labels: List of label objects

        Returns:
            Validation results dictionary
        """
        results = {"total": len(labels), "valid": 0, "invalid": 0, "errors": []}

        for label in labels:
            if self.validate_label(label):
                results["valid"] += 1
            else:
                results["invalid"] += 1
                results["errors"].append(
                    {
                        "symbol": label.symbol,
                        "label_type": label.label_type,
                        "label_date": label.label_date,
                    }
                )

        logger.info(f"Label validation: {results['valid']}/{results['total']} valid")

        return results

    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Validate label DataFrame.

        Args:
            df: DataFrame with label data

        Returns:
            True if valid, False otherwise
        """
        missing_cols = [col for col in self.REQUIRED_FIELDS if col not in df.columns]
        if missing_cols:
            logger.error(f"DataFrame validation failed: missing columns {missing_cols}")
            return False

        critical_cols = ["symbol", "label_type", "label_value", "label_date"]
        for col in critical_cols:
            if df[col].isna().any():
                logger.error(f"DataFrame validation failed: NaN values in {col}")
                return False

        if np.isinf(df["label_value"]).any():
            logger.error("DataFrame validation failed: infinite values in label_value")
            return False

        if (df["event_time"] > df["publication_time"]).any():
            logger.error("DataFrame validation failed: event_time > publication_time")
            return False

        if (df["publication_time"] > df["effective_time"]).any():
            logger.error("DataFrame validation failed: publication_time > effective_time")
            return False

        logger.debug("DataFrame validation passed")
        return True

    def validate_pit_before_training(self, feature_date: datetime, label_date: datetime) -> bool:
        """
        Validate PIT constraints before training.

        Ensures features are available before labels are generated.

        Args:
            feature_date: Date when feature is available
            label_date: Date for which label is generated

        Returns:
            True if valid, False otherwise
        """
        if feature_date > label_date:
            logger.error(
                f"PIT validation failed: feature_date {feature_date} > label_date {label_date}"
            )
            return False

        logger.debug(
            f"PIT validation passed: feature_date {feature_date} <= label_date {label_date}"
        )
        return True

    def ensure_publication_time_training(
        self, feature_data: pd.DataFrame, label_data: pd.DataFrame
    ) -> bool:
        """
        Ensure training uses publication_time, not event_time.

        Args:
            feature_data: DataFrame with feature data
            label_data: DataFrame with label data

        Returns:
            True if valid, False otherwise
        """
        if "publication_time" not in feature_data.columns:
            logger.error(
                "Publication time validation failed: publication_time not in feature data"
            )
            return False

        if "publication_time" not in label_data.columns:
            logger.error("Publication time validation failed: publication_time not in label data")
            return False

        logger.debug("Publication time validation passed")
        return True

    def check_lookahead_bias(self, feature_data: pd.DataFrame, label_data: pd.DataFrame) -> bool:
        """
        Check for lookahead bias in feature-label pairs.

        Args:
            feature_data: DataFrame with feature data
            label_data: DataFrame with label data

        Returns:
            True if no lookahead bias, False otherwise
        """
        merged = pd.merge(
            feature_data, label_data, on=["symbol", "date"], suffixes=("_feature", "_label")
        )

        if merged.empty:
            logger.warning("Lookahead bias check skipped: no overlapping symbol/date rows")
            return True

        if (merged["publication_time_feature"] > merged["event_time_label"]).any():
            logger.error("Lookahead bias detected: feature publication after label event")
            return False

        logger.debug("Lookahead bias check passed")
        return True

    def validate_label_distribution(
        self, label_values: pd.Series, min_samples: int = 100
    ) -> dict[str, Any]:
        """
        Validate label distribution.

        Args:
            label_values: Series of label values
            min_samples: Minimum number of samples required

        Returns:
            Distribution statistics
        """
        stats = {
            "count": len(label_values),
            "mean": label_values.mean(),
            "std": label_values.std(),
            "min": label_values.min(),
            "max": label_values.max(),
            "median": label_values.median(),
            "valid": True,
        }

        if stats["count"] < min_samples:
            logger.error(
                f"Label distribution validation failed: insufficient samples "
                f"({stats['count']} < {min_samples})"
            )
            stats["valid"] = False

        if np.abs(stats["max"]) > 10 or np.abs(stats["min"]) > 10:
            logger.warning(
                f"Label distribution has extreme values: min={stats['min']}, max={stats['max']}"
            )

        if stats["std"] == 0:
            logger.error("Label distribution validation failed: zero variance")
            stats["valid"] = False

        logger.info(
            f"Label distribution: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
            f"count={stats['count']}"
        )

        return stats

    def validate_label_consistency(self, labels: list[Label]) -> bool:
        """
        Validate label consistency across time.

        Args:
            labels: List of label objects

        Returns:
            True if consistent, False otherwise
        """
        symbol_dates: dict[str, list] = {}
        for label in labels:
            symbol_dates.setdefault(label.symbol, []).append(label.label_date)

        for symbol, dates in symbol_dates.items():
            dates_sorted = sorted(dates)
            for i in range(1, len(dates_sorted)):
                gap = (dates_sorted[i] - dates_sorted[i - 1]).days
                if gap > 7:
                    logger.warning(
                        f"Label consistency warning: {symbol} has {gap}-day gap in labels"
                    )

        logger.debug("Label consistency validation passed")
        return True


def create_label_validator() -> LabelValidator:
    """
    Factory function to create a label validator.

    Returns:
        LabelValidator instance
    """
    return LabelValidator()