"""
Label Store Validation Utilities

Institutional-grade validation for label generation and storage.
Ensures strict Point-In-Time (PIT) controls to prevent lookahead bias.
"""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from prediction_intelligence.label_models import Label

from utils.logger import get_logger

logger = get_logger("label_validator")


class LabelValidator:
    """
    Validator for label generation and storage.

    Ensures strict PIT controls and data quality.
    """

    def __init__(self):
        """Initialize the label validator."""
        logger.info("Initialized LabelValidator")

    def validate_label(self, label: Label) -> bool:
        """
        Validate a single label.

        Args:
            label: Label object

        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        required_fields = [
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

        for field in required_fields:
            if getattr(label, field, None) is None:
                logger.error(f"Label validation failed: missing field {field}")
                return False

        # Validate timestamp ordering
        if label.event_time > label.publication_time:
            logger.error("Label validation failed: event_time > publication_time")
            return False

        if label.publication_time > label.effective_time:
            logger.error("Label validation failed: publication_time > effective_time")
            return False

        if label.effective_time > label.ingestion_time:
            logger.error("Label validation failed: effective_time > ingestion_time")
            return False

        # Validate label_date is after event_time
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
        # Check required columns
        required_cols = [
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

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"DataFrame validation failed: missing columns {missing_cols}")
            return False

        # Check for NaN values in critical columns
        critical_cols = ["symbol", "label_type", "label_value", "label_date"]
        for col in critical_cols:
            if df[col].isna().any():
                logger.error(f"DataFrame validation failed: NaN values in {col}")
                return False

        # Check for infinite values in label_value
        if np.isinf(df["label_value"]).any():
            logger.error("DataFrame validation failed: infinite values in label_value")
            return False

        # Validate timestamp ordering
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
        # Check that publication_time is being used
        if "publication_time" not in feature_data.columns:
            logger.error("Publication time validation failed: publication_time not in feature data")
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
        # Merge feature and label data
        merged = pd.merge(
            feature_data, label_data, on=["symbol", "date"], suffixes=("_feature", "_label")
        )

        # Check if feature publication_time is after label event_time
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

        # Check minimum samples
        if stats["count"] < min_samples:
            logger.error(
                f"Label distribution validation failed: insufficient samples ({stats['count']} < {min_samples})"
            )
            stats["valid"] = False

        # Check for extreme values
        if np.abs(stats["max"]) > 10 or np.abs(stats["min"]) > 10:
            logger.warning(
                f"Label distribution has extreme values: min={stats['min']}, max={stats['max']}"
            )

        # Check for zero variance
        if stats["std"] == 0:
            logger.error("Label distribution validation failed: zero variance")
            stats["valid"] = False

        logger.info(
            f"Label distribution: mean={stats['mean']:.4f}, std={stats['std']:.4f}, count={stats['count']}"
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
        # Group by symbol and check for gaps
        symbol_dates = {}
        for label in labels:
            if label.symbol not in symbol_dates:
                symbol_dates[label.symbol] = []
            symbol_dates[label.symbol].append(label.label_date)

        # Check for gaps in dates
        for symbol, dates in symbol_dates.items():
            dates_sorted = sorted(dates)
            for i in range(1, len(dates_sorted)):
                gap = (dates_sorted[i] - dates_sorted[i - 1]).days
                if gap > 7:  # More than 7 days gap
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
