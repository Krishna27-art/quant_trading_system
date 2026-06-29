"""
Point-In-Time (PIT) Data Validation Utilities

CRITICAL for institutional quant work - prevents look-ahead bias by ensuring
training data respects publication timing.

MANDATORY RULE: Never train on event_time. Always train on publication_time.

MANDATORY DATA LINEAGE FIELDS for all datasets:
- source: Data source (e.g., nselib, vendor_feed, manual_correction)
- version: Data version/schema version
- ingestion_job: Ingestion job identifier
- checksum: Data integrity checksum

Critical for institutional quant work - Jane Street and Citadel care about:
"Where did this number come from?"
"""

import hashlib
from datetime import datetime
from typing import Any

import pandas as pd

from utils.logger import get_logger

logger = get_logger("pit_validator")


class PITValidator:
    """
    Validates Point-In-Time data integrity to prevent look-ahead bias.

    This is the most critical validation for institutional quant work.
    Without proper PIT validation, backtests will suffer from look-ahead bias.
    """

    REQUIRED_PIT_FIELDS = ["event_time", "publication_time", "ingestion_time", "effective_time"]
    REQUIRED_LINEAGE_FIELDS = ["source", "version", "ingestion_job"]

    def __init__(self):
        """Initialize PIT validator."""
        self.logger = logger

    def validate_dataframe(self, df: pd.DataFrame, dataset_name: str) -> dict[str, Any]:
        """
        Validate that a DataFrame has all required PIT and lineage fields.

        Args:
            df: DataFrame to validate
            dataset_name: Name of the dataset for logging

        Returns:
            Validation results dictionary
        """
        self.logger.info(f"Validating PIT and lineage fields for dataset: {dataset_name}")

        results = {
            "dataset_name": dataset_name,
            "is_valid": True,
            "missing_pit_fields": [],
            "missing_lineage_fields": [],
            "invalid_rows": 0,
            "total_rows": len(df),
            "errors": [],
        }

        # Check for required PIT fields
        missing_pit_fields = [
            field for field in self.REQUIRED_PIT_FIELDS if field not in df.columns
        ]
        if missing_pit_fields:
            results["is_valid"] = False
            results["missing_pit_fields"] = missing_pit_fields
            error_msg = f"Missing required PIT fields: {missing_pit_fields}"
            self.logger.error(error_msg)
            results["errors"].append(error_msg)

        # Validate timestamp ordering
        if not df.empty and all(field in df.columns for field in self.REQUIRED_PIT_FIELDS):
            # Check publication_time >= event_time
            invalid_pub = df[df["publication_time"] < df["event_time"]]
            if not invalid_pub.empty:
                results["is_valid"] = False
                results["invalid_rows"] += len(invalid_pub)
                error_msg = (
                    f"{len(invalid_pub)} rows have publication_time < event_time (lookahead bias)"
                )
                self.logger.error(error_msg)
                results["errors"].append(error_msg)

        # Check for required lineage fields after PIT ordering so causal errors lead.
        missing_lineage_fields = [
            field for field in self.REQUIRED_LINEAGE_FIELDS if field not in df.columns
        ]
        if missing_lineage_fields:
            results["is_valid"] = False
            results["missing_lineage_fields"] = missing_lineage_fields
            error_msg = f"Missing required lineage fields: {missing_lineage_fields}"
            self.logger.error(error_msg)
            results["errors"].append(error_msg)

            # Check effective_time >= publication_time
            invalid_eff = df[df["effective_time"] < df["publication_time"]]
            if not invalid_eff.empty:
                results["is_valid"] = False
                results["invalid_rows"] += len(invalid_eff)
                error_msg = f"{len(invalid_eff)} rows have effective_time < publication_time"
                self.logger.error(error_msg)
                results["errors"].append(error_msg)

        if results["is_valid"]:
            self.logger.info(f"PIT validation passed for {dataset_name}")
        else:
            self.logger.error(f"PIT validation failed for {dataset_name}")

        return results

    def validate_training_timestamp(self, timestamp_column: str) -> bool:
        """
        Validate that training uses publication_time, not event_time.

        CRITICAL: This prevents look-ahead bias.

        Args:
            timestamp_column: The timestamp column being used for training

        Returns:
            True if valid (using publication_time), False otherwise
        """
        if timestamp_column == "event_time":
            error_msg = "CRITICAL: Training on event_time causes look-ahead bias! Use publication_time instead."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if timestamp_column == "publication_time":
            self.logger.info("Training on publication_time - CORRECT (prevents look-ahead bias)")
            return True

        warning_msg = f"Warning: Training on '{timestamp_column}'. Ensure this is equivalent to publication_time."
        self.logger.warning(warning_msg)
        return True

    def check_lookahead_bias(
        self,
        df: pd.DataFrame,
        feature_timestamp_col: str,
        target_timestamp_col: str,
        max_lookahead_days: int = 0,
    ) -> dict[str, Any]:
        """
        Check for look-ahead bias in feature-target pairs.

        Args:
            df: DataFrame with features and targets
            feature_timestamp_col: Column name for feature timestamps
            target_timestamp_col: Column name for target timestamps
            max_lookahead_days: Maximum allowed lookahead in days (0 = none)

        Returns:
            Validation results
        """
        self.logger.info(
            f"Checking look-ahead bias between {feature_timestamp_col} and {target_timestamp_col}"
        )

        results = {
            "has_lookahead_bias": False,
            "biased_rows": 0,
            "total_rows": len(df),
            "max_lookahead_days": max_lookahead_days,
        }

        if df.empty:
            return results

        # Calculate time difference
        time_diff = pd.to_datetime(df[target_timestamp_col]) - pd.to_datetime(
            df[feature_timestamp_col]
        )

        # Check for negative time differences (target before feature)
        negative_diff = time_diff[time_diff < pd.Timedelta(days=0)]
        if not negative_diff.empty:
            results["has_lookahead_bias"] = True
            results["biased_rows"] = len(negative_diff)
            error_msg = (
                f"Found {len(negative_diff)} rows with target timestamp before feature timestamp"
            )
            self.logger.error(error_msg)

        # Check for excessive lookahead
        if max_lookahead_days > 0:
            excessive_lookahead = time_diff[time_diff > pd.Timedelta(days=max_lookahead_days)]
            if not excessive_lookahead.empty:
                results["has_lookahead_bias"] = True
                results["biased_rows"] += len(excessive_lookahead)
                error_msg = f"Found {len(excessive_lookahead)} rows with lookahead > {max_lookahead_days} days"
                self.logger.error(error_msg)

        if not results["has_lookahead_bias"]:
            self.logger.info("No look-ahead bias detected")

        return results

    def get_pit_snapshot(
        self, df: pd.DataFrame, as_of_date: datetime, timestamp_col: str = "publication_time"
    ) -> pd.DataFrame:
        """
        Get data as of a specific point in time.

        This is the correct way to query historical data for backtesting.
        Uses publication_time to ensure data availability.

        Args:
            df: DataFrame with PIT timestamps
            as_of_date: Point in time to query
            timestamp_col: Timestamp column to use (default: publication_time)

        Returns:
            DataFrame with data available as of as_of_date
        """
        self.logger.info(f"Getting PIT snapshot as of {as_of_date}")

        # Validate using publication_time
        if timestamp_col == "event_time":
            self.logger.warning("Using event_time for snapshot - may cause look-ahead bias")

        # Filter data available as of the specified date
        snapshot = df[pd.to_datetime(df[timestamp_col]) <= as_of_date].copy()

        self.logger.info(f"Snapshot contains {len(snapshot)} records available as of {as_of_date}")

        return snapshot

    def validate_pit_completeness(
        self, df: pd.DataFrame, required_fields: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Validate completeness of PIT fields in DataFrame.

        Args:
            df: DataFrame to validate
            required_fields: List of required fields (defaults to REQUIRED_PIT_FIELDS + REQUIRED_LINEAGE_FIELDS)

        Returns:
            Validation results
        """
        if required_fields is None:
            required_fields = self.REQUIRED_PIT_FIELDS + self.REQUIRED_LINEAGE_FIELDS

        results = {"total_rows": len(df), "field_completeness": {}, "is_complete": True}

        for field in required_fields:
            if field in df.columns:
                null_count = df[field].isna().sum()
                completeness = (len(df) - null_count) / len(df) * 100
                results["field_completeness"][field] = {
                    "completeness_pct": completeness,
                    "null_count": null_count,
                }

                if completeness < 100:
                    results["is_complete"] = False
                    self.logger.warning(
                        f"Field {field} is {completeness:.2f}% complete ({null_count} null values)"
                    )
            else:
                results["is_complete"] = False
                results["field_completeness"][field] = {
                    "completeness_pct": 0,
                    "null_count": len(df),
                }
                self.logger.error(f"Field {field} is missing from DataFrame")

        return results

    def validate_lineage(self, df: pd.DataFrame, dataset_name: str) -> dict[str, Any]:
        """
        Validate data lineage fields for provenance tracking.

        Critical for institutional quant work - Jane Street and Citadel care about:
        "Where did this number come from?"

        Args:
            df: DataFrame to validate
            dataset_name: Name of the dataset for logging

        Returns:
            Validation results dictionary
        """
        self.logger.info(f"Validating data lineage for dataset: {dataset_name}")

        results = {
            "dataset_name": dataset_name,
            "is_valid": True,
            "missing_fields": [],
            "source_distribution": {},
            "version_distribution": {},
            "job_distribution": {},
            "checksum_completeness": 0,
            "errors": [],
        }

        # Check for required lineage fields
        missing_fields = [
            field for field in self.REQUIRED_LINEAGE_FIELDS if field not in df.columns
        ]
        if missing_fields:
            results["is_valid"] = False
            results["missing_fields"] = missing_fields
            error_msg = f"Missing required lineage fields: {missing_fields}"
            self.logger.error(error_msg)
            results["errors"].append(error_msg)
            return results

        # Analyze source distribution
        if "source" in df.columns:
            source_counts = df["source"].value_counts().to_dict()
            results["source_distribution"] = source_counts
            self.logger.info(f"Source distribution: {source_counts}")

        # Analyze version distribution
        if "version" in df.columns:
            version_counts = df["version"].value_counts().to_dict()
            results["version_distribution"] = version_counts
            self.logger.info(f"Version distribution: {version_counts}")

        # Analyze job distribution
        if "ingestion_job" in df.columns:
            job_counts = df["ingestion_job"].value_counts().to_dict()
            results["job_distribution"] = job_counts
            self.logger.info(f"Ingestion job distribution: {job_counts}")

        # Check checksum completeness
        if "checksum" in df.columns:
            checksum_count = df["checksum"].notna().sum()
            results["checksum_completeness"] = (checksum_count / len(df)) * 100
            self.logger.info(f"Checksum completeness: {results['checksum_completeness']:.2f}%")

        if results["is_valid"]:
            self.logger.info(f"Data lineage validation passed for {dataset_name}")
        else:
            self.logger.error(f"Data lineage validation failed for {dataset_name}")

        return results

    def calculate_checksum(self, data: dict) -> str:
        """
        Calculate checksum for data integrity verification.

        Args:
            data: Dictionary of data fields

        Returns:
            SHA256 checksum as hex string
        """
        # Create string representation of data
        data_str = str(sorted(data.items()))
        return hashlib.sha256(data_str.encode()).hexdigest()

    def verify_checksum(self, data: dict, expected_checksum: str) -> bool:
        """
        Verify data integrity using checksum.

        Args:
            data: Dictionary of data fields
            expected_checksum: Expected checksum value

        Returns:
            True if checksum matches, False otherwise
        """
        calculated_checksum = self.calculate_checksum(data)
        return calculated_checksum == expected_checksum


def validate_pit_before_training(df: pd.DataFrame, dataset_name: str) -> bool:
    """
    Convenience function to validate PIT data before training.

    Args:
        df: DataFrame to validate
        dataset_name: Name of the dataset

    Returns:
        True if validation passes, raises exception otherwise
    """
    validator = PITValidator()
    results = validator.validate_dataframe(df, dataset_name)

    if not results["is_valid"]:
        error_msg = f"PIT validation failed for {dataset_name}: {results['errors']}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    return True


def ensure_publication_time_training(timestamp_col: str) -> None:
    """
    Convenience function to ensure training uses publication_time.

    Args:
        timestamp_col: The timestamp column being used

    Raises:
        ValueError if using event_time
    """
    validator = PITValidator()
    validator.validate_training_timestamp(timestamp_col)
