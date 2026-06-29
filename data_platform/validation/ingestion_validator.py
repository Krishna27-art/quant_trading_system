"""
Ingestion Validation Framework

Institutional-grade validation at ingestion time with:
- Schema enforcement
- Invalid data handling (reject/annotate/backfill)
- Schema versioning
- Data lineage integration
"""

import json
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from data_platform.validation.base_validator import (
    ValidationReport,
    ValidationResult,
    ValidationSeverity,
)
from data_platform.validation.corporate_rules import CorporateValidator
from data_platform.validation.equity_rules import EquityValidator
from data_platform.validation.options_rules import OptionsValidator
from utils.data_lineage import compute_checksum, write_lineage_record
from utils.logger import get_logger
from utils.time_utils import now_ist

logger = get_logger("ingestion_validator", pipeline_id="data_ingestion")


class InvalidDataAction(Enum):
    """Action to take when invalid data is detected."""

    REJECT = "reject"  # Reject entire batch
    ANNOTATE = "annotate"  # Keep data but mark as invalid
    BACKFILL = "backfill"  # Attempt to backfill missing values
    QUARANTINE = "quarantine"  # Move to quarantine for review


class SchemaVersion:
    """Schema version with migration support."""

    def __init__(
        self, version: str, schema: dict[str, str], migration_rules: dict[str, Any] | None = None
    ):
        self.version = version
        self.schema = schema
        self.migration_rules = migration_rules or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "schema": self.schema,
            "migration_rules": self.migration_rules,
        }


class SchemaRegistry:
    """Registry for schema versions with migration support."""

    def __init__(self, registry_path: Path):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.schemas: dict[str, dict[str, SchemaVersion]] = {}
        self._load_registry()

    def _load_registry(self):
        """Load schema registry from disk."""
        registry_file = self.registry_path / "schema_registry.json"
        if registry_file.exists():
            with open(registry_file) as f:
                data = json.load(f)
                for dataset_name, versions in data.items():
                    self.schemas[dataset_name] = {}
                    for version_str, schema_data in versions.items():
                        self.schemas[dataset_name][version_str] = SchemaVersion(
                            version=version_str,
                            schema=schema_data["schema"],
                            migration_rules=schema_data.get("migration_rules", {}),
                        )

    def _save_registry(self):
        """Save schema registry to disk."""
        data = {}
        for dataset_name, versions in self.schemas.items():
            data[dataset_name] = {}
            for version_str, schema_version in versions.items():
                data[dataset_name][version_str] = schema_version.to_dict()

        registry_file = self.registry_path / "schema_registry.json"
        with open(registry_file, "w") as f:
            json.dump(data, f, indent=2)

    def register_schema(
        self,
        dataset_name: str,
        version: str,
        schema: dict[str, str],
        migration_rules: dict[str, Any] | None = None,
    ):
        """Register a new schema version."""
        if dataset_name not in self.schemas:
            self.schemas[dataset_name] = {}

        self.schemas[dataset_name][version] = SchemaVersion(
            version=version, schema=schema, migration_rules=migration_rules
        )

        self._save_registry()
        logger.info(f"Registered schema version {version} for {dataset_name}")

    def get_schema(self, dataset_name: str, version: str) -> SchemaVersion | None:
        """Get a specific schema version."""
        if dataset_name in self.schemas and version in self.schemas[dataset_name]:
            return self.schemas[dataset_name][version]
        return None

    def get_latest_schema(self, dataset_name: str) -> SchemaVersion | None:
        """Get the latest schema version for a dataset."""
        if dataset_name not in self.schemas or not self.schemas[dataset_name]:
            return None

        # Sort versions and return latest
        versions = sorted(self.schemas[dataset_name].keys(), reverse=True)
        return self.schemas[dataset_name][versions[0]]

    def migrate_data(
        self, df: pd.DataFrame, dataset_name: str, from_version: str, to_version: str
    ) -> pd.DataFrame:
        """Migrate data from one schema version to another."""
        schema_from = self.get_schema(dataset_name, from_version)
        schema_to = self.get_schema(dataset_name, to_version)

        if not schema_from or not schema_to:
            raise ValueError("Cannot find schemas for migration")

        logger.info(f"Migrating {dataset_name} from {from_version} to {to_version}")

        # Apply migration rules
        migration_rules = schema_to.migration_rules
        df_migrated = df.copy()

        for rule in migration_rules.get("column_renames", {}):
            old_name, new_name = rule
            if old_name in df_migrated.columns:
                df_migrated = df_migrated.rename(columns={old_name: new_name})

        for rule in migration_rules.get("column_additions", {}):
            col_name, default_value = rule
            if col_name not in df_migrated.columns:
                df_migrated[col_name] = default_value

        for rule in migration_rules.get("column_deletions", []):
            if rule in df_migrated.columns:
                df_migrated = df_migrated.drop(columns=[rule])

        return df_migrated


class IngestionValidator:
    """
    Unified validation framework for data ingestion.

    Enforces schema checks, handles invalid data, tracks lineage,
    and supports schema versioning.
    """

    def __init__(
        self,
        schema_registry_path: Path,
        quarantine_path: Path,
        default_action: InvalidDataAction = InvalidDataAction.ANNOTATE,
    ):
        """
        Initialize ingestion validator.

        Args:
            schema_registry_path: Path to schema registry
            quarantine_path: Path to quarantine directory
            default_action: Default action for invalid data
        """
        self.schema_registry = SchemaRegistry(schema_registry_path)
        self.quarantine_path = Path(quarantine_path)
        self.quarantine_path.mkdir(parents=True, exist_ok=True)
        self.default_action = default_action
        self.logger = logger

        # Initialize validators
        self.equity_validator = EquityValidator("equity_history")
        self.options_validator = OptionsValidator("options_chain")
        self.corporate_validator = CorporateValidator("corporate_actions")

    def validate_at_ingestion(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        source: str,
        schema_version: str | None = None,
        action: InvalidDataAction | None = None,
    ) -> tuple[pd.DataFrame, ValidationReport, dict[str, Any]]:
        """
        Validate data at ingestion time.

        Args:
            df: DataFrame to validate
            dataset_name: Name of dataset
            source: Data source (e.g., 'NSE', 'BSE')
            schema_version: Expected schema version (None for latest)
            action: Action for invalid data (None for default)

        Returns:
            Tuple of (validated_df, validation_report, metadata)
        """
        self.logger.info(f"Validating {dataset_name} at ingestion from {source}")

        # Get schema version
        if schema_version is None:
            schema_version_obj = self.schema_registry.get_latest_schema(dataset_name)
            if schema_version_obj:
                schema_version = schema_version_obj.version
            else:
                pass
        else:
            schema_version_obj = self.schema_registry.get_schema(dataset_name, schema_version)

        # Determine action
        if action is None:
            action = self.default_action

        # Run validation
        report = self._run_validation(df, dataset_name)

        # Handle invalid data based on action
        validated_df, handling_metadata = self._handle_invalid_data(
            df, report, action, dataset_name
        )

        # Add validation metadata
        metadata = {
            "dataset_name": dataset_name,
            "source": source,
            "schema_version": schema_version,
            "validation_passed": report.is_acceptable(),
            "validation_score": report.calculate_score(),
            "action_taken": action.value,
            "total_records": len(df),
            "valid_records": len(validated_df),
            "rejected_records": len(df) - len(validated_df),
            "validation_timestamp": now_ist().isoformat(),
            "handling_metadata": handling_metadata,
        }

        # Write lineage record
        if not validated_df.empty:
            checksum = compute_checksum(validated_df)
            lineage_id = write_lineage_record(
                dataset=dataset_name, source=source, row_count=len(validated_df), checksum=checksum
            )
            metadata["lineage_id"] = lineage_id

        self.logger.info(
            f"Ingestion validation complete: {metadata['valid_records']}/{metadata['total_records']} "
            f"records valid (score: {metadata['validation_score']:.2f})"
        )

        return validated_df, report, metadata

    def _run_validation(self, df: pd.DataFrame, dataset_name: str) -> ValidationReport:
        """Run appropriate validation based on dataset type."""
        if "equity" in dataset_name.lower():
            full_ohlcv = {"open", "high", "low", "close", "volume"}
            if not full_ohlcv.issubset({str(col).lower() for col in df.columns}):
                return self._run_minimal_equity_validation(df, dataset_name)
            return self.equity_validator.validate(df)
        elif "option" in dataset_name.lower():
            return self.options_validator.validate(df)
        elif "corporate" in dataset_name.lower():
            return self.corporate_validator.validate(df)
        else:
            # Generic validation
            return self._run_generic_validation(df, dataset_name)

    def _run_minimal_equity_validation(
        self, df: pd.DataFrame, dataset_name: str
    ) -> ValidationReport:
        """Validate equity snapshots that only carry close/volume fields."""
        validator = self.equity_validator
        df = validator._canonicalize_columns(df)
        report = ValidationReport(dataset_name=dataset_name, total_records=len(df))

        required = ["date", "symbol", "close", "volume"]
        missing = [col for col in required if col not in df.columns]
        report.add_result(
            ValidationResult(
                rule_name="minimal_equity_columns_check",
                passed=not missing,
                severity=ValidationSeverity.CRITICAL,
                message=(
                    "Required minimal equity columns present"
                    if not missing
                    else f"Missing minimal equity columns: {missing}"
                ),
                details={"required_columns": required, "missing_columns": missing},
            )
        )
        if missing:
            return report

        report.add_result(validator._check_price_positive(df))
        report.add_result(validator._check_volume_positive(df))
        report.add_result(validator._check_not_null(df, required))
        report.add_result(validator._check_no_duplicates(df, ["date", "symbol"]))
        report.add_result(validator._check_date_order(df, "date"))
        return report

    def _run_generic_validation(self, df: pd.DataFrame, dataset_name: str) -> ValidationReport:
        """Run generic validation for unknown dataset types."""
        report = ValidationReport(dataset_name=dataset_name, total_records=len(df))

        # Check for null values in key columns
        key_columns = ["timestamp", "symbol", "date"]
        existing_key_columns = [col for col in key_columns if col in df.columns]
        if existing_key_columns:
            result = self.equity_validator._check_not_null(df, existing_key_columns)
            report.add_result(result)

        # Check for duplicates
        if existing_key_columns:
            result = self.equity_validator._check_no_duplicates(df, existing_key_columns)
            report.add_result(result)

        return report

    def _handle_invalid_data(
        self,
        df: pd.DataFrame,
        report: ValidationReport,
        action: InvalidDataAction,
        dataset_name: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Handle invalid data based on specified action.

        Args:
            df: Original DataFrame
            report: Validation report
            action: Action to take
            dataset_name: Dataset name

        Returns:
            Tuple of (handled_df, metadata)
        """
        metadata = {"action": action.value}

        if report.is_acceptable():
            # All data is valid
            return df, metadata

        if action == InvalidDataAction.REJECT:
            # Reject entire batch
            self.logger.warning(f"Rejecting entire batch for {dataset_name}")
            return pd.DataFrame(), metadata

        elif action == InvalidDataAction.ANNOTATE:
            # Keep data but add validation metadata
            df_annotated = df.copy()
            df_annotated["_validation_passed"] = report.is_acceptable()
            df_annotated["_validation_score"] = report.calculate_score()
            df_annotated["_validation_timestamp"] = now_ist().isoformat()
            metadata["annotated"] = True
            return df_annotated, metadata

        elif action == InvalidDataAction.BACKFILL:
            # Attempt to backfill missing values
            df_backfilled = self._backfill_data(df, report)
            metadata["backfilled"] = True
            metadata["backfilled_columns"] = self._get_backfilled_columns(df, df_backfilled)
            return df_backfilled, metadata

        elif action == InvalidDataAction.QUARANTINE:
            # Move invalid records to quarantine
            df_valid, df_quarantined = self._quarantine_data(df, report, dataset_name)
            metadata["quarantined"] = True
            metadata["quarantined_count"] = len(df_quarantined)
            return df_valid, metadata

        return df, metadata

    def _backfill_data(self, df: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
        """Attempt to backfill missing or invalid values."""
        df_backfilled = df.copy()

        # Backfill null values with forward fill then backward fill
        for col in df_backfilled.columns:
            if df_backfilled[col].isnull().any():
                # CRITICAL: Never use backward fill on financial data as it causes lookahead bias
                df_backfilled[col] = df_backfilled[col].ffill()

        return df_backfilled

    def _get_backfilled_columns(
        self, original_df: pd.DataFrame, backfilled_df: pd.DataFrame
    ) -> list[str]:
        """Get list of columns that were backfilled."""
        backfilled_columns = []
        for col in original_df.columns:
            if not original_df[col].equals(backfilled_df[col]):
                backfilled_columns.append(col)
        return backfilled_columns

    def _quarantine_data(
        self, df: pd.DataFrame, report: ValidationReport, dataset_name: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Separate valid and invalid data, quarantine invalid records.

        Returns:
            Tuple of (valid_df, quarantined_df)
        """
        # For now, quarantine all data if validation fails
        # In production, implement row-level quarantine based on validation results

        timestamp = now_ist().strftime("%Y%m%d_%H%M%S")
        quarantine_file = self.quarantine_path / f"{dataset_name}_quarantine_{timestamp}.parquet"

        # Save to quarantine
        df.to_parquet(quarantine_file, index=False)

        self.logger.warning(f"Quarantined {len(df)} records to {quarantine_file}")

        # Return empty valid DataFrame
        return pd.DataFrame(), df

    def register_dataset_schema(
        self,
        dataset_name: str,
        version: str,
        schema: dict[str, str],
        migration_rules: dict[str, Any] | None = None,
    ):
        """Register a schema for a dataset."""
        self.schema_registry.register_schema(dataset_name, version, schema, migration_rules)

    def get_schema_version(self, dataset_name: str, version: str) -> dict[str, str] | None:
        """Get schema version for a dataset."""
        schema_version = self.schema_registry.get_schema(dataset_name, version)
        return schema_version.schema if schema_version else None


def create_ingestion_validator(
    schema_registry_path: Path | None = None,
    quarantine_path: Path | None = None,
    default_action: InvalidDataAction = InvalidDataAction.ANNOTATE,
) -> IngestionValidator:
    """
    Factory function to create an ingestion validator.

    Args:
        schema_registry_path: Path to schema registry (default: data/bronze/schema_registry)
        quarantine_path: Path to quarantine directory (default: data/bronze/quarantine)
        default_action: Default action for invalid data

    Returns:
        IngestionValidator instance
    """
    from config.settings import BRONZE_DIR

    if schema_registry_path is None:
        schema_registry_path = BRONZE_DIR / "schema_registry"

    if quarantine_path is None:
        quarantine_path = BRONZE_DIR / "quarantine"

    return IngestionValidator(
        schema_registry_path=schema_registry_path,
        quarantine_path=quarantine_path,
        default_action=default_action,
    )
