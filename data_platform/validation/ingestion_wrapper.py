"""
Ingestion Wrapper for Easy Integration

Provides easy integration of validation framework into existing ingestion pipelines.
Can be used as a decorator or direct function call.
"""

from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import pandas as pd

from data_platform.validation.enhanced_monitoring import (
    create_enhanced_monitor,
)
from data_platform.validation.ingestion_validator import (
    InvalidDataAction,
    create_ingestion_validator,
)
from utils.logger import get_logger

logger = get_logger("ingestion_wrapper", pipeline_id="data_ingestion")


class IngestionWrapper:
    """
    Wrapper class for easy integration of validation into ingestion pipelines.

    Usage:
        wrapper = IngestionWrapper()

        # Direct usage
        validated_df, report = wrapper.validate_and_process(
            df=df,
            dataset_name="equity_history",
            source="NSE"
        )

        # As decorator
        @wrapper.validate_ingestion
        def fetch_data(symbol, from_date, to_date):
            # Your existing fetch logic
            return df
    """

    def __init__(
        self,
        schema_registry_path: Path | None = None,
        quarantine_path: Path | None = None,
        default_action: InvalidDataAction = InvalidDataAction.ANNOTATE,
        enable_monitoring: bool = True,
    ):
        """
        Initialize ingestion wrapper.

        Args:
            schema_registry_path: Path to schema registry
            quarantine_path: Path to quarantine directory
            default_action: Default action for invalid data
            enable_monitoring: Enable enhanced monitoring
        """
        self.validator = create_ingestion_validator(
            schema_registry_path=schema_registry_path,
            quarantine_path=quarantine_path,
            default_action=default_action,
        )

        self.monitor = None
        if enable_monitoring:
            self.monitor = create_enhanced_monitor(
                schema_registry_path=schema_registry_path, quarantine_path=quarantine_path
            )

        self.logger = logger

    def validate_and_process(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        source: str,
        schema_version: str | None = None,
        action: InvalidDataAction | None = None,
        enable_monitoring: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Validate and process data at ingestion.

        Args:
            df: DataFrame to validate
            dataset_name: Name of dataset
            source: Data source
            schema_version: Expected schema version
            action: Action for invalid data
            enable_monitoring: Enable monitoring

        Returns:
            Tuple of (validated_df, validation_metadata)
        """
        self.logger.info(f"Validating {dataset_name} from {source}")

        # Run validation
        validated_df, validation_report, validation_metadata = self.validator.validate_at_ingestion(
            df=df,
            dataset_name=dataset_name,
            source=source,
            schema_version=schema_version,
            action=action,
        )

        # Run monitoring if enabled
        if enable_monitoring and self.monitor:
            monitoring_results = self.monitor.monitor_dataset(
                df=validated_df, dataset_name=dataset_name, source=source
            )
            validation_metadata["monitoring"] = monitoring_results

        # Log results
        if validation_report.is_acceptable():
            self.logger.info(
                f"Validation passed for {dataset_name}: "
                f"score={validation_report.calculate_score():.2f}"
            )
        else:
            self.logger.warning(
                f"Validation issues for {dataset_name}: "
                f"score={validation_report.calculate_score():.2f}, "
                f"critical_failures={len(validation_report.critical_failures)}"
            )

        return validated_df, validation_metadata

    def validate_ingestion(
        self, dataset_name: str, source: str = "unknown", action: InvalidDataAction | None = None
    ):
        """
        Decorator for validating ingestion functions.

        Usage:
            @wrapper.validate_ingestion(dataset_name="equity_history", source="NSE")
            def fetch_equity_data(symbol, from_date, to_date):
                # Your existing logic
                return df
        """

        def decorator(func: Callable):
            @wraps(func)
            def wrapper_func(*args, **kwargs) -> tuple[pd.DataFrame, dict[str, Any]]:
                # Call original function
                df = func(*args, **kwargs)

                # Validate and process
                validated_df, metadata = self.validate_and_process(
                    df=df, dataset_name=dataset_name, source=source, action=action
                )

                return validated_df, metadata

            return wrapper_func

        return decorator


# Singleton instance for easy access
_default_wrapper: IngestionWrapper | None = None


def get_default_wrapper() -> IngestionWrapper:
    """Get or create default wrapper instance."""
    global _default_wrapper
    if _default_wrapper is None:
        _default_wrapper = IngestionWrapper()
    return _default_wrapper


def validate_at_ingestion(
    df: pd.DataFrame, dataset_name: str, source: str, action: InvalidDataAction | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Convenience function for validation at ingestion.

    Args:
        df: DataFrame to validate
        dataset_name: Name of dataset
        source: Data source
        action: Action for invalid data

    Returns:
        Tuple of (validated_df, validation_metadata)
    """
    wrapper = get_default_wrapper()
    return wrapper.validate_and_process(
        df=df, dataset_name=dataset_name, source=source, action=action
    )


# Example usage for integration into existing pipelines
def integrate_into_equity_pipeline():
    """
    Example of how to integrate validation into equity_history pipeline.

    Add this to the EquityHistoryPipeline class:

    from data_platform.validation.ingestion_wrapper import validate_at_ingestion

    def download_ohlcv(self) -> pd.DataFrame:
        # ... existing download logic ...

        # After getting df, validate it
        df, validation_metadata = validate_at_ingestion(
            df=df,
            dataset_name=f"equity_history_{self.config.symbol}",
            source="NSE"
        )

        # Check if validation passed
        if not validation_metadata['validation_passed']:
            raise ValueError(
                f"Validation failed: {validation_metadata['handling_metadata']}"
            )

        return df
    """
    pass
