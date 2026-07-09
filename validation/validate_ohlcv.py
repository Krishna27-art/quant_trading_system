"""
OHLCV Data Validation Module

Validates candlestick data for correctness and consistency.
Checks for price relationships, volume validity, timestamp ordering, and duplicates.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import pandas as pd
import numpy as np

from utils.logger import get_logger

logger = get_logger("validation.ohlcv")


@dataclass
class ValidationReport:
    """Report from OHLCV validation."""
    passed: bool
    total_checks: int
    failed_checks: int
    errors: List[str]
    warnings: List[str]
    timestamp: datetime
    dataframe_shape: tuple[int, int]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "passed": self.passed,
            "total_checks": self.total_checks,
            "failed_checks": self.failed_checks,
            "errors": self.errors,
            "warnings": self.warnings,
            "timestamp": self.timestamp.isoformat(),
            "dataframe_shape": self.dataframe_shape,
        }


class OHLCVValidator:
    """
    Validates OHLCV data for correctness.
    
    Checks:
    1. High >= Open
    2. High >= Close
    3. Low <= Open
    4. Low <= Close
    5. High >= Low
    6. Volume >= 0
    7. No duplicate timestamps
    8. Timestamp sorted ascending
    9. No duplicate rows
    10. No NaN values in critical columns
    11. No infinite values
    12. Price values > 0
    """

    REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

    def __init__(self, strict: bool = True):
        """
        Initialize validator.
        
        Args:
            strict: If True, fail on any error. If False, collect all errors.
        """
        self.strict = strict

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """
        Run all validation checks on OHLCV DataFrame.
        
        Args:
            df: DataFrame with columns: timestamp, open, high, low, close, volume
            
        Returns:
            ValidationReport with check results
        """
        errors = []
        warnings = []
        total_checks = 0
        failed_checks = 0

        # Check required columns
        total_checks += 1
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            failed_checks += 1
            if self.strict:
                return self._create_report(False, total_checks, failed_checks, errors, warnings, df.shape)
        else:
            logger.debug("Required columns check passed")

        # Check DataFrame is not empty
        total_checks += 1
        if df.empty:
            errors.append("DataFrame is empty")
            failed_checks += 1
            if self.strict:
                return self._create_report(False, total_checks, failed_checks, errors, warnings, df.shape)
        else:
            logger.debug("DataFrame non-empty check passed")

        # Check 1: High >= Open
        total_checks += 1
        if not (df["high"] >= df["open"]).all():
            invalid_count = (~(df["high"] >= df["open"])).sum()
            errors.append(f"High < Open in {invalid_count} rows")
            failed_checks += 1
            logger.warning(f"High < Open in {invalid_count} rows")
        else:
            logger.debug("High >= Open check passed")

        # Check 2: High >= Close
        total_checks += 1
        if not (df["high"] >= df["close"]).all():
            invalid_count = (~(df["high"] >= df["close"])).sum()
            errors.append(f"High < Close in {invalid_count} rows")
            failed_checks += 1
            logger.warning(f"High < Close in {invalid_count} rows")
        else:
            logger.debug("High >= Close check passed")

        # Check 3: Low <= Open
        total_checks += 1
        if not (df["low"] <= df["open"]).all():
            invalid_count = (~(df["low"] <= df["open"])).sum()
            errors.append(f"Low > Open in {invalid_count} rows")
            failed_checks += 1
            logger.warning(f"Low > Open in {invalid_count} rows")
        else:
            logger.debug("Low <= Open check passed")

        # Check 4: Low <= Close
        total_checks += 1
        if not (df["low"] <= df["close"]).all():
            invalid_count = (~(df["low"] <= df["close"])).sum()
            errors.append(f"Low > Close in {invalid_count} rows")
            failed_checks += 1
            logger.warning(f"Low > Close in {invalid_count} rows")
        else:
            logger.debug("Low <= Close check passed")

        # Check 5: High >= Low
        total_checks += 1
        if not (df["high"] >= df["low"]).all():
            invalid_count = (~(df["high"] >= df["low"])).sum()
            errors.append(f"High < Low in {invalid_count} rows")
            failed_checks += 1
            logger.warning(f"High < Low in {invalid_count} rows")
        else:
            logger.debug("High >= Low check passed")

        # Check 6: Volume >= 0
        total_checks += 1
        if not (df["volume"] >= 0).all():
            invalid_count = (~(df["volume"] >= 0)).sum()
            errors.append(f"Volume < 0 in {invalid_count} rows")
            failed_checks += 1
            logger.warning(f"Volume < 0 in {invalid_count} rows")
        else:
            logger.debug("Volume >= 0 check passed")

        # Check 7: No duplicate timestamps
        total_checks += 1
        if "timestamp" in df.columns:
            duplicate_count = df["timestamp"].duplicated().sum()
            if duplicate_count > 0:
                errors.append(f"Duplicate timestamps: {duplicate_count} rows")
                failed_checks += 1
                logger.warning(f"Duplicate timestamps: {duplicate_count} rows")
            else:
                logger.debug("No duplicate timestamps check passed")
        else:
            errors.append("Timestamp column missing")
            failed_checks += 1

        # Check 8: Timestamp sorted ascending
        total_checks += 1
        if "timestamp" in df.columns:
            if not df["timestamp"].is_monotonic_increasing:
                errors.append("Timestamps not sorted ascending")
                failed_checks += 1
                logger.warning("Timestamps not sorted ascending")
            else:
                logger.debug("Timestamp sorted ascending check passed")

        # Check 9: No duplicate rows
        total_checks += 1
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            errors.append(f"Duplicate rows: {duplicate_rows}")
            failed_checks += 1
            logger.warning(f"Duplicate rows: {duplicate_rows}")
        else:
            logger.debug("No duplicate rows check passed")

        # Check 10: No NaN values in critical columns
        total_checks += 1
        critical_cols = ["open", "high", "low", "close", "volume"]
        for col in critical_cols:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    errors.append(f"NaN values in {col}: {nan_count} rows")
                    failed_checks += 1
                    logger.warning(f"NaN values in {col}: {nan_count} rows")

        # Check 11: No infinite values
        total_checks += 1
        for col in critical_cols:
            if col in df.columns:
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    errors.append(f"Infinite values in {col}: {inf_count} rows")
                    failed_checks += 1
                    logger.warning(f"Infinite values in {col}: {inf_count} rows")

        # Check 12: Price values > 0
        total_checks += 1
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                non_positive = (df[col] <= 0).sum()
                if non_positive > 0:
                    errors.append(f"Non-positive values in {col}: {non_positive} rows")
                    failed_checks += 1
                    logger.warning(f"Non-positive values in {col}: {non_positive} rows")

        passed = failed_checks == 0
        if passed:
            logger.info(f"OHLCV validation PASSED: {total_checks}/{total_checks} checks passed")
        else:
            logger.error(f"OHLCV validation FAILED: {failed_checks}/{total_checks} checks failed")

        return self._create_report(passed, total_checks, failed_checks, errors, warnings, df.shape)

    def _create_report(
        self,
        passed: bool,
        total_checks: int,
        failed_checks: int,
        errors: List[str],
        warnings: List[str],
        df_shape: tuple[int, int],
    ) -> ValidationReport:
        """Create validation report."""
        return ValidationReport(
            passed=passed,
            total_checks=total_checks,
            failed_checks=failed_checks,
            errors=errors,
            warnings=warnings,
            timestamp=datetime.now(),
            dataframe_shape=df_shape,
        )


def validate_ohlcv(df: pd.DataFrame, strict: bool = True) -> ValidationReport:
    """
    Convenience function to validate OHLCV data.
    
    Args:
        df: DataFrame with OHLCV columns
        strict: If True, fail on first error
        
    Returns:
        ValidationReport
    """
    validator = OHLCVValidator(strict=strict)
    return validator.validate(df)
