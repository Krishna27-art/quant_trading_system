"""
Timezone Validation Module

Ensures all timestamps are handled consistently in UTC internally,
with conversion to IST only at API response and dashboard layers.
Prevents timezone mixing bugs that can cause look-ahead bias.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
import pytz

from utils.logger import get_logger

logger = get_logger("validation.timezones")


# Timezone constants
UTC = pytz.UTC
IST = pytz.timezone("Asia/Kolkata")


@dataclass
class TimezoneReport:
    """Report from timezone validation."""
    passed: bool
    total_checks: int
    failed_checks: int
    errors: List[str]
    warnings: List[str]
    detected_timezone: Optional[str]
    has_timezone_info: bool
    is_utc: bool
    mixed_timezones: bool
    timestamp_count: int

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "passed": self.passed,
            "total_checks": self.total_checks,
            "failed_checks": self.failed_checks,
            "errors": self.errors,
            "warnings": self.warnings,
            "detected_timezone": self.detected_timezone,
            "has_timezone_info": self.has_timezone_info,
            "is_utc": self.is_utc,
            "mixed_timezones": self.mixed_timezones,
            "timestamp_count": self.timestamp_count,
        }


class TimezoneValidator:
    """
    Validates timezone consistency in timestamp data.
    
    Rules:
    1. All internal data should use UTC
    2. Timestamps must have timezone info (not naive)
    3. No mixing of timezones in the same dataset
    4. Conversion to IST only at API/dashboard layer
    """

    def __init__(self, require_utc: bool = True, allow_naive: bool = False):
        """
        Initialize validator.
        
        Args:
            require_utc: If True, require all timestamps to be UTC
            allow_naive: If True, allow naive datetime (no timezone info)
        """
        self.require_utc = require_utc
        self.allow_naive = allow_naive

    def validate(self, df: pd.DataFrame, timestamp_col: str = "timestamp") -> TimezoneReport:
        """
        Validate timezone consistency in DataFrame.
        
        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column
            
        Returns:
            TimezoneReport with validation results
        """
        errors = []
        warnings = []
        total_checks = 0
        failed_checks = 0

        if df.empty:
            errors.append("DataFrame is empty")
            return self._create_report(
                False, total_checks, failed_checks, errors, warnings,
                None, False, False, False, 0
            )

        if timestamp_col not in df.columns:
            errors.append(f"Timestamp column '{timestamp_col}' not found")
            return self._create_report(
                False, total_checks, failed_checks, errors, warnings,
                None, False, False, False, 0
            )

        timestamps = df[timestamp_col]
        timestamp_count = len(timestamps)

        # Check 1: Timestamps have timezone info
        total_checks += 1
        has_tz = False
        if isinstance(timestamps.iloc[0], pd.Timestamp):
            has_tz = timestamps.iloc[0].tzinfo is not None
        elif isinstance(timestamps.iloc[0], datetime):
            has_tz = timestamps.iloc[0].tzinfo is not None

        if not has_tz:
            if not self.allow_naive:
                errors.append("Timestamps are naive (no timezone info)")
                failed_checks += 1
                logger.error("Timestamps are naive - this can cause timezone bugs")
            else:
                warnings.append("Timestamps are naive (no timezone info) - should use UTC")
                logger.warning("Timestamps are naive - should use UTC")
        else:
            logger.debug("Timestamps have timezone info")

        # Check 2: All timestamps use the same timezone
        total_checks += 1
        if has_tz:
            timezones = set()
            for ts in timestamps.head(100):  # Sample first 100 for performance
                if isinstance(ts, pd.Timestamp):
                    tz = ts.tzinfo
                elif isinstance(ts, datetime):
                    tz = ts.tzinfo
                else:
                    continue
                if tz is not None:
                    timezones.add(str(tz))

            if len(timezones) > 1:
                errors.append(f"Mixed timezones detected: {timezones}")
                failed_checks += 1
                logger.error(f"Mixed timezones detected: {timezones}")
                mixed_timezones = True
            else:
                mixed_timezones = False
                logger.debug("All timestamps use consistent timezone")
        else:
            mixed_timezones = False

        # Check 3: Timezone is UTC (if required)
        total_checks += 1
        is_utc = False
        detected_timezone = None
        if has_tz:
            first_ts = timestamps.iloc[0]
            if isinstance(first_ts, pd.Timestamp):
                detected_timezone = str(first_ts.tzinfo)
                is_utc = first_ts.tzinfo == UTC
            elif isinstance(first_ts, datetime):
                detected_timezone = str(first_ts.tzinfo)
                is_utc = first_ts.tzinfo == UTC or first_ts.tzinfo == timezone.utc

            if self.require_utc and not is_utc:
                errors.append(f"Timestamps not in UTC: {detected_timezone}")
                failed_checks += 1
                logger.error(f"Timestamps not in UTC: {detected_timezone}")
            else:
                logger.debug(f"Timestamps in UTC: {detected_timezone}")

        # Check 4: Timestamps are sorted
        total_checks += 1
        if not timestamps.is_monotonic_increasing:
            errors.append("Timestamps not sorted ascending")
            failed_checks += 1
            logger.warning("Timestamps not sorted ascending")
        else:
            logger.debug("Timestamps sorted ascending")

        passed = failed_checks == 0
        if passed:
            logger.info(f"Timezone validation PASSED: {detected_timezone or 'naive'}")
        else:
            logger.error(f"Timezone validation FAILED: {failed_checks}/{total_checks} checks failed")

        return self._create_report(
            passed, total_checks, failed_checks, errors, warnings,
            detected_timezone, has_tz, is_utc, mixed_timezones, timestamp_count
        )

    def _create_report(
        self,
        passed: bool,
        total_checks: int,
        failed_checks: int,
        errors: List[str],
        warnings: List[str],
        detected_timezone: Optional[str],
        has_timezone_info: bool,
        is_utc: bool,
        mixed_timezones: bool,
        timestamp_count: int,
    ) -> TimezoneReport:
        """Create timezone validation report."""
        return TimezoneReport(
            passed=passed,
            total_checks=total_checks,
            failed_checks=failed_checks,
            errors=errors,
            warnings=warnings,
            detected_timezone=detected_timezone,
            has_timezone_info=has_timezone_info,
            is_utc=is_utc,
            mixed_timezones=mixed_timezones,
            timestamp_count=timestamp_count,
        )

    @staticmethod
    def convert_to_utc(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        """
        Convert timestamps to UTC.
        
        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column
            
        Returns:
            DataFrame with UTC timestamps
        """
        df = df.copy()
        
        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found")

        timestamps = df[timestamp_col]
        
        # Handle naive timestamps - assume UTC
        if isinstance(timestamps.iloc[0], pd.Timestamp):
            if timestamps.iloc[0].tzinfo is None:
                logger.warning("Converting naive timestamps to UTC (assumed UTC)")
                df[timestamp_col] = timestamps.dt.tz_localize(UTC)
            else:
                df[timestamp_col] = timestamps.dt.tz_convert(UTC)
        elif isinstance(timestamps.iloc[0], datetime):
            if timestamps.iloc[0].tzinfo is None:
                logger.warning("Converting naive timestamps to UTC (assumed UTC)")
                df[timestamp_col] = timestamps.apply(lambda x: x.replace(tzinfo=UTC))
            else:
                df[timestamp_col] = timestamps.apply(lambda x: x.astimezone(UTC))
        
        logger.info("Converted all timestamps to UTC")
        return df

    @staticmethod
    def convert_to_ist(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        """
        Convert UTC timestamps to IST for API/dashboard display.
        
        Args:
            df: DataFrame with UTC timestamp column
            timestamp_col: Name of timestamp column
            
        Returns:
            DataFrame with IST timestamps
        """
        df = df.copy()
        
        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found")

        timestamps = df[timestamp_col]
        
        # Ensure timestamps are UTC first
        if isinstance(timestamps.iloc[0], pd.Timestamp):
            if timestamps.iloc[0].tzinfo is None:
                logger.warning("Timestamps are naive, assuming UTC before IST conversion")
                df[timestamp_col] = timestamps.dt.tz_localize(UTC).dt.tz_convert(IST)
            else:
                df[timestamp_col] = timestamps.dt.tz_convert(IST)
        elif isinstance(timestamps.iloc[0], datetime):
            if timestamps.iloc[0].tzinfo is None:
                logger.warning("Timestamps are naive, assuming UTC before IST conversion")
                df[timestamp_col] = timestamps.apply(lambda x: x.replace(tzinfo=UTC).astimezone(IST))
            else:
                df[timestamp_col] = timestamps.apply(lambda x: x.astimezone(IST))
        
        logger.info("Converted timestamps to IST for display")
        return df


def validate_timezones(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    require_utc: bool = True,
    allow_naive: bool = False,
) -> TimezoneReport:
    """
    Convenience function to validate timezone consistency.
    
    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column
        require_utc: If True, require UTC
        allow_naive: If True, allow naive timestamps
        
    Returns:
        TimezoneReport
    """
    validator = TimezoneValidator(require_utc=require_utc, allow_naive=allow_naive)
    return validator.validate(df, timestamp_col=timestamp_col)
