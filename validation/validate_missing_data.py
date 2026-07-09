"""
Missing Data Detection Module

Detects missing candles, gaps, and irregularities in OHLCV time series data.
Handles weekend gaps, market holidays, and trading hour validation.
"""

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import List, Optional, Set

import pandas as pd
import numpy as np

from utils.logger import get_logger

logger = get_logger("validation.missing_data")


@dataclass
class GapReport:
    """Report of detected gaps in time series."""
    missing_timestamps: List[datetime]
    duplicate_timestamps: List[datetime]
    unexpected_gaps: List[tuple[datetime, datetime, timedelta]]
    weekend_gaps: List[tuple[datetime, datetime, timedelta]]
    holiday_gaps: List[tuple[datetime, datetime, timedelta]]
    trading_hour_violations: List[datetime]
    total_expected_candles: int
    total_actual_candles: int
    completeness_pct: float

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "missing_timestamps": [ts.isoformat() for ts in self.missing_timestamps],
            "duplicate_timestamps": [ts.isoformat() for ts in self.duplicate_timestamps],
            "unexpected_gaps": [
                (start.isoformat(), end.isoformat(), gap.total_seconds())
                for start, end, gap in self.unexpected_gaps
            ],
            "weekend_gaps": [
                (start.isoformat(), end.isoformat(), gap.total_seconds())
                for start, end, gap in self.weekend_gaps
            ],
            "holiday_gaps": [
                (start.isoformat(), end.isoformat(), gap.total_seconds())
                for start, end, gap in self.holiday_gaps
            ],
            "trading_hour_violations": [ts.isoformat() for ts in self.trading_hour_violations],
            "total_expected_candles": self.total_expected_candles,
            "total_actual_candles": self.total_actual_candles,
            "completeness_pct": round(self.completeness_pct, 2),
        }


class MissingDataValidator:
    """
    Validates time series data for missing candles and gaps.
    
    Supports multiple intervals:
    - 1m, 5m, 15m, 30m (intraday)
    - 1h (hourly)
    - 1d (daily)
    - 1w (weekly)
    
    Handles:
    - Weekend gaps (expected for daily data)
    - Market holidays (configurable)
    - Trading hour violations (for intraday data)
    """

    # NSE trading hours (IST)
    MARKET_OPEN = time(9, 15)
    MARKET_CLOSE = time(15, 30)
    
    # Interval mappings to timedelta
    INTERVAL_DELTAS = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30),
        "1h": timedelta(hours=1),
        "1d": timedelta(days=1),
        "1w": timedelta(weeks=1),
    }

    def __init__(
        self,
        interval: str = "1d",
        holidays: Optional[Set[datetime]] = None,
        timezone: str = "Asia/Kolkata",
    ):
        """
        Initialize validator.
        
        Args:
            interval: Data interval (1m, 5m, 15m, 30m, 1h, 1d, 1w)
            holidays: Set of holiday dates to ignore
            timezone: Timezone for trading hour validation
        """
        self.interval = interval
        self.holidays = holidays or set()
        self.timezone = timezone
        
        if interval not in self.INTERVAL_DELTAS:
            raise ValueError(f"Unsupported interval: {interval}. Supported: {list(self.INTERVAL_DELTAS.keys())}")

    def validate(self, df: pd.DataFrame) -> GapReport:
        """
        Validate DataFrame for missing data and gaps.
        
        Args:
            df: DataFrame with 'timestamp' index or column
            
        Returns:
            GapReport with detected issues
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to MissingDataValidator")
            return GapReport(
                missing_timestamps=[],
                duplicate_timestamps=[],
                unexpected_gaps=[],
                weekend_gaps=[],
                holiday_gaps=[],
                trading_hour_violations=[],
                total_expected_candles=0,
                total_actual_candles=0,
                completeness_pct=0.0,
            )

        # Ensure timestamp is index
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have 'timestamp' column or DatetimeIndex")

        # Sort by timestamp
        df = df.sort_index()

        # Detect duplicates
        duplicate_timestamps = self._detect_duplicates(df)

        # Detect missing timestamps
        missing_timestamps = self._detect_missing_timestamps(df)

        # Categorize gaps
        unexpected_gaps, weekend_gaps, holiday_gaps = self._categorize_gaps(df)

        # Check trading hours for intraday data
        trading_hour_violations = []
        if self.interval in ["1m", "5m", "15m", "30m", "1h"]:
            trading_hour_violations = self._check_trading_hours(df)

        # Calculate expected vs actual
        if len(df) > 0:
            start_date = df.index[0]
            end_date = df.index[-1]
            expected_count = self._calculate_expected_count(start_date, end_date)
            actual_count = len(df)
            completeness_pct = (actual_count / expected_count * 100) if expected_count > 0 else 0.0
        else:
            expected_count = 0
            actual_count = 0
            completeness_pct = 0.0

        logger.info(
            f"Missing data validation: {len(missing_timestamps)} missing, "
            f"{len(duplicate_timestamps)} duplicates, {len(unexpected_gaps)} unexpected gaps, "
            f"completeness: {completeness_pct:.1f}%"
        )

        return GapReport(
            missing_timestamps=missing_timestamps,
            duplicate_timestamps=duplicate_timestamps,
            unexpected_gaps=unexpected_gaps,
            weekend_gaps=weekend_gaps,
            holiday_gaps=holiday_gaps,
            trading_hour_violations=trading_hour_violations,
            total_expected_candles=expected_count,
            total_actual_candles=actual_count,
            completeness_pct=completeness_pct,
        )

    def _detect_duplicates(self, df: pd.DataFrame) -> List[datetime]:
        """Detect duplicate timestamps."""
        duplicates = df.index[df.index.duplicated()].tolist()
        if duplicates:
            logger.warning(f"Found {len(duplicates)} duplicate timestamps")
        return duplicates

    def _detect_missing_timestamps(self, df: pd.DataFrame) -> List[datetime]:
        """Detect missing timestamps in the expected sequence."""
        if len(df) < 2:
            return []

        expected_delta = self.INTERVAL_DELTAS[self.interval]
        expected_range = pd.date_range(
            start=df.index[0],
            end=df.index[-1],
            freq=expected_delta,
        )

        missing = expected_range.difference(df.index)
        return missing.tolist()

    def _categorize_gaps(
        self, df: pd.DataFrame
    ) -> tuple[List[tuple[datetime, datetime, timedelta]], List[tuple[datetime, datetime, timedelta]], List[tuple[datetime, datetime, timedelta]]]:
        """Categorize gaps as unexpected, weekend, or holiday."""
        if len(df) < 2:
            return [], [], []

        unexpected_gaps = []
        weekend_gaps = []
        holiday_gaps = []

        for i in range(1, len(df)):
            prev_ts = df.index[i - 1]
            curr_ts = df.index[i]
            gap = curr_ts - prev_ts
            expected_delta = self.INTERVAL_DELTAS[self.interval]

            # Allow small tolerance for minor timing issues
            tolerance = timedelta(seconds=30)
            if gap <= expected_delta + tolerance:
                continue

            # Check if gap spans weekend
            if self._is_weekend_gap(prev_ts, curr_ts):
                weekend_gaps.append((prev_ts, curr_ts, gap))
            # Check if gap spans holiday
            elif self._is_holiday_gap(prev_ts, curr_ts):
                holiday_gaps.append((prev_ts, curr_ts, gap))
            else:
                unexpected_gaps.append((prev_ts, curr_ts, gap))

        return unexpected_gaps, weekend_gaps, holiday_gaps

    def _is_weekend_gap(self, start: datetime, end: datetime) -> bool:
        """Check if gap spans weekend (Saturday/Sunday)."""
        # For daily data, check if any day between is weekend
        if self.interval == "1d":
            current = start + timedelta(days=1)
            while current < end:
                if current.weekday() >= 5:  # Saturday=5, Sunday=6
                    return True
                current += timedelta(days=1)
        return False

    def _is_holiday_gap(self, start: datetime, end: datetime) -> bool:
        """Check if gap spans any configured holiday."""
        current = start + timedelta(days=1)
        while current < end:
            if current.date() in {h.date() for h in self.holidays}:
                return True
            current += timedelta(days=1)
        return False

    def _check_trading_hours(self, df: pd.DataFrame) -> List[datetime]:
        """Check if timestamps fall within trading hours (for intraday data)."""
        violations = []
        
        for ts in df.index:
            ts_time = ts.time()
            # Check if timestamp is outside trading hours
            if ts_time < self.MARKET_OPEN or ts_time > self.MARKET_CLOSE:
                # Skip if it's a weekend
                if ts.weekday() < 5:  # Monday=0, Friday=4
                    violations.append(ts)
        
        if violations:
            logger.warning(f"Found {len(violations)} trading hour violations")
        
        return violations

    def _calculate_expected_count(self, start: datetime, end: datetime) -> int:
        """Calculate expected number of candles for the date range."""
        delta = end - start
        interval_delta = self.INTERVAL_DELTAS[self.interval]
        
        # Basic calculation
        expected = int(delta / interval_delta) + 1
        
        # Adjust for weekends (daily data only)
        if self.interval == "1d":
            # Count weekends in range
            weekends = 0
            current = start
            while current <= end:
                if current.weekday() >= 5:
                    weekends += 1
                current += timedelta(days=1)
            expected -= weekends
        
        # Adjust for holidays
        holiday_count = sum(1 for h in self.holidays if start <= h <= end)
        expected -= holiday_count
        
        return max(0, expected)


def validate_missing_data(
    df: pd.DataFrame,
    interval: str = "1d",
    holidays: Optional[Set[datetime]] = None,
) -> GapReport:
    """
    Convenience function to validate missing data.
    
    Args:
        df: DataFrame with timestamp column or index
        interval: Data interval
        holidays: Set of holiday dates
        
    Returns:
        GapReport
    """
    validator = MissingDataValidator(interval=interval, holidays=holidays)
    return validator.validate(df)
