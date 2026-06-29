"""
Market Calendar

Institutional-grade market calendar with special session handling.
Supports expiry shifts, half-days, Muhurat trading, and holiday revisions.
"""

from datetime import datetime, time, timedelta
from enum import Enum

import duckdb

from config.settings import DB_PATH
from utils.logger import get_logger

logger = get_logger("market_calendar")


class SessionType(str, Enum):
    """Types of trading sessions."""

    NORMAL = "NORMAL"
    PRE_OPEN = "PRE_OPEN"
    POST_CLOSE = "POST_CLOSE"
    HALF_DAY = "HALF_DAY"
    MUHURAT = "MUHURAT"
    SPECIAL = "SPECIAL"


class HolidayType(str, Enum):
    """Types of holidays."""

    WEEKEND = "WEEKEND"
    TRADING_HOLIDAY = "TRADING_HOLIDAY"
    CLEARING_HOLIDAY = "CLEARING_HOLIDAY"
    SETTLEMENT_HOLIDAY = "SETTLEMENT_HOLIDAY"


class MarketCalendar:
    """
    Institutional-grade market calendar.

    Handles special sessions, expiry shifts, holiday revisions, and half-days.
    """

    # Standard trading hours
    NORMAL_START = time(9, 15)  # 9:15 AM
    NORMAL_END = time(15, 30)  # 3:30 PM
    PRE_OPEN_START = time(9, 0)  # 9:00 AM
    PRE_OPEN_END = time(9, 8)  # 9:08 AM
    POST_CLOSE_START = time(15, 40)  # 3:40 PM
    POST_CLOSE_END = time(16, 0)  # 4:00 PM

    # Half-day trading hours
    HALF_DAY_START = time(9, 15)
    HALF_DAY_END = time(13, 30)  # 1:30 PM

    def __init__(self):
        """Initialize the market calendar."""
        self.db_path = DB_PATH
        self.logger = logger

    def is_trading_day(self, date: datetime) -> bool:
        """
        Check if a date is a trading day.

        Args:
            date: Date to check

        Returns:
            True if trading day, False otherwise
        """
        try:
            conn = duckdb.connect(str(self.db_path))

            query = """
                SELECT is_trading_day FROM market_calendar
                WHERE date = ?
            """

            result = conn.execute(query, [date.date()]).fetchone()
            conn.close()

            if result:
                return result[0]

            # Default: check if weekend
            return date.weekday() < 5

        except Exception as e:
            self.logger.error(f"Failed to check trading day: {str(e)}")
            return date.weekday() < 5

    def get_session_hours(self, date: datetime) -> dict[str, time]:
        """
        Get trading session hours for a date.

        Args:
            date: Date to get session hours for

        Returns:
            Dictionary with session start and end times
        """
        try:
            conn = duckdb.connect(str(self.db_path))

            query = """
                SELECT session_type, session_start, session_end
                FROM market_calendar
                WHERE date = ?
            """

            result = conn.execute(query, [date.date()]).fetchone()
            conn.close()

            if result:
                session_type = result[0]
                if session_type == SessionType.HALF_DAY.value:
                    return {
                        "start": self.HALF_DAY_START,
                        "end": self.HALF_DAY_END,
                        "session_type": SessionType.HALF_DAY,
                    }
                elif session_type == SessionType.MUHURAT.value:
                    # Muhurat trading typically 6:15 PM to 7:15 PM
                    return {
                        "start": time(18, 15),
                        "end": time(19, 15),
                        "session_type": SessionType.MUHURAT,
                    }

            return {
                "start": self.NORMAL_START,
                "end": self.NORMAL_END,
                "session_type": SessionType.NORMAL,
            }

        except Exception as e:
            self.logger.error(f"Failed to get session hours: {str(e)}")
            return {
                "start": self.NORMAL_START,
                "end": self.NORMAL_END,
                "session_type": SessionType.NORMAL,
            }

    def is_expiry_day(self, date: datetime) -> bool:
        """
        Check if a date is an expiry day.

        Args:
            date: Date to check

        Returns:
            True if expiry day, False otherwise
        """
        try:
            conn = duckdb.connect(str(self.db_path))

            query = """
                SELECT is_expiry FROM market_calendar
                WHERE date = ?
            """

            result = conn.execute(query, [date.date()]).fetchone()
            conn.close()

            if result:
                return result[0]

            # Fallback: calculate last Thursday of month
            if date.weekday() == 3:  # Thursday
                next_thursday = date + timedelta(days=7)
                if next_thursday.month != date.month:
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to check expiry day: {str(e)}")
            return False

    def get_expiry_shift(self, original_date: datetime) -> datetime | None:
        """
        Get shifted expiry date if original date is a holiday.

        Args:
            original_date: Original expiry date

        Returns:
            Shifted expiry date or None if no shift
        """
        try:
            conn = duckdb.connect(str(self.db_path))

            query = """
                SELECT shifted_expiry_date FROM market_calendar
                WHERE date = ? AND shifted_expiry_date IS NOT NULL
            """

            result = conn.execute(query, [original_date.date()]).fetchone()
            conn.close()

            if result:
                return datetime.combine(result[0], self.NORMAL_END)

            # Check if original date is a holiday
            if not self.is_trading_day(original_date):
                # Shift to previous trading day
                shifted = original_date - timedelta(days=1)
                while not self.is_trading_day(shifted):
                    shifted -= timedelta(days=1)
                return shifted

            return None

        except Exception as e:
            self.logger.error(f"Failed to get expiry shift: {str(e)}")
            return None

    def get_holiday_info(self, date: datetime) -> dict | None:
        """
        Get holiday information for a date.

        Args:
            date: Date to check

        Returns:
            Dictionary with holiday information or None
        """
        try:
            conn = duckdb.connect(str(self.db_path))

            query = """
                SELECT
                    is_holiday,
                    holiday_name,
                    holiday_type,
                    is_revised,
                    original_holiday_date,
                    revision_reason
                FROM market_calendar
                WHERE date = ? AND is_holiday = TRUE
            """

            result = conn.execute(query, [date.date()]).fetchone()
            conn.close()

            if result:
                return {
                    "is_holiday": result[0],
                    "holiday_name": result[1],
                    "holiday_type": result[2],
                    "is_revised": result[3],
                    "original_holiday_date": result[4],
                    "revision_reason": result[5],
                }

            return None

        except Exception as e:
            self.logger.error(f"Failed to get holiday info: {str(e)}")
            return None

    def get_trading_days(self, start_date: datetime, end_date: datetime) -> list[datetime]:
        """
        Get list of trading days in a date range.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            List of trading days
        """
        try:
            conn = duckdb.connect(str(self.db_path))

            query = """
                SELECT date FROM market_calendar
                WHERE date >= ? AND date <= ? AND is_trading_day = TRUE
                ORDER BY date ASC
            """

            df = conn.execute(query, [start_date.date(), end_date.date()]).df()
            conn.close()

            return [datetime.combine(row["date"], time(0, 0)) for _, row in df.iterrows()]

        except Exception as e:
            self.logger.error(f"Failed to get trading days: {str(e)}")
            return []

    def get_expiry_days(self, start_date: datetime, end_date: datetime) -> list[datetime]:
        """
        Get list of expiry days in a date range.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            List of expiry days
        """
        try:
            conn = duckdb.connect(str(self.db_path))

            query = """
                SELECT date FROM market_calendar
                WHERE date >= ? AND date <= ? AND is_expiry = TRUE
                ORDER BY date ASC
            """

            df = conn.execute(query, [start_date.date(), end_date.date()]).df()
            conn.close()

            return [datetime.combine(row["date"], time(0, 0)) for _, row in df.iterrows()]

        except Exception as e:
            self.logger.error(f"Failed to get expiry days: {str(e)}")
            return []

    def get_half_days(self, start_date: datetime, end_date: datetime) -> list[datetime]:
        """
        Get list of half-day trading sessions in a date range.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            List of half-day dates
        """
        try:
            conn = duckdb.connect(str(self.db_path))

            query = """
                SELECT date FROM market_calendar
                WHERE date >= ? AND date <= ? AND session_type = ?
                ORDER BY date ASC
            """

            df = conn.execute(
                query, [start_date.date(), end_date.date(), SessionType.HALF_DAY.value]
            ).df()
            conn.close()

            return [datetime.combine(row["date"], time(0, 0)) for _, row in df.iterrows()]

        except Exception as e:
            self.logger.error(f"Failed to get half days: {str(e)}")
            return []

    def get_muhurat_sessions(self, year: int) -> dict | None:
        """
        Get Muhurat trading session information for a year.

        Args:
            year: Year to get Muhurat session for

        Returns:
            Dictionary with Muhurat session information
        """
        try:
            conn = duckdb.connect(str(self.db_path))

            query = """
                SELECT date, session_start, session_end, description
                FROM market_calendar
                WHERE session_type = ? AND EXTRACT(YEAR FROM date) = ?
            """

            result = conn.execute(query, [SessionType.MUHURAT.value, year]).fetchone()
            conn.close()

            if result:
                return {
                    "date": result[0],
                    "session_start": result[1],
                    "session_end": result[2],
                    "description": result[3],
                }

            return None

        except Exception as e:
            self.logger.error(f"Failed to get Muhurat session: {str(e)}")
            return None

    def add_holiday_revision(
        self, original_date: datetime, new_date: datetime, reason: str
    ) -> None:
        """
        Add a holiday revision (when a holiday date is changed).

        Args:
            original_date: Originally announced holiday date
            new_date: Revised holiday date
            reason: Reason for revision
        """
        try:
            conn = duckdb.connect(str(self.db_path))

            # Update original date
            conn.execute(
                """
                UPDATE market_calendar
                SET is_holiday = FALSE,
                holiday_name = NULL,
                is_revised = TRUE,
                revision_reason = ?
                WHERE date = ?
            """,
                [reason, original_date.date()],
            )

            # Update new date
            conn.execute(
                """
                UPDATE market_calendar
                SET is_holiday = TRUE,
                holiday_name = 'Revised Holiday',
                is_revised = TRUE,
                original_holiday_date = ?,
                revision_reason = ?
                WHERE date = ?
            """,
                [original_date.date(), reason, new_date.date()],
            )

            conn.close()

            self.logger.info(f"Added holiday revision: {original_date.date()} -> {new_date.date()}")

        except Exception as e:
            self.logger.error(f"Failed to add holiday revision: {str(e)}")
            raise

    def calculate_next_expiry(self, from_date: datetime) -> datetime:
        """
        Calculate the next expiry date from a given date.

        Args:
            from_date: Date to calculate from

        Returns:
            Next expiry date
        """
        # Start from next day
        current = from_date + timedelta(days=1)

        while True:
            # Check if it's a Thursday
            if current.weekday() == 3:
                # Check if it's the last Thursday of the month
                next_thursday = current + timedelta(days=7)
                if next_thursday.month != current.month:
                    # Check if it's a trading day
                    if self.is_trading_day(current):
                        return current
                    else:
                        # Get shifted expiry
                        shifted = self.get_expiry_shift(current)
                        if shifted:
                            return shifted

            current += timedelta(days=1)

            # Safety check
            if current > from_date + timedelta(days=60):
                self.logger.warning(f"Could not find expiry date within 60 days from {from_date}")
                return current
