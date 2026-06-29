"""
Time Utilities

Centralized utility for generating timezone-aware datetimes.
Hardened for the Indian market (Asia/Kolkata).
Includes Clock Sync and PTP monotonic tracking utility.
"""

import threading
import time
from datetime import datetime

import pytz

# Strictly enforce Indian Standard Time
IST = pytz.timezone("Asia/Kolkata")


def now_ist() -> datetime:
    """
    Get current time strictly localized to IST.
    Replaces naive datetime.now() across the system.

    Returns:
        timezone-aware datetime object in Asia/Kolkata
    """
    return datetime.now(IST)


class MonotonicClockSync:
    """
    Tracks monotonic time and system time to detect clock drift, NTP adjustments,
    and simulate Precision Time Protocol (PTP) sync status.
    Essential for high-frequency trading where time jumps can corrupt latency measurements.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init()
            return cls._instance

    def _init(self):
        self.base_monotonic = time.monotonic()
        self.base_time = time.time()
        self.max_drift_ns = 50_000  # 50 microseconds
        self.last_sync_time = self.base_time

    def sync(self):
        """Re-synchronizes base times and calculates drift."""
        current_mono = time.monotonic()
        current_time = time.time()

        expected_time = self.base_time + (current_mono - self.base_monotonic)
        drift = current_time - expected_time

        self.base_monotonic = current_mono
        self.base_time = current_time
        self.last_sync_time = current_time

        return drift

    def get_synced_time(self) -> float:
        """Returns highly accurate time using monotonic progression."""
        mono_delta = time.monotonic() - self.base_monotonic
        return self.base_time + mono_delta

    def is_clock_stable(self) -> bool:
        """Checks if recent drift is within acceptable PTP limits."""
        drift = self.sync()
        return abs(drift * 1e9) <= self.max_drift_ns


# Global clock sync instance
clock_sync = MonotonicClockSync()
