"""
API Rate Limiter

Institutional-grade rate limiting for API calls.
Prevents API throttling and ensures compliance with rate limits.
"""

import threading
import time
from collections import defaultdict, deque
from datetime import timedelta

from utils.logger import get_logger
from utils.time_utils import now_ist

logger = get_logger("rate_limiter", pipeline_id="ingestion_rate_limiter")


class RateLimiter:
    """
    Token bucket rate limiter for API calls.

    Implements token bucket algorithm for rate limiting.
    """

    def __init__(self, max_calls: int, time_window_seconds: int):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum number of calls allowed in time window
            time_window_seconds: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window_seconds = time_window_seconds
        self.logger = logger

        # Track calls per endpoint
        self.calls: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def acquire(self, endpoint: str) -> bool:
        """
        Acquire permission to make API call.

        Args:
            endpoint: API endpoint identifier

        Returns:
            True if call is allowed, False otherwise
        """
        with self._lock:
            now = now_ist()
            cutoff = now - timedelta(seconds=self.time_window_seconds)

            # Clean up old calls
            while self.calls[endpoint] and self.calls[endpoint][0] < cutoff:
                self.calls[endpoint].popleft()

            # Check if limit exceeded
            if len(self.calls[endpoint]) >= self.max_calls:
                self.logger.warning(f"Rate limit exceeded for {endpoint}")
                return False

            # Add current call
            self.calls[endpoint].append(now)
            return True

    def wait_if_needed(self, endpoint: str) -> None:
        """
        Wait if rate limit would be exceeded.

        Args:
            endpoint: API endpoint identifier
        """
        if not self.acquire(endpoint):
            # Calculate wait time
            oldest_call = self.calls[endpoint][0]
            wait_time = (oldest_call + timedelta(seconds=self.time_window_seconds)) - now_ist()
            wait_seconds = max(0, wait_time.total_seconds())

            self.logger.info(f"Rate limiting: waiting {wait_seconds:.2f}s for {endpoint}")
            time.sleep(wait_seconds)

            # Retry after waiting
            self.acquire(endpoint)

    def get_remaining_calls(self, endpoint: str) -> int:
        """
        Get remaining calls for endpoint.

        Args:
            endpoint: API endpoint identifier

        Returns:
            Number of remaining calls
        """
        now = now_ist()
        cutoff = now - timedelta(seconds=self.time_window_seconds)

        # Clean up old calls
        while self.calls[endpoint] and self.calls[endpoint][0] < cutoff:
            self.calls[endpoint].popleft()

        return self.max_calls - len(self.calls[endpoint])


class NSERateLimiter(RateLimiter):
    """
    Rate limiter specifically for NSE APIs.

    NSE rate limits:
    - Equity history: ~10 calls per minute
    - Options chain: ~5 calls per minute
    - Corporate actions: ~10 calls per minute
    """

    def __init__(self):
        """Initialize NSE rate limiter with conservative limits."""
        super().__init__(
            max_calls=5,
            time_window_seconds=60,  # Conservative limit  # 1 minute window
        )
        self.logger = logger

    def acquire_equity_history(self) -> bool:
        """Acquire permission for equity history call."""
        return self.acquire("equity_history")

    def acquire_options_chain(self) -> bool:
        """Acquire permission for options chain call."""
        return self.acquire("options_chain")

    def acquire_corporate_actions(self) -> bool:
        """Acquire permission for corporate actions call."""
        return self.acquire("corporate_actions")

    def wait_if_needed_equity_history(self) -> None:
        """Wait if equity history rate limit would be exceeded."""
        self.wait_if_needed("equity_history")

    def wait_if_needed_options_chain(self) -> None:
        """Wait if options chain rate limit would be exceeded."""
        self.wait_if_needed("options_chain")

    def wait_if_needed_corporate_actions(self) -> None:
        """Wait if corporate actions rate limit would be exceeded."""
        self.wait_if_needed("corporate_actions")


# Global rate limiter instance
_nse_rate_limiter = None


def get_nse_rate_limiter() -> NSERateLimiter:
    """
    Get global NSE rate limiter instance.

    Returns:
        NSERateLimiter instance
    """
    global _nse_rate_limiter
    if _nse_rate_limiter is None:
        _nse_rate_limiter = NSERateLimiter()
    return _nse_rate_limiter
