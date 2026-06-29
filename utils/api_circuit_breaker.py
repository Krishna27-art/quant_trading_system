"""
Circuit Breaker Pattern Implementation

Protects against cascading failures when external services (NSE API) are unreliable.
Implements exponential backoff and circuit breaker state machine.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Circuit is tripped, requests fail fast
- HALF_OPEN: Testing if service has recovered

This prevents:
- Cascading failures
- Resource exhaustion
- Unnecessary retries against dead services
"""

import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any

from utils.logger import get_logger
from utils.time_utils import now_ist

logger = get_logger("circuit_breaker")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit tripped, fail fast
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """
    Circuit breaker implementation with exponential backoff.

    Prevents cascading failures by failing fast when external service is down.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Exception = Exception,
        max_backoff: int = 300,
        initial_backoff: int = 1,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to track as failures
            max_backoff: Maximum backoff time in seconds
            initial_backoff: Initial backoff time in seconds
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.max_backoff = max_backoff
        self.initial_backoff = initial_backoff

        self.failure_count = 0
        self.last_failure_time: datetime | None = None
        self.state = CircuitState.CLOSED
        self.lock = threading.Lock()

        logger.info(
            f"Circuit breaker initialized: threshold={failure_threshold}, "
            f"recovery_timeout={recovery_timeout}s, max_backoff={max_backoff}s"
        )

    def _should_attempt_reset(self) -> bool:
        """
        Check if circuit should attempt to reset to HALF_OPEN state.

        Returns:
            True if recovery timeout has passed
        """
        if self.last_failure_time is None:
            return False

        elapsed = now_ist() - self.last_failure_time
        return elapsed.total_seconds() >= self.recovery_timeout

    def _reset_to_half_open(self) -> None:
        """Reset circuit to HALF_OPEN state for recovery testing."""
        self.state = CircuitState.HALF_OPEN
        self.failure_count = 0
        logger.info("Circuit breaker reset to HALF_OPEN state")

    def _trip_circuit(self) -> None:
        """Trip circuit to OPEN state."""
        self.state = CircuitState.OPEN
        self.last_failure_time = now_ist()
        logger.warning(f"Circuit breaker tripped to OPEN state after {self.failure_count} failures")

    def _reset_to_closed(self) -> None:
        """Reset circuit to CLOSED state (recovery successful)."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        logger.info("Circuit breaker reset to CLOSED state (recovery successful)")

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: Original exception or CircuitBreakerOpenError
        """
        with self.lock:
            # Check circuit state
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._reset_to_half_open()
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker is OPEN. "
                        f"Last failure: {self.last_failure_time}, "
                        f"Recovery timeout: {self.recovery_timeout}s"
                    )

        try:
            result = func(*args, **kwargs)

            # Success - reset circuit if in HALF_OPEN
            with self.lock:
                if self.state == CircuitState.HALF_OPEN:
                    self._reset_to_closed()
                elif self.state == CircuitState.CLOSED:
                    self.failure_count = 0  # Reset on success in closed state

            return result

        except self.expected_exception as e:
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = now_ist()

                if self.failure_count >= self.failure_threshold:
                    self._trip_circuit()
                else:
                    logger.warning(
                        f"Circuit breaker failure {self.failure_count}/{self.failure_threshold}: {str(e)}"
                    )

            raise


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is in OPEN state."""

    pass


def exponential_backoff(
    max_retries: int = 3,
    initial_backoff: int = 1,
    max_backoff: int = 300,
    backoff_multiplier: float = 2.0,
):
    """
    Decorator for exponential backoff retry logic.

    Args:
        max_retries: Maximum number of retry attempts
        initial_backoff: Initial backoff time in seconds
        max_backoff: Maximum backoff time in seconds
        backoff_multiplier: Multiplier for backoff time
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_backoff = initial_backoff
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except CircuitBreakerOpenError:
                    # Institutional Fix: DO NOT RETRY IF CIRCUIT IS OPEN - FAIL FAST
                    raise
                except Exception as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries: {str(e)}"
                        )
                        raise

                    # Calculate backoff with exponential increase
                    wait_time = min(current_backoff, max_backoff)

                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}), "
                        f"retrying in {wait_time}s: {str(e)}"
                    )

                    time.sleep(wait_time)
                    current_backoff *= backoff_multiplier

            raise last_exception

        return wrapper

    return decorator


def with_circuit_breaker(circuit_breaker: CircuitBreaker, fallback: Callable | None = None):
    """
    Decorator to apply circuit breaker to a function.

    Args:
        circuit_breaker: Circuit breaker instance
        fallback: Optional fallback function when circuit is open
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return circuit_breaker.call(func, *args, **kwargs)
            except CircuitBreakerOpenError as e:
                if fallback is not None:
                    logger.warning(f"Circuit breaker open, using fallback: {str(e)}")
                    return fallback(*args, **kwargs)
                raise

        return wrapper

    return decorator


# Default circuit breaker instance for NSE API
nse_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60,
    expected_exception=Exception,
    max_backoff=300,
    initial_backoff=1,
)


def nse_api_call_with_circuit_breaker(func: Callable) -> Callable:
    """
    Decorator to apply NSE circuit breaker and exponential backoff.

    Combines circuit breaker protection with exponential backoff retries.
    """

    @exponential_backoff(max_retries=3, initial_backoff=1, max_backoff=300)
    @with_circuit_breaker(nse_circuit_breaker)
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        return func(*args, **kwargs)

    return wrapper


def get_circuit_breaker_state() -> dict:
    """
    Get current circuit breaker state for monitoring.

    Returns:
        Dictionary with circuit breaker state information
    """
    return {
        "state": nse_circuit_breaker.state.value,
        "failure_count": nse_circuit_breaker.failure_count,
        "last_failure_time": (
            nse_circuit_breaker.last_failure_time.isoformat()
            if nse_circuit_breaker.last_failure_time
            else None
        ),
        "failure_threshold": nse_circuit_breaker.failure_threshold,
        "recovery_timeout": nse_circuit_breaker.recovery_timeout,
        "time_until_reset": (
            (
                nse_circuit_breaker.last_failure_time
                + timedelta(seconds=nse_circuit_breaker.recovery_timeout)
                - now_ist()
            ).total_seconds()
            if nse_circuit_breaker.last_failure_time
            and nse_circuit_breaker.state == CircuitState.OPEN
            else 0
        ),
    }


def reset_circuit_breaker() -> None:
    """Reset circuit breaker to CLOSED state manually."""
    nse_circuit_breaker._reset_to_closed()
    logger.info("Circuit breaker manually reset to CLOSED state")
