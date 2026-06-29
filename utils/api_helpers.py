"""
API Helper Utilities

Provides retry logic with exponential backoff and circuit breaker protection for NSE API calls.
Designed for large-scale ingestion with resilience against random NSE failures.
"""

import random
import time
from collections.abc import Callable
from functools import wraps

from config.settings import (
    NSE_API_RATE_LIMIT_MAX,
    NSE_API_RATE_LIMIT_MIN,
    NSE_API_RETRY_ATTEMPTS,
    NSE_API_RETRY_DELAY,
)
from utils.api_circuit_breaker import (
    exponential_backoff,
    get_circuit_breaker_state,
    nse_circuit_breaker,
    with_circuit_breaker,
)
from utils.logger import get_logger

logger = get_logger("api_helpers")


def rate_limit(func: Callable) -> Callable:
    """
    Decorator to add rate limiting to API calls.

    Args:
        func: Function to decorate

    Returns:
        Decorated function with rate limiting
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Sleep for random duration to avoid rate limiting
        sleep_time = random.uniform(NSE_API_RATE_LIMIT_MIN, NSE_API_RATE_LIMIT_MAX)
        time.sleep(sleep_time)
        return func(*args, **kwargs)

    return wrapper


def nse_api_retry(func: Callable) -> Callable:
    """
    Decorator to add exponential backoff retry logic to NSE API calls.

    Replaces fixed wait with exponential backoff for better resilience under load.

    Args:
        func: Function to decorate

    Returns:
        Decorated function with exponential backoff retry logic
    """

    @exponential_backoff(
        max_retries=NSE_API_RETRY_ATTEMPTS,
        initial_backoff=NSE_API_RETRY_DELAY,
        max_backoff=300,
        backoff_multiplier=2.0,
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def nse_api_call(func: Callable) -> Callable:
    """
    Decorator that combines rate limiting, exponential backoff, and circuit breaker protection.

    This provides comprehensive resilience for large-scale NSE API ingestion:
    - Rate limiting to avoid API throttling
    - Exponential backoff for transient failures
    - Circuit breaker to prevent cascading failures when NSE is down

    Args:
        func: Function to decorate

    Returns:
        Decorated function with rate limiting, exponential backoff, and circuit breaker
    """

    @rate_limit
    @nse_api_retry
    @with_circuit_breaker(nse_circuit_breaker)
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def log_circuit_breaker_state() -> None:
    """
    Log current circuit breaker state for monitoring.
    """
    state = get_circuit_breaker_state()
    logger.info(f"Circuit Breaker State: {state}")
