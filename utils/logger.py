"""
Centralized Logging Module

Provides institutional-grade structured logging with JSON output.
All log events include run_id, pipeline_id, dataset_id, symbol, and latency_ms.
"""

from config.settings import LOG_DIR
from utils.structured_logger import get_structured_logger


def get_logger(
    name: str,
    pipeline_id: str | None = None,
    dataset_id: str | None = None,
    symbol: str | None = None,
):
    """
    Get or create a structured logger with the specified name.

    Args:
        name: Name of the logger (typically module name)
        pipeline_id: Pipeline identifier
        dataset_id: Dataset identifier
        symbol: Stock symbol (if applicable)

    Returns:
        Configured StructuredLogger instance with JSON output
    """
    # Create log directory if it doesn't exist
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Return structured logger
    return get_structured_logger(name)
