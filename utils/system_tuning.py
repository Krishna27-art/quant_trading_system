"""
System Tuning Utilities for High-Frequency Trading.
Handles CPU Core Pinning, Garbage Collection control, and Memory Limits.
"""

import gc
import os
import resource

from utils.logger import get_logger

logger = get_logger(__name__)


def enforce_memory_limit(max_memory_mb: int = 4096):
    """
    Enforces a strict upper bound on RAM usage to prevent the OS OOM killer
    from randomly terminating the strategy without a graceful shutdown.
    Raises MemoryError when exceeded, allowing the orchestrator to catch it.
    """
    try:
        max_bytes = max_memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
        logger.info(f"Strict memory limit enforced: {max_memory_mb} MB")
    except Exception as e:
        logger.error(f"Failed to set memory limit: {e}")


def pin_to_cores(cores: list[int]):
    """
    Pins the current process to a specific set of CPU cores.
    Prevents the OS scheduler from context-switching the strategy onto other cores.
    Must be run as root or with CAP_SYS_NICE on Linux.
    """
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(cores))
            logger.info(f"Process {os.getpid()} pinned to CPU cores: {cores}")
        else:
            logger.warning("CPU affinity pinning not supported on this OS.")
    except PermissionError:
        logger.warning(
            f"Permission denied: Could not pin process to cores {cores}. Run with elevated privileges."
        )
    except Exception as e:
        logger.error(f"Failed to pin CPU cores: {e}")


class GCTuning:
    """
    Manages Python Garbage Collection to prevent Stop-The-World pauses
    during critical trading windows.
    """

    @staticmethod
    def enter_trading_window():
        """Disable automatic GC during active high-volatility market hours."""
        gc.disable()
        logger.info("Garbage collection disabled for trading window.")

    @staticmethod
    def exit_trading_window():
        """Re-enable GC and run a manual collection during calm periods."""
        gc.enable()
        collected = gc.collect()
        logger.info(f"Garbage collection re-enabled. Cleaned {collected} objects.")
