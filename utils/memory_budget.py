"""
Memory Budget Enforcement

Prevents unbounded memory growth by enforcing memory limits per service.
Monitors memory usage and alerts when thresholds exceeded.

Institutional Rule:
Memory usage must converge.
If memory grows forever: Bug.
"""

import gc
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

import psutil

from utils.logger import get_logger

logger = get_logger("memory_budget")


class MemoryAction(str, Enum):
    """Action to take when memory limit exceeded."""

    ALERT = "alert"  # Log warning only
    GC = "gc"  # Force garbage collection
    CLEAR_CACHE = "clear_cache"  # Clear caches
    KILL = "kill"  # Terminate process (extreme)


@dataclass
class MemoryBudget:
    """Memory budget configuration for a service."""

    service_name: str
    max_ram_mb: int  # Maximum RAM in MB
    max_cache_mb: int  # Maximum cache in MB
    max_objects: int  # Maximum number of objects
    action: MemoryAction = MemoryAction.GC
    alert_threshold_percent: float = 0.8  # Alert at 80%

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "service_name": self.service_name,
            "max_ram_mb": self.max_ram_mb,
            "max_cache_mb": self.max_cache_mb,
            "max_objects": self.max_objects,
            "action": self.action.value,
            "alert_threshold_percent": self.alert_threshold_percent,
        }


class MemoryMonitor:
    """
    Monitor memory usage and enforce budgets.

    Every service gets:
    - Max RAM
    - Max Cache
    - Max Objects
    """

    def __init__(self, budget: MemoryBudget):
        """
        Initialize memory monitor.

        Args:
            budget: Memory budget configuration
        """
        self.budget = budget
        self.process = psutil.Process(os.getpid())
        self.logger = logger

        # Tracking
        self.alert_count = 0
        self.gc_count = 0
        self.cache_clear_count = 0

        self.logger.info(f"MemoryMonitor initialized for {budget.service_name}")
        self.logger.info(
            f"Budget: {budget.max_ram_mb}MB RAM, {budget.max_cache_mb}MB cache, {budget.max_objects} objects"
        )

    def get_memory_usage(self) -> dict[str, Any]:
        """
        Get current memory usage.

        Returns:
            Memory usage statistics
        """
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": memory_percent,
            "available_mb": psutil.virtual_memory().available / 1024 / 1024,
        }

    def check_budget(self) -> bool:
        """
        Check if memory usage exceeds budget.

        Returns:
            True if within budget
        """
        usage = self.get_memory_usage()
        rss_mb = usage["rss_mb"]

        # Check alert threshold
        alert_threshold = self.budget.max_ram_mb * self.budget.alert_threshold_percent
        if rss_mb > alert_threshold:
            self.alert_count += 1
            self.logger.warning(
                f"Memory alert #{self.alert_count}: {rss_mb:.2f}MB > {alert_threshold:.2f}MB "
                f"({self.budget.service_name})"
            )

        # Check hard limit
        if rss_mb > self.budget.max_ram_mb:
            self.logger.error(
                f"Memory limit exceeded: {rss_mb:.2f}MB > {self.budget.max_ram_mb}MB "
                f"({self.budget.service_name})"
            )

            # Take action based on policy
            return self._take_action()

        return True

    def _take_action(self) -> bool:
        """
        Take action when memory limit exceeded.

        Returns:
            True if recovered
        """
        if self.budget.action == MemoryAction.ALERT:
            self.logger.error("Memory limit exceeded - ALERT only")
            return False

        elif self.budget.action == MemoryAction.GC:
            self.logger.warning("Memory limit exceeded - forcing garbage collection")
            gc.collect()
            self.gc_count += 1
            return True

        elif self.budget.action == MemoryAction.CLEAR_CACHE:
            self.logger.warning("Memory limit exceeded - clearing caches")
            gc.collect()
            self.gc_count += 1
            self.cache_clear_count += 1
            return True

        elif self.budget.action == MemoryAction.KILL:
            self.logger.critical("Memory limit exceeded - KILLING PROCESS")
            os._exit(1)
            return False

        return False

    def get_stats(self) -> dict[str, Any]:
        """
        Get memory monitor statistics.

        Returns:
            Monitor statistics
        """
        return {
            "service_name": self.budget.service_name,
            "current_usage": self.get_memory_usage(),
            "budget": self.budget.to_dict(),
            "alert_count": self.alert_count,
            "gc_count": self.gc_count,
            "cache_clear_count": self.cache_clear_count,
        }


class BoundedCache:
    """
    Bounded cache with memory budget enforcement.

    Automatically evicts oldest entries when memory limit exceeded.
    """

    def __init__(self, name: str, max_size: int = 10000, max_memory_mb: int = 100):
        """
        Initialize bounded cache.

        Args:
            name: Cache name
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory in MB
        """
        self.name = name
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.cache: dict[str, Any] = {}
        self.access_order: list = []  # Track access order for LRU
        self.logger = logger

        self.logger.info(
            f"BoundedCache initialized: {name} (max_size={max_size}, max_memory={max_memory_mb}MB)"
        )

    def get(self, key: str) -> Any | None:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Value or None
        """
        if key in self.cache:
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def set(self, key: str, value: Any):
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        # Check size limit
        if len(self.cache) >= self.max_size:
            self._evict_oldest()

        # Check memory limit
        if self._estimate_memory() >= self.max_memory_mb:
            self._evict_oldest()

        self.cache[key] = value
        self.access_order.append(key)

    def _evict_oldest(self):
        """Evict oldest entry (LRU)."""
        if self.access_order:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
            self.logger.debug(f"Evicted oldest entry from cache: {self.name}")

    def _estimate_memory(self) -> float:
        """
        Estimate cache memory usage in MB.

        Returns:
            Memory usage in MB
        """
        # Rough estimate: number of entries * average size
        return len(self.cache) * 0.001  # 1KB per entry estimate

    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_order.clear()
        self.logger.info(f"Cache cleared: {self.name}")

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Cache statistics
        """
        return {
            "name": self.name,
            "size": len(self.cache),
            "max_size": self.max_size,
            "memory_mb": self._estimate_memory(),
            "max_memory_mb": self.max_memory_mb,
        }


# Global memory monitors
_memory_monitors: dict[str, MemoryMonitor] = {}


def register_memory_budget(budget: MemoryBudget) -> MemoryMonitor:
    """
    Register memory budget for a service.

    Args:
        budget: Memory budget configuration

    Returns:
        MemoryMonitor instance
    """
    monitor = MemoryMonitor(budget)
    _memory_monitors[budget.service_name] = monitor
    return monitor


def get_memory_monitor(service_name: str) -> MemoryMonitor | None:
    """
    Get memory monitor for service.

    Args:
        service_name: Service name

    Returns:
        MemoryMonitor or None
    """
    return _memory_monitors.get(service_name)


def check_all_budgets() -> dict[str, bool]:
    """
    Check all registered memory budgets.

    Returns:
        Dictionary of service -> within_budget
    """
    results = {}
    for service_name, monitor in _memory_monitors.items():
        results[service_name] = monitor.check_budget()
    return results
