"""
History Persistence to Disk

Persists historical data to disk to prevent unbounded memory growth.
Keeps only working window in memory.

Architecture:
Old Data → Disk
Working Window → Memory
"""

import gzip
import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.logger import get_logger

logger = get_logger("history_persistence")


@dataclass
class PersistenceConfig:
    """Configuration for history persistence."""

    base_path: str = "data/history"
    max_memory_items: int = 10000  # Keep only last N items in memory
    compress: bool = True  # Use gzip compression
    format: str = "pickle"  # pickle or json

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "base_path": self.base_path,
            "max_memory_items": self.max_memory_items,
            "compress": self.compress,
            "format": self.format,
        }


class HistoryPersistence:
    """
    Persist historical data to disk.

    Old data → Disk
    Working window → Memory
    """

    def __init__(self, name: str, config: PersistenceConfig | None = None):
        """
        Initialize history persistence.

        Args:
            name: History name (e.g., "orders", "fills", "metrics")
            config: Persistence configuration
        """
        self.name = name
        self.config = config or PersistenceConfig()
        self.logger = logger

        # Create base directory
        self.base_path = Path(self.config.base_path) / name
        self.base_path.mkdir(parents=True, exist_ok=True)

        # In-memory working window
        self._memory_window: list[Any] = []
        self._disk_count = 0

        self.logger.info(f"HistoryPersistence initialized: {name}")
        self.logger.info(f"Base path: {self.base_path}")
        self.logger.info(f"Max memory items: {self.config.max_memory_items}")

    def add(self, item: Any):
        """
        Add item to history.

        Args:
            item: Item to persist
        """
        self._memory_window.append(item)

        # Check if we need to persist to disk
        if len(self._memory_window) > self.config.max_memory_items:
            self._persist_to_disk()

    def _persist_to_disk(self):
        """Persist oldest items to disk."""
        # Calculate how many items to persist
        items_to_persist = len(self._memory_window) - self.config.max_memory_items
        if items_to_persist <= 0:
            return

        # Get oldest items
        items_to_save = self._memory_window[:items_to_persist]

        # Remove from memory
        self._memory_window = self._memory_window[items_to_persist:]

        # Save to disk
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.name}_{timestamp}_{self._disk_count}"

        if self.config.format == "pickle":
            self._save_pickle(items_to_save, filename)
        else:
            self._save_json(items_to_save, filename)

        self._disk_count += 1
        self.logger.info(f"Persisted {len(items_to_save)} items to disk: {filename}")

    def _save_pickle(self, items: list[Any], filename: str):
        """Save items as pickle."""
        filepath = self.base_path / f"{filename}.pkl"

        if self.config.compress:
            filepath = self.base_path / f"{filename}.pkl.gz"
            with gzip.open(filepath, "wb") as f:
                pickle.dump(items, f)
        else:
            with open(filepath, "wb") as f:
                pickle.dump(items, f)

    def _save_json(self, items: list[Any], filename: str):
        """Save items as JSON."""
        filepath = self.base_path / f"{filename}.json"

        if self.config.compress:
            filepath = self.base_path / f"{filename}.json.gz"
            with gzip.open(filepath, "wt") as f:
                json.dump(items, f)
        else:
            with open(filepath, "w") as f:
                json.dump(items, f)

    def get_memory_window(self) -> list[Any]:
        """
        Get current memory window.

        Returns:
            List of items in memory
        """
        return self._memory_window.copy()

    def load_from_disk(self, filename: str) -> list[Any] | None:
        """
        Load items from disk.

        Args:
            filename: Filename to load

        Returns:
            List of items or None
        """
        filepath = self.base_path / filename

        if not filepath.exists():
            self.logger.warning(f"File not found: {filepath}")
            return None

        try:
            if filename.endswith(".pkl.gz"):
                with gzip.open(filepath, "rb") as f:
                    return pickle.load(f)
            elif filename.endswith(".pkl"):
                with open(filepath, "rb") as f:
                    return pickle.load(f)
            elif filename.endswith(".json.gz"):
                with gzip.open(filepath, "rt") as f:
                    return json.load(f)
            elif filename.endswith(".json"):
                with open(filepath) as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load {filename}: {e}")
            return None

    def list_disk_files(self) -> list[str]:
        """
        List all disk files.

        Returns:
            List of filenames
        """
        files = []
        for filepath in self.base_path.iterdir():
            if filepath.is_file():
                files.append(filepath.name)
        return sorted(files)

    def cleanup_old_files(self, keep_count: int = 10):
        """
        Cleanup old disk files.

        Args:
            keep_count: Number of files to keep
        """
        files = self.list_disk_files()

        if len(files) > keep_count:
            files_to_delete = files[:-keep_count]

            for filename in files_to_delete:
                filepath = self.base_path / filename
                try:
                    filepath.unlink()
                    self.logger.info(f"Deleted old file: {filename}")
                except Exception as e:
                    self.logger.error(f"Failed to delete {filename}: {e}")

    def get_stats(self) -> dict[str, Any]:
        """
        Get persistence statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "name": self.name,
            "memory_items": len(self._memory_window),
            "disk_files": len(self.list_disk_files()),
            "disk_count": self._disk_count,
            "config": self.config.to_dict(),
        }


class BoundedHistory:
    """
    Bounded history with automatic persistence.

    Combines deque for memory with disk persistence for overflow.
    """

    def __init__(
        self,
        name: str,
        max_memory: int = 10000,
        persist_threshold: float = 0.8,
        config: PersistenceConfig | None = None,
    ):
        """
        Initialize bounded history.

        Args:
            name: History name
            max_memory: Maximum items in memory
            persist_threshold: Persist when memory usage exceeds this fraction
            config: Persistence configuration
        """
        from collections import deque

        self.name = name
        self.max_memory = max_memory
        self.persist_threshold = persist_threshold
        self.persistence = HistoryPersistence(name, config)

        # In-memory deque
        self._deque: deque = deque(maxlen=max_memory)
        self.logger = logger

    def append(self, item: Any):
        """
        Append item to history.

        Args:
            item: Item to append
        """
        self._deque.append(item)

        # Check if we need to persist
        if len(self._deque) >= self.max_memory * self.persist_threshold:
            self.persistence.add(item)

    def get_recent(self, count: int) -> list[Any]:
        """
        Get recent items.

        Args:
            count: Number of items to get

        Returns:
            List of recent items
        """
        return list(self._deque)[-count:]

    def get_all(self) -> list[Any]:
        """
        Get all items (memory + disk).

        Returns:
            List of all items
        """
        # Load from disk
        disk_files = self.persistence.list_disk_files()
        all_items = []

        for filename in disk_files:
            items = self.persistence.load_from_disk(filename)
            if items:
                all_items.extend(items)

        # Add memory items
        all_items.extend(self._deque)

        return all_items

    def clear(self):
        """Clear history."""
        self._deque.clear()
        self.persistence._memory_window.clear()
        self.logger.info(f"History cleared: {self.name}")

    def get_stats(self) -> dict[str, Any]:
        """
        Get history statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "name": self.name,
            "memory_items": len(self._deque),
            "max_memory": self.max_memory,
            "memory_usage": len(self._deque) / self.max_memory,
            "persistence_stats": self.persistence.get_stats(),
        }
