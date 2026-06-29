"""
Write-Ahead Log (WAL) Journal for OMS

Provides crash recovery by durably recording all state changes before they
are applied to the in-memory state. Follows database best practices for
append-only persistence.
"""

import binascii
import json
import os
import threading
from dataclasses import asdict, dataclass
from typing import Any

from utils.logger import get_logger
from utils.time_utils import now_ist

logger = get_logger(__name__)


@dataclass
class WALEntry:
    """A single log entry in the WAL."""

    entry_id: int
    timestamp: float
    operation: str
    payload: dict[str, Any]
    checksum: int = 0

    def compute_checksum(self) -> int:
        """Compute CRC32 checksum of the core fields for data integrity."""
        data = f"{self.entry_id}:{self.timestamp}:{self.operation}:{json.dumps(self.payload, sort_keys=True)}"
        return binascii.crc32(data.encode("utf-8"))

    def to_json(self) -> str:
        self.checksum = self.compute_checksum()
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "WALEntry":
        data = json.loads(json_str)
        entry = cls(**data)

        # Verify checksum
        expected_checksum = entry.compute_checksum()
        if entry.checksum != expected_checksum:
            raise ValueError(
                f"WAL Entry {entry.entry_id} checksum mismatch! Data corruption detected."
            )
        return entry


class WALJournal:
    """
    Append-only Write-Ahead Log for durability and crash recovery.
    Performs synchronous durable disk I/O.
    """

    def __init__(self, log_dir: str = "logs/wal", max_file_size_bytes: int = 10 * 1024 * 1024):
        self.log_dir = log_dir
        self.max_file_size_bytes = max_file_size_bytes
        self.lock = threading.Lock()

        os.makedirs(self.log_dir, exist_ok=True)

        self.current_file_path = os.path.join(self.log_dir, "oms_wal_current.log")
        self.next_entry_id = 1
        self._file = None

        self._open_file()
        self._init_entry_id()

    def _open_file(self):
        """Open the current WAL file in append mode."""
        self._file = open(self.current_file_path, "a", encoding="utf-8")  # noqa: SIM115

    def _init_entry_id(self):
        """Find the next valid entry ID based on existing logs."""
        try:
            if (
                os.path.exists(self.current_file_path)
                and os.path.getsize(self.current_file_path) > 0
            ):
                with open(self.current_file_path, encoding="utf-8") as f:
                    lines = f.readlines()
                    if lines:
                        last_entry = WALEntry.from_json(lines[-1])
                        self.next_entry_id = last_entry.entry_id + 1
        except Exception as e:
            logger.error(f"Error initializing WAL entry ID: {e}")
            self.next_entry_id = 1

    def _rotate_log_if_needed(self):
        """Rotate the WAL file if it exceeds the maximum size."""
        if not self._file:
            return

        self._file.flush()
        if os.path.getsize(self.current_file_path) >= self.max_file_size_bytes:
            self._file.close()
            timestamp_str = now_ist().strftime("%Y%m%d_%H%M%S")
            rotated_path = os.path.join(self.log_dir, f"oms_wal_{timestamp_str}.log")
            os.rename(self.current_file_path, rotated_path)
            logger.info(f"WAL log rotated to {rotated_path}")
            self._open_file()

    def write(self, operation: str, payload: dict[str, Any]) -> int:
        """
        Durably write an operation to the log file and fsync() synchronously.
        """
        with self.lock:
            entry_id = self.next_entry_id
            self.next_entry_id += 1

            entry = WALEntry(
                entry_id=entry_id,
                timestamp=now_ist().timestamp(),
                operation=operation,
                payload=payload,
            )

            log_line = entry.to_json() + "\n"
            if self._file:
                try:
                    self._file.write(log_line)
                    self._file.flush()
                    os.fsync(self._file.fileno())  # Ensure durable write
                    self._rotate_log_if_needed()
                except Exception as e:
                    logger.error(f"Failed to write WAL entry synchronously: {e}")
                    raise
            return entry.entry_id

    def begin_transaction(self, tx_id: str, operations: list[dict[str, Any]]) -> int:
        """Phase 1: Write prepare log."""
        return self.write("PREPARE", {"tx_id": tx_id, "operations": operations})

    def commit_transaction(self, tx_id: str) -> int:
        """Phase 2: Write commit log."""
        return self.write("COMMIT", {"tx_id": tx_id})

    def rollback_transaction(self, tx_id: str) -> int:
        """Phase 2 abort: Write rollback log."""
        return self.write("ROLLBACK", {"tx_id": tx_id})

    def read_all(self) -> list[WALEntry]:
        """Read all valid entries from the current WAL file."""
        entries = []
        if not os.path.exists(self.current_file_path):
            return entries

        with self.lock, open(self.current_file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = WALEntry.from_json(line)
                    entries.append(entry)
                except Exception as e:
                    logger.error(
                        f"Failed to read WAL entry: {e}. Stopping read to prevent corruption."
                    )
                    break
        return entries

    def replay(self, oms: Any) -> None:
        """
        Replay all entries from the WAL into the provided OMS instance.
        Only replays fully committed transactions in a Two-Phase Commit protocol.
        """
        logger.info("Starting WAL replay for crash recovery...")
        entries = self.read_all()

        # 2PC Resolution
        valid_entries = []
        pending_tx = {}

        for entry in entries:
            op = entry.operation
            if op == "PREPARE":
                tx_id = entry.payload.get("tx_id")
                pending_tx[tx_id] = entry
            elif op == "COMMIT":
                tx_id = entry.payload.get("tx_id")
                if tx_id in pending_tx:
                    valid_entries.append(pending_tx[tx_id])
                    del pending_tx[tx_id]
            elif op == "ROLLBACK":
                tx_id = entry.payload.get("tx_id")
                if tx_id in pending_tx:
                    del pending_tx[tx_id]
            else:
                valid_entries.append(entry)

        # Sort by entry ID to maintain causal order
        valid_entries.sort(key=lambda x: x.entry_id)

        success_count = 0
        for entry in valid_entries:
            try:
                if hasattr(oms, "handle_wal_operation"):
                    if entry.operation == "PREPARE":
                        for sub_op in entry.payload.get("operations", []):
                            oms.handle_wal_operation(sub_op.get("operation"), sub_op.get("payload"))
                    else:
                        oms.handle_wal_operation(entry.operation, entry.payload)
                    success_count += 1
                else:
                    logger.warning("OMS instance lacks handle_wal_operation method.")
            except Exception as e:
                logger.error(f"Error replaying WAL entry {entry.entry_id} ({entry.operation}): {e}")

        logger.info(f"WAL replay completed. Replayed {success_count}/{len(valid_entries)} entries.")

    def checkpoint(self) -> None:
        """
        Create a checkpoint by rotating the current log.
        """
        with self.lock:
            if self._file:
                self._file.close()
                timestamp_str = now_ist().strftime("%Y%m%d_%H%M%S")
                checkpoint_path = os.path.join(self.log_dir, f"oms_checkpoint_{timestamp_str}.log")
                if os.path.exists(self.current_file_path):
                    os.rename(self.current_file_path, checkpoint_path)
                logger.info(f"WAL checkpoint created at {checkpoint_path}")
                self._open_file()

    def close(self):
        """Close the WAL file."""
        with self.lock:
            if self._file:
                self._file.flush()
                os.fsync(self._file.fileno())
                self._file.close()
                self._file = None
