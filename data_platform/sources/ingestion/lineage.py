"""
Ingestion Lineage Tracking

Tracks data lineage for all ingestion operations.
Critical for audit trail and reproducibility.
"""

import json
import uuid
from pathlib import Path
from typing import Any

from config.settings import BRONZE_DIR
from utils.logger import get_logger
from utils.time_utils import now_ist

logger = get_logger("ingestion_lineage", pipeline_id="ingestion_lineage")


class IngestionLineage:
    """
    Tracks data lineage for all ingestion operations.

    Records:
    - Source used
    - Fallback triggered
    - API version
    - Ingestion timestamp
    - Success/failure status
    - Latency
    - Error details
    """

    def __init__(self, base_path: Path | None = None):
        """
        Initialize ingestion lineage tracker.

        Args:
            base_path: Base path for lineage storage
        """
        self.base_path = base_path or BRONZE_DIR / "lineage"
        self.logger = logger
        self.base_path.mkdir(parents=True, exist_ok=True)

    def record_ingestion(
        self,
        dataset: str,
        source: str,
        success: bool,
        latency_ms: float,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
        fallback_used: bool = False,
        fallback_source: str | None = None,
        api_version: str | None = None,
    ) -> str:
        """
        Record an ingestion operation.

        Args:
            dataset: Dataset identifier
            source: Source used
            success: Success status
            latency_ms: Operation latency
            metadata: Additional metadata
            error: Error message if failed
            fallback_used: Whether fallback was triggered
            fallback_source: Fallback source used
            api_version: API version used

        Returns:
            Lineage record ID
        """
        try:
            record_id = str(uuid.uuid4())
            timestamp = now_ist().isoformat()

            record = {
                "id": record_id,
                "dataset": dataset,
                "source": source,
                "success": success,
                "latency_ms": latency_ms,
                "timestamp": timestamp,
                "metadata": metadata or {},
                "error": error,
                "fallback_used": fallback_used,
                "fallback_source": fallback_source,
                "api_version": api_version,
            }

            # Store in dataset-specific file
            dataset_file = self.base_path / f"{dataset}.jsonl"

            with open(dataset_file, "a") as f:
                f.write(json.dumps(record) + "\n")

            self.logger.info(f"Recorded ingestion lineage: {record_id}")
            return record_id

        except Exception as e:
            self.logger.error(f"Failed to record ingestion lineage: {str(e)}")
            raise

    def get_ingestion_history(self, dataset: str, limit: int = 100) -> list[dict[str, Any]]:
        """
        Get ingestion history for a dataset.

        Args:
            dataset: Dataset identifier
            limit: Maximum number of records

        Returns:
            List of ingestion records
        """
        try:
            dataset_file = self.base_path / f"{dataset}.jsonl"

            if not dataset_file.exists():
                return []

            records = []

            with open(dataset_file) as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))

            # Sort by timestamp (newest first)
            records.sort(key=lambda x: x["timestamp"], reverse=True)

            return records[:limit]

        except Exception as e:
            self.logger.error(f"Failed to get ingestion history: {str(e)}")
            return []

    def get_source_statistics(self, dataset: str) -> dict[str, Any]:
        """
        Get source usage statistics for a dataset.

        Args:
            dataset: Dataset identifier

        Returns:
            Dictionary with source statistics
        """
        try:
            records = self.get_ingestion_history(dataset, limit=1000)

            if not records:
                return {}

            stats = {
                "total_attempts": len(records),
                "successful": sum(1 for r in records if r["success"]),
                "failed": sum(1 for r in records if not r["success"]),
                "fallback_used": sum(1 for r in records if r["fallback_used"]),
                "sources": {},
                "avg_latency_ms": 0,
            }

            # Source breakdown
            for record in records:
                source = record["source"]
                if source not in stats["sources"]:
                    stats["sources"][source] = {"count": 0, "success": 0, "failure": 0}

                stats["sources"][source]["count"] += 1
                if record["success"]:
                    stats["sources"][source]["success"] += 1
                else:
                    stats["sources"][source]["failure"] += 1

            # Average latency
            latencies = [r["latency_ms"] for r in records if r["latency_ms"] > 0]
            if latencies:
                stats["avg_latency_ms"] = sum(latencies) / len(latencies)

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get source statistics: {str(e)}")
            return {}

    def get_recent_failures(self, dataset: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get recent failures for a dataset.

        Args:
            dataset: Dataset identifier
            limit: Maximum number of records

        Returns:
            List of failure records
        """
        try:
            records = self.get_ingestion_history(dataset, limit=1000)

            failures = [r for r in records if not r["success"]]

            # Sort by timestamp (newest first)
            failures.sort(key=lambda x: x["timestamp"], reverse=True)

            return failures[:limit]

        except Exception as e:
            self.logger.error(f"Failed to get recent failures: {str(e)}")
            return []

    def get_fallback_events(self, dataset: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get fallback events for a dataset.

        Args:
            dataset: Dataset identifier
            limit: Maximum number of records

        Returns:
            List of fallback records
        """
        try:
            records = self.get_ingestion_history(dataset, limit=1000)

            fallbacks = [r for r in records if r["fallback_used"]]

            # Sort by timestamp (newest first)
            fallbacks.sort(key=lambda x: x["timestamp"], reverse=True)

            return fallbacks[:limit]

        except Exception as e:
            self.logger.error(f"Failed to get fallback events: {str(e)}")
            return []
