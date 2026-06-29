"""
Raw Bronze Layer

Institutional-grade immutable raw storage.
Stores EXACT responses as-is for replayability and audit trail.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from config.settings import BRONZE_DIR
from utils.logger import get_logger

logger = get_logger("raw_bronze", pipeline_id="raw_bronze_storage")


class RawBronzeLayer:
    """
    Immutable raw storage layer for institutional-grade data preservation.

    Stores EXACT responses as-is for replayability and audit trail.
    No transformation, no normalization, no feature engineering.
    """

    def __init__(self, base_path: Path | None = None):
        """
        Initialize raw bronze layer.

        Args:
            base_path: Base path for raw bronze storage
        """
        self.base_path = base_path or BRONZE_DIR / "raw"
        self.logger = logger
        self.base_path.mkdir(parents=True, exist_ok=True)

    def store_raw_response(
        self, dataset: str, source: str, raw_data: Any, metadata: dict[str, Any] | None = None
    ) -> str:
        """
        Store raw response as immutable JSON.

        Args:
            dataset: Dataset identifier (e.g., 'equity_history_RELIANCE')
            source: Source name (e.g., 'nselib', 'scraper')
            raw_data: Raw data to store (dict, list, or DataFrame)
            metadata: Additional metadata

        Returns:
            Path to stored file
        """
        try:
            # Create dataset directory
            dataset_dir = self.base_path / dataset
            dataset_dir.mkdir(parents=True, exist_ok=True)

            # Generate unique filename with timestamp (explicitly in IST to prevent ambiguity)
            ist_now = datetime.now(ZoneInfo("Asia/Kolkata"))
            timestamp = ist_now.strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{source}_{timestamp}.json"
            filepath = dataset_dir / filename

            # Prepare storage object
            storage_obj = {
                "id": str(uuid.uuid4()),
                "dataset": dataset,
                "source": source,
                "timestamp": ist_now.isoformat(),
                "raw_data": self._serialize_data(raw_data),
                "metadata": metadata or {},
            }

            # Store as JSON
            with open(filepath, "w") as f:
                json.dump(storage_obj, f, indent=2, default=str)

            self.logger.info(f"Stored raw response: {filepath}")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"Failed to store raw response: {str(e)}")
            raise

    def _serialize_data(self, data: Any) -> Any:
        """
        Serialize data for JSON storage.

        Args:
            data: Data to serialize

        Returns:
            Serializable data
        """
        if isinstance(data, pd.DataFrame):
            return {
                "type": "dataframe",
                "columns": data.columns.tolist(),
                "index": data.index.tolist(),
                "data": data.to_dict(orient="records"),
            }
        elif isinstance(data, (dict, list)):
            return data
        else:
            return str(data)

    def retrieve_raw_response(
        self, dataset: str, source: str | None = None, timestamp: str | None = None
    ) -> dict[str, Any] | None:
        """
        Retrieve raw response from storage.

        Args:
            dataset: Dataset identifier
            source: Source filter (optional)
            timestamp: Timestamp filter (optional)

        Returns:
            Raw response data or None
        """
        try:
            dataset_dir = self.base_path / dataset

            if not dataset_dir.exists():
                return None

            # List files in dataset directory
            files = list(dataset_dir.glob("*.json"))

            if not files:
                return None

            # Filter by source if provided
            if source:
                files = [f for f in files if f.name.startswith(f"{source}_")]

            # Sort by timestamp (newest first)
            files.sort(reverse=True)

            # If timestamp provided, find specific file
            if timestamp:
                files = [f for f in files if timestamp in f.name]

            if not files:
                return None

            # Load most recent file
            with open(files[0]) as f:
                data = json.load(f)

            return data

        except Exception as e:
            self.logger.error(f"Failed to retrieve raw response: {str(e)}")
            return None

    def list_raw_snapshots(self, dataset: str) -> list:
        """
        List all raw snapshots for a dataset.

        Args:
            dataset: Dataset identifier

        Returns:
            List of snapshot metadata
        """
        try:
            dataset_dir = self.base_path / dataset

            if not dataset_dir.exists():
                return []

            snapshots = []

            for filepath in dataset_dir.glob("*.json"):
                try:
                    with open(filepath) as f:
                        data = json.load(f)

                    snapshots.append(
                        {
                            "filepath": str(filepath),
                            "source": data["source"],
                            "timestamp": data["timestamp"],
                            "id": data["id"],
                        }
                    )

                except Exception as e:
                    self.logger.warning(f"Failed to read snapshot {filepath}: {str(e)}")

            # Sort by timestamp (newest first)
            snapshots.sort(key=lambda x: x["timestamp"], reverse=True)

            return snapshots

        except Exception as e:
            self.logger.error(f"Failed to list snapshots: {str(e)}")
            return []

    def get_latest_snapshot(self, dataset: str, source: str | None = None) -> dict[str, Any] | None:
        """
        Get the latest snapshot for a dataset.

        Args:
            dataset: Dataset identifier
            source: Source filter (optional)

        Returns:
            Latest snapshot data or None
        """
        snapshots = self.list_raw_snapshots(dataset)

        if not snapshots:
            return None

        if source:
            snapshots = [s for s in snapshots if s["source"] == source]

        if not snapshots:
            return None

        filepath = snapshots[0]["filepath"]

        with open(filepath) as f:
            return json.load(f)

    def delete_old_snapshots(self, dataset: str, keep_count: int = 10) -> int:
        """
        Delete old snapshots, keeping only the most recent N.

        Args:
            dataset: Dataset identifier
            keep_count: Number of snapshots to keep

        Returns:
            Number of snapshots deleted
        """
        try:
            snapshots = self.list_raw_snapshots(dataset)

            if len(snapshots) <= keep_count:
                return 0

            # Delete oldest snapshots
            to_delete = snapshots[keep_count:]
            deleted_count = 0

            for snapshot in to_delete:
                try:
                    Path(snapshot["filepath"]).unlink()
                    deleted_count += 1
                    self.logger.info(f"Deleted old snapshot: {snapshot['filepath']}")
                except Exception as e:
                    self.logger.error(f"Failed to delete snapshot {snapshot['filepath']}: {str(e)}")

            return deleted_count

        except Exception as e:
            self.logger.error(f"Failed to delete old snapshots: {str(e)}")
            return 0
