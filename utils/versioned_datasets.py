"""
Versioned Datasets

Institutional-grade versioned dataset management.
Prevents data loss by creating snapshots instead of overwriting.
Supports reproducibility and auditability.
"""

import json
import shutil
from pathlib import Path

import pandas as pd

from utils.logger import get_logger
from utils.time_utils import now_ist

logger = get_logger("versioned_datasets")


class VersionedDataset:
    """
    Versioned dataset manager.

    Creates timestamped snapshots instead of overwriting data.
    Supports version history, rollback, and reproducibility.
    """

    def __init__(self, base_path: Path):
        """
        Initialize versioned dataset manager.

        Args:
            base_path: Base directory for dataset (e.g., data/bronze/equity_history)
        """
        self.base_path = Path(base_path)
        self.versions_path = self.base_path / "versions"
        self.current_path = self.base_path / "current"
        self.logger = logger

        # Create directories
        self.versions_path.mkdir(parents=True, exist_ok=True)
        self.current_path.mkdir(parents=True, exist_ok=True)

    def save_snapshot(
        self, df: pd.DataFrame, dataset_name: str, metadata: dict | None = None
    ) -> Path:
        """
        Save a new snapshot of the dataset.

        Args:
            df: DataFrame to save
            dataset_name: Name of the dataset
            metadata: Optional metadata dictionary

        Returns:
            Path to the saved snapshot
        """
        timestamp = now_ist().strftime("%Y%m%d_%H%M%S")
        snapshot_name = f"snapshot_{timestamp}"
        snapshot_path = self.versions_path / dataset_name / snapshot_name

        # Create snapshot directory
        snapshot_path.mkdir(parents=True, exist_ok=True)

        # Save parquet
        parquet_path = snapshot_path / f"{dataset_name}.parquet"
        df.to_parquet(parquet_path, index=False)

        # Save metadata
        if metadata is None:
            metadata = {}

        metadata.update(
            {
                "snapshot_name": snapshot_name,
                "timestamp": timestamp,
                "dataset_name": dataset_name,
                "row_count": len(df),
                "columns": list(df.columns),
                "saved_at": now_ist().isoformat(),
            }
        )

        metadata_path = snapshot_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Update current symlink
        self._update_current(dataset_name, snapshot_path)

        # Update version catalog
        self._update_catalog(dataset_name, snapshot_name, metadata)

        self.logger.info(f"Saved snapshot {snapshot_name} for {dataset_name}")
        return parquet_path

    def load_current(self, dataset_name: str) -> pd.DataFrame | None:
        """
        Load the current version of a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            DataFrame or None if not found
        """
        current_path = self.current_path / dataset_name / f"{dataset_name}.parquet"

        if current_path.exists():
            return pd.read_parquet(current_path)

        self.logger.warning(f"No current version found for {dataset_name}")
        return None

    def load_snapshot(self, dataset_name: str, snapshot_name: str) -> pd.DataFrame | None:
        """
        Load a specific snapshot of a dataset.

        Args:
            dataset_name: Name of the dataset
            snapshot_name: Name of the snapshot

        Returns:
            DataFrame or None if not found
        """
        snapshot_path = (
            self.versions_path / dataset_name / snapshot_name / f"{dataset_name}.parquet"
        )

        if snapshot_path.exists():
            return pd.read_parquet(snapshot_path)

        self.logger.warning(f"Snapshot {snapshot_name} not found for {dataset_name}")
        return None

    def list_snapshots(self, dataset_name: str) -> list[dict]:
        """
        List all snapshots for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            List of snapshot metadata dictionaries
        """
        dataset_path = self.versions_path / dataset_name

        if not dataset_path.exists():
            return []

        snapshots = []
        for snapshot_dir in sorted(dataset_path.iterdir(), reverse=True):
            if snapshot_dir.is_dir():
                metadata_path = snapshot_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                        snapshots.append(metadata)

        return snapshots

    def get_current_snapshot(self, dataset_name: str) -> dict | None:
        """
        Get metadata of the current snapshot.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Metadata dictionary or None if not found
        """
        current_path = self.current_path / dataset_name / "metadata.json"

        if current_path.exists():
            with open(current_path) as f:
                return json.load(f)

        return None

    def rollback(self, dataset_name: str, snapshot_name: str) -> bool:
        """
        Rollback to a specific snapshot.

        Args:
            dataset_name: Name of the dataset
            snapshot_name: Name of the snapshot to rollback to

        Returns:
            True if successful, False otherwise
        """
        snapshot_path = self.versions_path / dataset_name / snapshot_name

        if not snapshot_path.exists():
            self.logger.error(f"Snapshot {snapshot_name} not found for {dataset_name}")
            return False

        # Update current symlink
        self._update_current(dataset_name, snapshot_path)

        # Update catalog
        metadata_path = snapshot_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                metadata["rolled_back_at"] = now_ist().isoformat()
                self._update_catalog(dataset_name, snapshot_name, metadata)

        self.logger.info(f"Rolled back {dataset_name} to {snapshot_name}")
        return True

    def delete_old_snapshots(self, dataset_name: str, keep_count: int = 10) -> int:
        """
        Delete old snapshots, keeping only the most recent N.

        Args:
            dataset_name: Name of the dataset
            keep_count: Number of snapshots to keep

        Returns:
            Number of snapshots deleted
        """
        snapshots = self.list_snapshots(dataset_name)

        if len(snapshots) <= keep_count:
            return 0

        # Delete oldest snapshots
        to_delete = snapshots[keep_count:]
        deleted_count = 0

        for snapshot in to_delete:
            snapshot_path = self.versions_path / dataset_name / snapshot["snapshot_name"]
            try:
                shutil.rmtree(snapshot_path)
                deleted_count += 1
                self.logger.info(f"Deleted old snapshot {snapshot['snapshot_name']}")
            except Exception as e:
                self.logger.error(
                    f"Failed to delete snapshot {snapshot['snapshot_name']}: {str(e)}"
                )

        return deleted_count

    def _update_current(self, dataset_name: str, snapshot_path: Path) -> None:
        """
        Update the current symlink to point to the latest snapshot.

        Args:
            dataset_name: Name of the dataset
            snapshot_path: Path to the snapshot
        """
        current_link = self.current_path / dataset_name

        # Remove existing symlink or directory
        if current_link.exists():
            if current_link.is_symlink():
                current_link.unlink()
            else:
                shutil.rmtree(current_link)

        # Create new symlink
        current_link.symlink_to(snapshot_path)

        # Copy metadata to current
        metadata_path = snapshot_path / "metadata.json"
        if metadata_path.exists():
            shutil.copy2(metadata_path, self.current_path / dataset_name / "metadata.json")

    def _update_catalog(self, dataset_name: str, snapshot_name: str, metadata: dict) -> None:
        """
        Update the version catalog.

        Args:
            dataset_name: Name of the dataset
            snapshot_name: Name of the snapshot
            metadata: Metadata dictionary
        """
        catalog_path = self.base_path / "version_catalog.json"

        catalog = {}
        if catalog_path.exists():
            with open(catalog_path) as f:
                catalog = json.load(f)

        if dataset_name not in catalog:
            catalog[dataset_name] = []

        # Add or update snapshot in catalog
        existing = next(
            (s for s in catalog[dataset_name] if s["snapshot_name"] == snapshot_name), None
        )

        if existing:
            existing.update(metadata)
        else:
            catalog[dataset_name].append(metadata)

        # Sort by timestamp (newest first)
        catalog[dataset_name].sort(key=lambda x: x["timestamp"], reverse=True)

        with open(catalog_path, "w") as f:
            json.dump(catalog, f, indent=2)

    def get_catalog(self) -> dict:
        """
        Get the version catalog.

        Returns:
            Dictionary with all dataset versions
        """
        catalog_path = self.base_path / "version_catalog.json"

        if catalog_path.exists():
            with open(catalog_path) as f:
                return json.load(f)

        return {}
