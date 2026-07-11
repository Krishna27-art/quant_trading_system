"""
Dataset Version Tracker

Tracks dataset versions used in experiments for reproducibility.
"""

import hashlib
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from research_platform.experiments.base import DatasetSnapshot
from utils.logger import get_logger

logger = get_logger("experiments.dataset_version")


class DatasetVersionTracker:
    """
    Dataset Version Tracker.
    
    Tracks:
    - Dataset versions
    - Row counts
    - Feature counts
    - Date ranges
    - Symbol lists
    - Data hashes for verification
    """
    
    def __init__(self):
        """Initialize dataset version tracker."""
        self.snapshots: Dict[str, DatasetSnapshot] = {}
        self._logger = get_logger("experiments.dataset_version")
    
    def create_snapshot(
        self,
        experiment_id: str,
        dataset_name: str,
        df: pd.DataFrame,
        date_column: str = "date",
        symbol_column: str = "symbol",
    ) -> DatasetSnapshot:
        """
        Create a dataset snapshot.
        
        Args:
            experiment_id: Experiment ID
            dataset_name: Name of the dataset
            df: DataFrame containing the data
            date_column: Name of date column
            symbol_column: Name of symbol column
            
        Returns:
            DatasetSnapshot object
        """
        snapshot_id = f"DS-{uuid.uuid4().hex[:8].upper()}"
        
        # Calculate dataset statistics
        rows = len(df)
        features = len(df.columns) - 2  # Exclude date and symbol
        
        # Get date range
        if date_column in df.columns:
            min_date = df[date_column].min()
            max_date = df[date_column].max()
            date_range = f"{min_date} to {max_date}"
        else:
            date_range = "Unknown"
        
        # Get unique symbols
        if symbol_column in df.columns:
            symbols = df[symbol_column].unique().tolist()
        else:
            symbols = []
        
        # Generate version based on date and hash
        version = self._generate_version(df, date_column)
        
        snapshot = DatasetSnapshot(
            snapshot_id=snapshot_id,
            experiment_id=experiment_id,
            dataset_name=dataset_name,
            version=version,
            rows=rows,
            features=features,
            date_range=date_range,
            symbols=symbols,
        )
        
        self.snapshots[snapshot_id] = snapshot
        
        self._logger.info(
            f"Created dataset snapshot {snapshot_id}: {dataset_name} v{version} "
            f"({rows} rows, {features} features)"
        )
        
        return snapshot
    
    def _generate_version(self, df: pd.DataFrame, date_column: str) -> str:
        """
        Generate version string based on data hash and date.
        
        Args:
            df: DataFrame
            date_column: Date column name
            
        Returns:
            Version string
        """
        # Get current date
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Generate hash of data shape and column names
        hash_input = f"{len(df)}_{len(df.columns)}_{'_'.join(df.columns)}"
        data_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        
        return f"{today}-{data_hash}"
    
    def get_snapshot(self, snapshot_id: str) -> Optional[DatasetSnapshot]:
        """Get a snapshot by ID."""
        return self.snapshots.get(snapshot_id)
    
    def get_snapshots_by_experiment(self, experiment_id: str) -> List[DatasetSnapshot]:
        """Get all snapshots for an experiment."""
        return [
            snapshot for snapshot in self.snapshots.values()
            if snapshot.experiment_id == experiment_id
        ]
    
    def get_snapshots_by_dataset(self, dataset_name: str) -> List[DatasetSnapshot]:
        """Get all snapshots for a dataset name."""
        return [
            snapshot for snapshot in self.snapshots.values()
            if snapshot.dataset_name == dataset_name
        ]
    
    def compare_snapshots(
        self,
        snapshot_id_1: str,
        snapshot_id_2: str,
    ) -> Optional[Dict]:
        """
        Compare two dataset snapshots.
        
        Args:
            snapshot_id_1: First snapshot ID
            snapshot_id_2: Second snapshot ID
            
        Returns:
            Dictionary with comparison results
        """
        snapshot_1 = self.snapshots.get(snapshot_id_1)
        snapshot_2 = self.snapshots.get(snapshot_id_2)
        
        if not snapshot_1 or not snapshot_2:
            return None
        
        comparison = {
            'snapshot_1': snapshot_1.to_dict(),
            'snapshot_2': snapshot_2.to_dict(),
            'differences': {
                'rows': snapshot_2.rows - snapshot_1.rows,
                'features': snapshot_2.features - snapshot_1.features,
                'version_changed': snapshot_1.version != snapshot_2.version,
            },
        }
        
        return comparison
    
    def get_latest_snapshot(self, dataset_name: str) -> Optional[DatasetSnapshot]:
        """Get the latest snapshot for a dataset."""
        snapshots = self.get_snapshots_by_dataset(dataset_name)
        
        if not snapshots:
            return None
        
        # Sort by creation time descending
        snapshots.sort(key=lambda x: x.created_at, reverse=True)
        
        return snapshots[0]
    
    def delete_snapshot(self, snapshot_id: str) -> bool:
        """
        Delete a snapshot.
        
        Args:
            snapshot_id: Snapshot ID
            
        Returns:
            True if deleted successfully
        """
        if snapshot_id not in self.snapshots:
            self._logger.error(f"Snapshot not found: {snapshot_id}")
            return False
        
        del self.snapshots[snapshot_id]
        self._logger.info(f"Deleted snapshot {snapshot_id}")
        return True
