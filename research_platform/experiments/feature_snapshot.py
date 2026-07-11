"""
Feature Snapshot Manager

Tracks exactly which features were used in each experiment.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from research_platform.experiments.base import FeatureSnapshot
from utils.logger import get_logger

logger = get_logger("experiments.feature_snapshot")


class FeatureSnapshotManager:
    """
    Feature Snapshot Manager.
    
    Tracks:
    - Feature names used
    - Feature counts
    - Feature types
    - Feature versions
    """
    
    def __init__(self):
        """Initialize feature snapshot manager."""
        self.snapshots: Dict[str, FeatureSnapshot] = {}
        self._logger = get_logger("experiments.feature_snapshot")
    
    def create_snapshot(
        self,
        experiment_id: str,
        feature_names: List[str],
        df: Optional[pd.DataFrame] = None,
    ) -> FeatureSnapshot:
        """
        Create a feature snapshot.
        
        Args:
            experiment_id: Experiment ID
            feature_names: List of feature names
            df: Optional DataFrame to infer feature types
            
        Returns:
            FeatureSnapshot object
        """
        snapshot_id = f"FS-{uuid.uuid4().hex[:8].upper()}"
        
        # Infer feature types if DataFrame provided
        feature_types = {}
        if df is not None:
            for col in feature_names:
                if col in df.columns:
                    dtype = df[col].dtype
                    if pd.api.types.is_numeric_dtype(dtype):
                        feature_types[col] = "numeric"
                    elif pd.api.types.is_datetime64_any_dtype(dtype):
                        feature_types[col] = "datetime"
                    else:
                        feature_types[col] = "categorical"
                else:
                    feature_types[col] = "unknown"
        else:
            feature_types = {name: "unknown" for name in feature_names}
        
        snapshot = FeatureSnapshot(
            snapshot_id=snapshot_id,
            experiment_id=experiment_id,
            feature_names=feature_names,
            feature_count=len(feature_names),
            feature_types=feature_types,
        )
        
        self.snapshots[snapshot_id] = snapshot
        
        self._logger.info(
            f"Created feature snapshot {snapshot_id}: "
            f"{len(feature_names)} features"
        )
        
        return snapshot
    
    def get_snapshot(self, snapshot_id: str) -> Optional[FeatureSnapshot]:
        """Get a snapshot by ID."""
        return self.snapshots.get(snapshot_id)
    
    def get_snapshots_by_experiment(self, experiment_id: str) -> List[FeatureSnapshot]:
        """Get all snapshots for an experiment."""
        return [
            snapshot for snapshot in self.snapshots.values()
            if snapshot.experiment_id == experiment_id
        ]
    
    def compare_snapshots(
        self,
        snapshot_id_1: str,
        snapshot_id_2: str,
    ) -> Optional[Dict]:
        """
        Compare two feature snapshots.
        
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
        
        features_1 = set(snapshot_1.feature_names)
        features_2 = set(snapshot_2.feature_names)
        
        comparison = {
            'snapshot_1': snapshot_1.to_dict(),
            'snapshot_2': snapshot_2.to_dict(),
            'differences': {
                'added_features': list(features_2 - features_1),
                'removed_features': list(features_1 - features_2),
                'common_features': list(features_1 & features_2),
                'count_change': snapshot_2.feature_count - snapshot_1.feature_count,
            },
        }
        
        return comparison
    
    def get_feature_usage_stats(self) -> Dict[str, int]:
        """
        Get statistics about feature usage across all experiments.
        
        Returns:
            Dictionary mapping feature names to usage counts
        """
        feature_usage = {}
        
        for snapshot in self.snapshots.values():
            for feature in snapshot.feature_names:
                feature_usage[feature] = feature_usage.get(feature, 0) + 1
        
        return feature_usage
    
    def get_most_used_features(self, top_n: int = 10) -> List[tuple[str, int]]:
        """
        Get most used features across all experiments.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            List of (feature_name, usage_count) tuples
        """
        feature_usage = self.get_feature_usage_stats()
        
        sorted_features = sorted(
            feature_usage.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        return sorted_features[:top_n]
    
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
