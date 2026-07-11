"""
Signal Snapshot Manager

Tracks signal versions and weights used in experiments.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from research_platform.experiments.base import SignalSnapshot
from utils.logger import get_logger

logger = get_logger("experiments.signal_snapshot")


class SignalSnapshotManager:
    """
    Signal Snapshot Manager.
    
    Tracks:
    - Signal versions (e.g., Trend V3, Volume V2)
    - Signal weights
    - Signal configurations
    """
    
    def __init__(self):
        """Initialize signal snapshot manager."""
        self.snapshots: Dict[str, SignalSnapshot] = {}
        self._logger = get_logger("experiments.signal_snapshot")
    
    def create_snapshot(
        self,
        experiment_id: str,
        signal_versions: Dict[str, str],
        signal_weights: Dict[str, float],
    ) -> SignalSnapshot:
        """
        Create a signal snapshot.
        
        Args:
            experiment_id: Experiment ID
            signal_versions: Dictionary mapping signal names to versions
            signal_weights: Dictionary mapping signal names to weights
            
        Returns:
            SignalSnapshot object
        """
        snapshot_id = f"SS-{uuid.uuid4().hex[:8].upper()}"
        
        snapshot = SignalSnapshot(
            snapshot_id=snapshot_id,
            experiment_id=experiment_id,
            signal_versions=signal_versions,
            signal_weights=signal_weights,
        )
        
        self.snapshots[snapshot_id] = snapshot
        
        self._logger.info(
            f"Created signal snapshot {snapshot_id}: "
            f"{len(signal_versions)} signals"
        )
        
        return snapshot
    
    def get_snapshot(self, snapshot_id: str) -> Optional[SignalSnapshot]:
        """Get a snapshot by ID."""
        return self.snapshots.get(snapshot_id)
    
    def get_snapshots_by_experiment(self, experiment_id: str) -> List[SignalSnapshot]:
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
        Compare two signal snapshots.
        
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
            'version_changes': self._compare_versions(
                snapshot_1.signal_versions,
                snapshot_2.signal_versions,
            ),
            'weight_changes': self._compare_weights(
                snapshot_1.signal_weights,
                snapshot_2.signal_weights,
            ),
        }
        
        return comparison
    
    def _compare_versions(
        self,
        versions_1: Dict[str, str],
        versions_2: Dict[str, str],
    ) -> Dict:
        """Compare signal versions."""
        changes = {
            'upgraded': [],
            'downgraded': [],
            'added': [],
            'removed': [],
        }
        
        all_signals = set(versions_1.keys()) | set(versions_2.keys())
        
        for signal in all_signals:
            if signal in versions_1 and signal in versions_2:
                if versions_1[signal] != versions_2[signal]:
                    # Simple version comparison
                    v1_parts = versions_1[signal].replace('V', '').split('.')
                    v2_parts = versions_2[signal].replace('V', '').split('.')
                    
                    try:
                        if int(v2_parts[0]) > int(v1_parts[0]):
                            changes['upgraded'].append(signal)
                        else:
                            changes['downgraded'].append(signal)
                    except (ValueError, IndexError):
                        changes['upgraded'].append(signal)
            elif signal in versions_2:
                changes['added'].append(signal)
            else:
                changes['removed'].append(signal)
        
        return changes
    
    def _compare_weights(
        self,
        weights_1: Dict[str, float],
        weights_2: Dict[str, float],
    ) -> Dict:
        """Compare signal weights."""
        changes = {
            'increased': [],
            'decreased': [],
            'added': [],
            'removed': [],
        }
        
        all_signals = set(weights_1.keys()) | set(weights_2.keys())
        
        for signal in all_signals:
            if signal in weights_1 and signal in weights_2:
                if weights_2[signal] > weights_1[signal]:
                    changes['increased'].append(
                        (signal, weights_1[signal], weights_2[signal])
                    )
                elif weights_2[signal] < weights_1[signal]:
                    changes['decreased'].append(
                        (signal, weights_1[signal], weights_2[signal])
                    )
            elif signal in weights_2:
                changes['added'].append((signal, weights_2[signal]))
            else:
                changes['removed'].append((signal, weights_1[signal]))
        
        return changes
    
    def get_signal_usage_stats(self) -> Dict[str, int]:
        """
        Get statistics about signal usage across all experiments.
        
        Returns:
            Dictionary mapping signal names to usage counts
        """
        signal_usage = {}
        
        for snapshot in self.snapshots.values():
            for signal in snapshot.signal_versions.keys():
                signal_usage[signal] = signal_usage.get(signal, 0) + 1
        
        return signal_usage
    
    def get_most_used_signals(self, top_n: int = 10) -> List[tuple[str, int]]:
        """
        Get most used signals across all experiments.
        
        Args:
            top_n: Number of top signals to return
            
        Returns:
            List of (signal_name, usage_count) tuples
        """
        signal_usage = self.get_signal_usage_stats()
        
        sorted_signals = sorted(
            signal_usage.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        return sorted_signals[:top_n]
    
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
