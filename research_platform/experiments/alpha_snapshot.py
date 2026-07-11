"""
Alpha Snapshot Manager

Tracks alpha configurations and weights used in experiments.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from research_platform.experiments.base import AlphaSnapshot
from utils.logger import get_logger

logger = get_logger("experiments.alpha_snapshot")


class AlphaSnapshotManager:
    """
    Alpha Snapshot Manager.
    
    Tracks:
    - Alpha versions
    - Alpha weights
    - Alpha rules
    - Alpha configurations
    """
    
    def __init__(self):
        """Initialize alpha snapshot manager."""
        self.snapshots: Dict[str, AlphaSnapshot] = {}
        self._logger = get_logger("experiments.alpha_snapshot")
    
    def create_snapshot(
        self,
        experiment_id: str,
        alpha_version: str,
        alpha_weights: Dict[str, float],
        alpha_rules: List[str],
    ) -> AlphaSnapshot:
        """
        Create an alpha snapshot.
        
        Args:
            experiment_id: Experiment ID
            alpha_version: Alpha version string
            alpha_weights: Dictionary mapping alpha names to weights
            alpha_rules: List of alpha rules
            
        Returns:
            AlphaSnapshot object
        """
        snapshot_id = f"AS-{uuid.uuid4().hex[:8].upper()}"
        
        snapshot = AlphaSnapshot(
            snapshot_id=snapshot_id,
            experiment_id=experiment_id,
            alpha_version=alpha_version,
            alpha_weights=alpha_weights,
            alpha_rules=alpha_rules,
        )
        
        self.snapshots[snapshot_id] = snapshot
        
        self._logger.info(
            f"Created alpha snapshot {snapshot_id}: "
            f"{alpha_version} ({len(alpha_weights)} weights, {len(alpha_rules)} rules)"
        )
        
        return snapshot
    
    def get_snapshot(self, snapshot_id: str) -> Optional[AlphaSnapshot]:
        """Get a snapshot by ID."""
        return self.snapshots.get(snapshot_id)
    
    def get_snapshots_by_experiment(self, experiment_id: str) -> List[AlphaSnapshot]:
        """Get all snapshots for an experiment."""
        return [
            snapshot for snapshot in self.snapshots.values()
            if snapshot.experiment_id == experiment_id
        ]
    
    def get_snapshots_by_version(self, alpha_version: str) -> List[AlphaSnapshot]:
        """Get all snapshots for a specific alpha version."""
        return [
            snapshot for snapshot in self.snapshots.values()
            if snapshot.alpha_version == alpha_version
        ]
    
    def compare_snapshots(
        self,
        snapshot_id_1: str,
        snapshot_id_2: str,
    ) -> Optional[Dict]:
        """
        Compare two alpha snapshots.
        
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
            'version_changed': snapshot_1.alpha_version != snapshot_2.alpha_version,
            'weight_changes': self._compare_weights(
                snapshot_1.alpha_weights,
                snapshot_2.alpha_weights,
            ),
            'rule_changes': self._compare_rules(
                snapshot_1.alpha_rules,
                snapshot_2.alpha_rules,
            ),
        }
        
        return comparison
    
    def _compare_weights(
        self,
        weights_1: Dict[str, float],
        weights_2: Dict[str, float],
    ) -> Dict:
        """Compare alpha weights."""
        changes = {
            'increased': [],
            'decreased': [],
            'added': [],
            'removed': [],
        }
        
        all_alphas = set(weights_1.keys()) | set(weights_2.keys())
        
        for alpha in all_alphas:
            if alpha in weights_1 and alpha in weights_2:
                if weights_2[alpha] > weights_1[alpha]:
                    changes['increased'].append(
                        (alpha, weights_1[alpha], weights_2[alpha])
                    )
                elif weights_2[alpha] < weights_1[alpha]:
                    changes['decreased'].append(
                        (alpha, weights_1[alpha], weights_2[alpha])
                    )
            elif alpha in weights_2:
                changes['added'].append((alpha, weights_2[alpha]))
            else:
                changes['removed'].append((alpha, weights_1[alpha]))
        
        return changes
    
    def _compare_rules(
        self,
        rules_1: List[str],
        rules_2: List[str],
    ) -> Dict:
        """Compare alpha rules."""
        set_1 = set(rules_1)
        set_2 = set(rules_2)
        
        return {
            'added': list(set_2 - set_1),
            'removed': list(set_1 - set_2),
            'common': list(set_1 & set_2),
            'count_change': len(rules_2) - len(rules_1),
        }
    
    def get_alpha_version_history(self) -> Dict[str, List[AlphaSnapshot]]:
        """
        Get history of alpha versions.
        
        Returns:
            Dictionary mapping alpha versions to list of snapshots
        """
        version_history = {}
        
        for snapshot in self.snapshots.values():
            if snapshot.alpha_version not in version_history:
                version_history[snapshot.alpha_version] = []
            version_history[snapshot.alpha_version].append(snapshot)
        
        # Sort snapshots by creation time
        for version in version_history:
            version_history[version].sort(key=lambda x: x.created_at)
        
        return version_history
    
    def get_latest_alpha_version(self) -> Optional[str]:
        """Get the latest alpha version."""
        if not self.snapshots:
            return None
        
        # Sort by creation time descending
        sorted_snapshots = sorted(
            self.snapshots.values(),
            key=lambda x: x.created_at,
            reverse=True,
        )
        
        return sorted_snapshots[0].alpha_version
    
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
