"""
Feature Importance Tracker

Tracks feature importance from experiments.
Automatically saves top features, worst features, SHAP, and permutation importance.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from research_platform.experiments.base import FeatureImportance
from utils.logger import get_logger

logger = get_logger("experiments.feature_importance")


class FeatureImportanceTracker:
    """
    Feature Importance Tracker.
    
    Tracks:
    - Top features by importance
    - Worst features by importance
    - Feature importance methods (SHAP, permutation, gain)
    - Feature importance history
    """
    
    def __init__(self):
        """Initialize feature importance tracker."""
        self.importance_records: Dict[str, FeatureImportance] = {}
        self._logger = get_logger("experiments.feature_importance")
    
    def log_feature_importance(
        self,
        experiment_id: str,
        feature_importance: Dict[str, float],
        method: str = "gain",
        top_n: int = 10,
        worst_n: int = 5,
    ) -> FeatureImportance:
        """
        Log feature importance.
        
        Args:
            experiment_id: Experiment ID
            feature_importance: Dictionary mapping feature names to importance scores
            method: Method used (shap, permutation, gain)
            top_n: Number of top features to store
            worst_n: Number of worst features to store
            
        Returns:
            FeatureImportance object
        """
        importance_id = f"FI-{uuid.uuid4().hex[:8].upper()}"
        
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        # Extract top and worst features
        top_features = dict(sorted_features[:top_n])
        worst_features = dict(sorted_features[-worst_n:]) if len(sorted_features) >= worst_n else {}
        
        importance = FeatureImportance(
            importance_id=importance_id,
            experiment_id=experiment_id,
            top_features=top_features,
            worst_features=worst_features,
            method=method,
        )
        
        self.importance_records[importance_id] = importance
        
        self._logger.info(
            f"Logged feature importance {importance_id}: "
            f"{len(feature_importance)} features, method={method}"
        )
        
        return importance
    
    def get_importance(self, importance_id: str) -> Optional[FeatureImportance]:
        """Get feature importance by ID."""
        return self.importance_records.get(importance_id)
    
    def get_importance_by_experiment(self, experiment_id: str) -> List[FeatureImportance]:
        """Get all feature importance records for an experiment."""
        return [
            importance for importance in self.importance_records.values()
            if importance.experiment_id == experiment_id
        ]
    
    def get_feature_usage_stats(self) -> Dict[str, int]:
        """
        Get statistics about feature appearance in top features.
        
        Returns:
            Dictionary mapping feature names to appearance counts
        """
        feature_usage = {}
        
        for importance in self.importance_records.values():
            for feature in importance.top_features.keys():
                feature_usage[feature] = feature_usage.get(feature, 0) + 1
        
        return feature_usage
    
    def get_most_important_features(self, top_n: int = 20) -> List[tuple[str, int]]:
        """
        Get most frequently appearing features in top importance.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            List of (feature_name, appearance_count) tuples
        """
        feature_usage = self.get_feature_usage_stats()
        
        sorted_features = sorted(
            feature_usage.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        return sorted_features[:top_n]
    
    def get_least_important_features(self, bottom_n: int = 20) -> List[tuple[str, int]]:
        """
        Get most frequently appearing features in worst importance.
        
        Args:
            bottom_n: Number of bottom features to return
            
        Returns:
            List of (feature_name, appearance_count) tuples
        """
        feature_usage = {}
        
        for importance in self.importance_records.values():
            for feature in importance.worst_features.keys():
                feature_usage[feature] = feature_usage.get(feature, 0) + 1
        
        sorted_features = sorted(
            feature_usage.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        return sorted_features[:bottom_n]
    
    def compare_feature_importance(
        self,
        importance_id_1: str,
        importance_id_2: str,
    ) -> Optional[Dict]:
        """
        Compare two feature importance records.
        
        Args:
            importance_id_1: First importance ID
            importance_id_2: Second importance ID
            
        Returns:
            Dictionary with comparison results
        """
        importance_1 = self.importance_records.get(importance_id_1)
        importance_2 = self.importance_records.get(importance_id_2)
        
        if not importance_1 or not importance_2:
            return None
        
        comparison = {
            'importance_1': importance_1.to_dict(),
            'importance_2': importance_2.to_dict(),
            'top_features_changes': self._compare_top_features(
                importance_1.top_features,
                importance_2.top_features,
            ),
            'method_changed': importance_1.method != importance_2.method,
        }
        
        return comparison
    
    def _compare_top_features(
        self,
        top_1: Dict[str, float],
        top_2: Dict[str, float],
    ) -> Dict:
        """Compare top features between two records."""
        set_1 = set(top_1.keys())
        set_2 = set(top_2.keys())
        
        return {
            'new_top_features': list(set_2 - set_1),
            'dropped_top_features': list(set_1 - set_2),
            'common_top_features': list(set_1 & set_2),
            'rank_changes': self._calculate_rank_changes(top_1, top_2),
        }
    
    def _calculate_rank_changes(
        self,
        top_1: Dict[str, float],
        top_2: Dict[str, float],
    ) -> List[Dict]:
        """Calculate rank changes for common features."""
        rank_changes = []
        
        common_features = set(top_1.keys()) & set(top_2.keys())
        
        for feature in common_features:
            rank_1 = list(top_1.keys()).index(feature) + 1
            rank_2 = list(top_2.keys()).index(feature) + 1
            
            rank_changes.append({
                'feature': feature,
                'old_rank': rank_1,
                'new_rank': rank_2,
                'change': rank_2 - rank_1,
            })
        
        return rank_changes
    
    def get_feature_stability_score(
        self,
        feature: str,
        window: int = 10,
    ) -> float:
        """
        Calculate stability score for a feature across recent experiments.
        
        Args:
            feature: Feature name
            window: Number of recent experiments to consider
            
        Returns:
            Stability score (0 to 1)
        """
        recent_importance = list(self.importance_records.values())[-window:]
        
        if not recent_importance:
            return 0.0
        
        appearances = sum(
            1 for imp in recent_importance
            if feature in imp.top_features
        )
        
        stability = appearances / len(recent_importance)
        return stability
    
    def get_stable_features(
        self,
        min_stability: float = 0.7,
        window: int = 10,
    ) -> List[tuple[str, float]]:
        """
        Get features that are stable across recent experiments.
        
        Args:
            min_stability: Minimum stability score
            window: Number of recent experiments to consider
            
        Returns:
            List of (feature_name, stability_score) tuples
        """
        all_features = set()
        
        for imp in list(self.importance_records.values())[-window:]:
            all_features.update(imp.top_features.keys())
        
        stable_features = []
        
        for feature in all_features:
            stability = self.get_feature_stability_score(feature, window)
            if stability >= min_stability:
                stable_features.append((feature, stability))
        
        stable_features.sort(key=lambda x: x[1], reverse=True)
        
        return stable_features
