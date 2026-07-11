"""
Feature Learning Engine

Learns feature performance over time and updates feature rankings.
Tracks which features consistently lead to successful predictions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

from prediction_layer.prediction_learning.prediction_history import PredictionMetadata
from prediction_layer.prediction_learning.prediction_result import PredictionResult

from utils.logger import get_logger

logger = get_logger("prediction_layer.prediction_learning.feature_learning")


@dataclass
class FeaturePerformance:
    """Performance metrics for a feature."""
    feature_name: str
    total_uses: int
    successful_uses: int
    failed_uses: int
    win_rate: float
    average_return: float
    total_return: float
    last_updated: datetime
    decay_factor: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "feature_name": self.feature_name,
            "total_uses": self.total_uses,
            "successful_uses": self.successful_uses,
            "failed_uses": self.failed_uses,
            "win_rate": round(self.win_rate, 4),
            "average_return": round(self.average_return, 4),
            "total_return": round(self.total_return, 4),
            "last_updated": self.last_updated.isoformat(),
            "decay_factor": round(self.decay_factor, 4),
        }


@dataclass
class FeatureLearningUpdate:
    """Update to feature ranking based on learning."""
    feature_name: str
    old_rank: Optional[int]
    new_rank: Optional[int]
    old_importance: Optional[float]
    new_importance: float
    reason: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "feature_name": self.feature_name,
            "old_rank": self.old_rank,
            "new_rank": self.new_rank,
            "old_importance": round(self.old_importance, 4) if self.old_importance else None,
            "new_importance": round(self.new_importance, 4),
            "reason": self.reason,
        }


class FeatureLearningEngine:
    """
    Learns feature performance over time.
    
    Tracks:
    - Win rate for each feature
    - Average return when feature is used
    - Feature decay over time
    - Updates feature importance scores
    """
    
    def __init__(
        self,
        min_uses_for_learning: int = 30,
        decay_rate: float = 0.95,
    ):
        """
        Initialize feature learning engine.
        
        Args:
            min_uses_for_learning: Minimum uses before feature is considered for learning
            decay_rate: Decay rate for older data (0-1)
        """
        self.min_uses_for_learning = min_uses_for_learning
        self.decay_rate = decay_rate
        self._feature_performance: Dict[str, FeaturePerformance] = {}
        self._logger = get_logger("prediction_layer.prediction_learning.feature_learning")
    
    def update_from_predictions(
        self,
        predictions: List[PredictionMetadata],
        results: List[PredictionResult],
    ) -> List[FeatureLearningUpdate]:
        """
        Update feature learning from new predictions and results.
        
        Args:
            predictions: List of predictions
            results: List of results
            
        Returns:
            List of FeatureLearningUpdate objects
        """
        # Create prediction ID to result mapping
        result_map = {r.prediction_id: r for r in results}
        
        updates = []
        
        for pred in predictions:
            result = result_map.get(pred.prediction_id)
            if not result:
                continue
            
            # Update each feature used in this prediction
            for feature_name in pred.features.keys():
                update = self._update_feature_performance(
                    feature_name,
                    result.actual_return_percentage,
                    result.actual_return_percentage > 0,
                )
                
                if update:
                    updates.append(update)
        
        self._logger.info(f"Updated feature learning from {len(predictions)} predictions")
        
        return updates
    
    def _update_feature_performance(
        self,
        feature_name: str,
        return_percentage: float,
        is_successful: bool,
    ) -> Optional[FeatureLearningUpdate]:
        """
        Update performance for a single feature.
        
        Args:
            feature_name: Feature name
            return_percentage: Return percentage
            is_successful: Whether prediction was successful
            
        Returns:
            FeatureLearningUpdate if ranking changed, None otherwise
        """
        # Get or create feature performance
        if feature_name not in self._feature_performance:
            self._feature_performance[feature_name] = FeaturePerformance(
                feature_name=feature_name,
                total_uses=0,
                successful_uses=0,
                failed_uses=0,
                win_rate=0.0,
                average_return=0.0,
                total_return=0.0,
                last_updated=datetime.now(),
                decay_factor=1.0,
            )
        
        perf = self._feature_performance[feature_name]
        
        # Apply decay to old data
        perf.decay_factor *= self.decay_rate
        
        # Update counts
        perf.total_uses += 1
        if is_successful:
            perf.successful_uses += 1
        else:
            perf.failed_uses += 1
        
        # Update returns
        perf.total_return += return_percentage * perf.decay_factor
        perf.average_return = perf.total_return / perf.total_uses
        
        # Update win rate
        perf.win_rate = perf.successful_uses / perf.total_uses
        
        perf.last_updated = datetime.now()
        
        # Check if feature has enough data for learning
        if perf.total_uses < self.min_uses_for_learning:
            return None
        
        # Calculate new importance score
        new_importance = self._calculate_importance_score(perf)
        
        # Determine if ranking should change
        old_importance = getattr(perf, 'importance_score', None)
        perf.importance_score = new_importance
        
        if old_importance is None or abs(new_importance - old_importance) > 0.1:
            return FeatureLearningUpdate(
                feature_name=feature_name,
                old_rank=None,
                new_rank=None,
                old_importance=old_importance,
                new_importance=new_importance,
                reason=self._generate_update_reason(perf, old_importance, new_importance),
            )
        
        return None
    
    def _calculate_importance_score(self, perf: FeaturePerformance) -> float:
        """
        Calculate importance score from performance metrics.
        
        Args:
            perf: FeaturePerformance object
            
        Returns:
            Importance score (0-1)
        """
        # Base score from win rate
        importance = perf.win_rate * 0.5
        
        # Add average return component
        return_component = max(0.0, min(1.0, perf.average_return / 5.0))  # Normalize by 5%
        importance += return_component * 0.3
        
        # Add sample size component (more data = higher confidence)
        sample_size_component = min(1.0, perf.total_uses / 100.0)
        importance += sample_size_component * 0.2
        
        return max(0.0, min(1.0, importance))
    
    def _generate_update_reason(
        self,
        perf: FeaturePerformance,
        old_importance: Optional[float],
        new_importance: float,
    ) -> str:
        """
        Generate reason for feature update.
        
        Args:
            perf: FeaturePerformance object
            old_importance: Old importance score
            new_importance: New importance score
            
        Returns:
            Reason string
        """
        if old_importance is None:
            return f"Feature reached minimum sample size ({self.min_uses_for_learning})"
        
        if new_importance > old_importance:
            return f"Improved performance: win rate {perf.win_rate:.1%}, avg return {perf.average_return:.2f}%"
        else:
            return f"Declining performance: win rate {perf.win_rate:.1%}, avg return {perf.average_return:.2f}%"
    
    def get_feature_performance(self, feature_name: str) -> Optional[FeaturePerformance]:
        """
        Get performance for a specific feature.
        
        Args:
            feature_name: Feature name
            
        Returns:
            FeaturePerformance if found, None otherwise
        """
        return self._feature_performance.get(feature_name)
    
    def get_all_feature_performance(self) -> Dict[str, FeaturePerformance]:
        """
        Get performance for all features.
        
        Returns:
            Dictionary mapping feature name to FeaturePerformance
        """
        return self._feature_performance.copy()
    
    def get_top_features(self, n: int = 10) -> List[FeaturePerformance]:
        """
        Get top performing features.
        
        Args:
            n: Number of top features to return
            
        Returns:
            List of FeaturePerformance objects sorted by importance
        """
        # Filter features with enough data
        eligible_features = [
            perf for perf in self._feature_performance.values()
            if perf.total_uses >= self.min_uses_for_learning
        ]
        
        # Sort by importance score
        sorted_features = sorted(
            eligible_features,
            key=lambda x: getattr(x, 'importance_score', 0.0),
            reverse=True,
        )
        
        return sorted_features[:n]
    
    def get_worst_features(self, n: int = 10) -> List[FeaturePerformance]:
        """
        Get worst performing features.
        
        Args:
            n: Number of worst features to return
            
        Returns:
            List of FeaturePerformance objects sorted by importance
        """
        # Filter features with enough data
        eligible_features = [
            perf for perf in self._feature_performance.values()
            if perf.total_uses >= self.min_uses_for_learning
        ]
        
        # Sort by importance score
        sorted_features = sorted(
            eligible_features,
            key=lambda x: getattr(x, 'importance_score', 0.0),
        )
        
        return sorted_features[:n]
    
    def detect_feature_decay(self, threshold: float = 0.3) -> List[str]:
        """
        Detect features that have decayed in performance.
        
        Args:
            threshold: Threshold for decay detection
            
        Returns:
            List of feature names that have decayed
        """
        decayed_features = []
        
        for feature_name, perf in self._feature_performance.items():
            if perf.total_uses < self.min_uses_for_learning:
                continue
            
            importance = getattr(perf, 'importance_score', 0.0)
            if importance < threshold:
                decayed_features.append(feature_name)
        
        return decayed_features
    
    def get_learning_summary(self) -> Dict:
        """
        Get summary of feature learning.
        
        Returns:
            Dictionary with learning summary
        """
        total_features = len(self._feature_performance)
        eligible_features = sum(
            1 for perf in self._feature_performance.values()
            if perf.total_uses >= self.min_uses_for_learning
        )
        
        if eligible_features == 0:
            return {
                "total_features": total_features,
                "eligible_features": eligible_features,
                "top_features": [],
                "decayed_features": [],
            }
        
        top_features = self.get_top_features(5)
        decayed_features = self.detect_feature_decay()
        
        return {
            "total_features": total_features,
            "eligible_features": eligible_features,
            "top_features": [f.feature_name for f in top_features],
            "decayed_features": decayed_features,
        }


def update_feature_learning(
    predictions: List[PredictionMetadata],
    results: List[PredictionResult],
) -> List[FeatureLearningUpdate]:
    """
    Convenience function to update feature learning.
    
    Args:
        predictions: List of predictions
        results: List of results
        
    Returns:
        List of FeatureLearningUpdate objects
    """
    engine = FeatureLearningEngine()
    return engine.update_from_predictions(predictions, results)
