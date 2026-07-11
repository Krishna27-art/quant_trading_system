"""
Signal Learning Engine

Learns signal performance over time and updates signal weights.
Tracks which signals consistently lead to successful predictions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

from prediction_layer.prediction_learning.prediction_history import PredictionMetadata
from prediction_layer.prediction_learning.prediction_result import PredictionResult

from utils.logger import get_logger

logger = get_logger("prediction_layer.prediction_learning.signal_learning")


@dataclass
class SignalPerformance:
    """Performance metrics for a signal."""
    signal_name: str
    signal_type: str
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
            "signal_name": self.signal_name,
            "signal_type": self.signal_type,
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
class SignalLearningUpdate:
    """Update to signal weights based on learning."""
    signal_name: str
    signal_type: str
    old_weight: Optional[float]
    new_weight: float
    reason: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "signal_name": self.signal_name,
            "signal_type": self.signal_type,
            "old_weight": round(self.old_weight, 4) if self.old_weight else None,
            "new_weight": round(self.new_weight, 4),
            "reason": self.reason,
        }


class SignalLearningEngine:
    """
    Learns signal performance over time.
    
    Tracks:
    - Win rate for each signal type
    - Average return when signal is used
    - Signal decay over time
    - Updates signal weights
    """
    
    def __init__(
        self,
        min_uses_for_learning: int = 30,
        decay_rate: float = 0.95,
    ):
        """
        Initialize signal learning engine.
        
        Args:
            min_uses_for_learning: Minimum uses before signal is considered for learning
            decay_rate: Decay rate for older data (0-1)
        """
        self.min_uses_for_learning = min_uses_for_learning
        self.decay_rate = decay_rate
        self._signal_performance: Dict[str, SignalPerformance] = {}
        self._signal_weights: Dict[str, float] = {}
        self._logger = get_logger("prediction_layer.prediction_learning.signal_learning")
    
    def update_from_predictions(
        self,
        predictions: List[PredictionMetadata],
        results: List[PredictionResult],
    ) -> List[SignalLearningUpdate]:
        """
        Update signal learning from new predictions and results.
        
        Args:
            predictions: List of predictions
            results: List of results
            
        Returns:
            List of SignalLearningUpdate objects
        """
        # Create prediction ID to result mapping
        result_map = {r.prediction_id: r for r in results}
        
        updates = []
        
        for pred in predictions:
            result = result_map.get(pred.prediction_id)
            if not result:
                continue
            
            # Update each signal used in this prediction
            for signal in pred.signals:
                signal_name = signal.get("name", "unknown")
                signal_type = signal.get("type", "unknown")
                
                update = self._update_signal_performance(
                    signal_name,
                    signal_type,
                    result.actual_return_percentage,
                    result.actual_return_percentage > 0,
                )
                
                if update:
                    updates.append(update)
        
        self._logger.info(f"Updated signal learning from {len(predictions)} predictions")
        
        return updates
    
    def _update_signal_performance(
        self,
        signal_name: str,
        signal_type: str,
        return_percentage: float,
        is_successful: bool,
    ) -> Optional[SignalLearningUpdate]:
        """
        Update performance for a single signal.
        
        Args:
            signal_name: Signal name
            signal_type: Signal type
            return_percentage: Return percentage
            is_successful: Whether prediction was successful
            
        Returns:
            SignalLearningUpdate if weight changed, None otherwise
        """
        # Get or create signal performance
        if signal_name not in self._signal_performance:
            self._signal_performance[signal_name] = SignalPerformance(
                signal_name=signal_name,
                signal_type=signal_type,
                total_uses=0,
                successful_uses=0,
                failed_uses=0,
                win_rate=0.0,
                average_return=0.0,
                total_return=0.0,
                last_updated=datetime.now(),
                decay_factor=1.0,
            )
            
            # Initialize weight
            self._signal_weights[signal_name] = 1.0
        
        perf = self._signal_performance[signal_name]
        
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
        
        # Check if signal has enough data for learning
        if perf.total_uses < self.min_uses_for_learning:
            return None
        
        # Calculate new weight
        new_weight = self._calculate_signal_weight(perf)
        
        # Determine if weight should change
        old_weight = self._signal_weights[signal_name]
        
        if abs(new_weight - old_weight) > 0.1:
            self._signal_weights[signal_name] = new_weight
            
            return SignalLearningUpdate(
                signal_name=signal_name,
                signal_type=signal_type,
                old_weight=old_weight,
                new_weight=new_weight,
                reason=self._generate_update_reason(perf, old_weight, new_weight),
            )
        
        return None
    
    def _calculate_signal_weight(self, perf: SignalPerformance) -> float:
        """
        Calculate signal weight from performance metrics.
        
        Args:
            perf: SignalPerformance object
            
        Returns:
            Signal weight (0-2, where 1.0 is neutral)
        """
        # Base weight from win rate
        # Win rate > 0.5 increases weight, < 0.5 decreases weight
        weight = 1.0 + (perf.win_rate - 0.5) * 2.0
        
        # Add average return component
        return_component = max(-1.0, min(1.0, perf.average_return / 5.0))  # Normalize by 5%
        weight += return_component * 0.5
        
        # Add sample size component (more data = higher confidence in weight)
        sample_size_component = min(1.0, perf.total_uses / 100.0)
        weight = 1.0 + (weight - 1.0) * sample_size_component
        
        # Clamp weight to reasonable range
        return max(0.1, min(2.0, weight))
    
    def _generate_update_reason(
        self,
        perf: SignalPerformance,
        old_weight: float,
        new_weight: float,
    ) -> str:
        """
        Generate reason for signal update.
        
        Args:
            perf: SignalPerformance object
            old_weight: Old weight
            new_weight: New weight
            
        Returns:
            Reason string
        """
        if new_weight > old_weight:
            return f"Improved performance: win rate {perf.win_rate:.1%}, avg return {perf.average_return:.2f}%"
        else:
            return f"Declining performance: win rate {perf.win_rate:.1%}, avg return {perf.average_return:.2f}%"
    
    def get_signal_performance(self, signal_name: str) -> Optional[SignalPerformance]:
        """
        Get performance for a specific signal.
        
        Args:
            signal_name: Signal name
            
        Returns:
            SignalPerformance if found, None otherwise
        """
        return self._signal_performance.get(signal_name)
    
    def get_all_signal_performance(self) -> Dict[str, SignalPerformance]:
        """
        Get performance for all signals.
        
        Returns:
            Dictionary mapping signal name to SignalPerformance
        """
        return self._signal_performance.copy()
    
    def get_signal_weights(self) -> Dict[str, float]:
        """
        Get current signal weights.
        
        Returns:
            Dictionary mapping signal name to weight
        """
        return self._signal_weights.copy()
    
    def get_top_signals(self, n: int = 10) -> List[SignalPerformance]:
        """
        Get top performing signals.
        
        Args:
            n: Number of top signals to return
            
        Returns:
            List of SignalPerformance objects sorted by weight
        """
        # Filter signals with enough data
        eligible_signals = [
            perf for perf in self._signal_performance.values()
            if perf.total_uses >= self.min_uses_for_learning
        ]
        
        # Sort by weight
        sorted_signals = sorted(
            eligible_signals,
            key=lambda x: self._signal_weights[x.signal_name],
            reverse=True,
        )
        
        return sorted_signals[:n]
    
    def get_worst_signals(self, n: int = 10) -> List[SignalPerformance]:
        """
        Get worst performing signals.
        
        Args:
            n: Number of worst signals to return
            
        Returns:
            List of SignalPerformance objects sorted by weight
        """
        # Filter signals with enough data
        eligible_signals = [
            perf for perf in self._signal_performance.values()
            if perf.total_uses >= self.min_uses_for_learning
        ]
        
        # Sort by weight
        sorted_signals = sorted(
            eligible_signals,
            key=lambda x: self._signal_weights[x.signal_name],
        )
        
        return sorted_signals[:n]
    
    def detect_signal_decay(self, threshold: float = 0.5) -> List[str]:
        """
        Detect signals that have decayed in performance.
        
        Args:
            threshold: Threshold for decay detection
            
        Returns:
            List of signal names that have decayed
        """
        decayed_signals = []
        
        for signal_name, weight in self._signal_weights.items():
            perf = self._signal_performance.get(signal_name)
            if not perf or perf.total_uses < self.min_uses_for_learning:
                continue
            
            if weight < threshold:
                decayed_signals.append(signal_name)
        
        return decayed_signals
    
    def get_signal_performance_by_type(self, signal_type: str) -> List[SignalPerformance]:
        """
        Get performance for signals of a specific type.
        
        Args:
            signal_type: Signal type
            
        Returns:
            List of SignalPerformance objects
        """
        return [
            perf for perf in self._signal_performance.values()
            if perf.signal_type == signal_type
        ]
    
    def get_learning_summary(self) -> Dict:
        """
        Get summary of signal learning.
        
        Returns:
            Dictionary with learning summary
        """
        total_signals = len(self._signal_performance)
        eligible_signals = sum(
            1 for perf in self._signal_performance.values()
            if perf.total_uses >= self.min_uses_for_learning
        )
        
        if eligible_signals == 0:
            return {
                "total_signals": total_signals,
                "eligible_signals": eligible_signals,
                "top_signals": [],
                "decayed_signals": [],
                "signal_type_performance": {},
            }
        
        top_signals = self.get_top_signals(5)
        decayed_signals = self.detect_signal_decay()
        
        # Calculate performance by signal type
        signal_type_performance = {}
        for perf in self._signal_performance.values():
            if perf.signal_type not in signal_type_performance:
                signal_type_performance[perf.signal_type] = {
                    "total_uses": 0,
                    "win_rate": 0.0,
                    "average_return": 0.0,
                }
            
            signal_type_performance[perf.signal_type]["total_uses"] += perf.total_uses
            signal_type_performance[perf.signal_type]["win_rate"] += perf.win_rate
            signal_type_performance[perf.signal_type]["average_return"] += perf.average_return
        
        # Average by type
        for signal_type, data in signal_type_performance.items():
            count = sum(
                1 for perf in self._signal_performance.values()
                if perf.signal_type == signal_type
            )
            data["win_rate"] /= count
            data["average_return"] /= count
        
        return {
            "total_signals": total_signals,
            "eligible_signals": eligible_signals,
            "top_signals": [s.signal_name for s in top_signals],
            "decayed_signals": decayed_signals,
            "signal_type_performance": signal_type_performance,
        }


def update_signal_learning(
    predictions: List[PredictionMetadata],
    results: List[PredictionResult],
) -> List[SignalLearningUpdate]:
    """
    Convenience function to update signal learning.
    
    Args:
        predictions: List of predictions
        results: List of results
        
    Returns:
        List of SignalLearningUpdate objects
    """
    engine = SignalLearningEngine()
    return engine.update_from_predictions(predictions, results)
