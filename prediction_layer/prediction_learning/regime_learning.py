"""
Regime Learning Engine

Learns regime-specific performance and updates confidence engine.
Tracks which regimes are favorable for different prediction types.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

from prediction_layer.prediction_learning.prediction_history import PredictionMetadata
from prediction_layer.prediction_learning.prediction_result import PredictionResult

from utils.logger import get_logger

logger = get_logger("prediction_layer.prediction_learning.regime_learning")


@dataclass
class RegimePerformance:
    """Performance metrics for a market regime."""
    regime: str
    total_predictions: int
    successful_predictions: int
    failed_predictions: int
    win_rate: float
    average_return: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    last_updated: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "regime": self.regime,
            "total_predictions": self.total_predictions,
            "successful_predictions": self.successful_predictions,
            "failed_predictions": self.failed_predictions,
            "win_rate": round(self.win_rate, 4),
            "average_return": round(self.average_return, 4),
            "total_return": round(self.total_return, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class RegimeLearningUpdate:
    """Update to regime preferences based on learning."""
    regime: str
    old_accuracy: Optional[float]
    new_accuracy: float
    recommendation: str
    reason: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "regime": self.regime,
            "old_accuracy": round(self.old_accuracy, 4) if self.old_accuracy else None,
            "new_accuracy": round(self.new_accuracy, 4),
            "recommendation": self.recommendation,
            "reason": self.reason,
        }


class RegimeLearningEngine:
    """
    Learns regime-specific performance over time.
    
    Tracks:
    - Win rate for each regime
    - Average return in each regime
    - Sharpe ratio for each regime
    - Updates confidence engine with regime preferences
    """
    
    def __init__(
        self,
        min_predictions_for_learning: int = 30,
    ):
        """
        Initialize regime learning engine.
        
        Args:
            min_predictions_for_learning: Minimum predictions before regime is considered for learning
        """
        self.min_predictions_for_learning = min_predictions_for_learning
        self._regime_performance: Dict[str, RegimePerformance] = {}
        self._logger = get_logger("prediction_layer.prediction_learning.regime_learning")
    
    def update_from_predictions(
        self,
        predictions: List[PredictionMetadata],
        results: List[PredictionResult],
    ) -> List[RegimeLearningUpdate]:
        """
        Update regime learning from new predictions and results.
        
        Args:
            predictions: List of predictions
            results: List of results
            
        Returns:
            List of RegimeLearningUpdate objects
        """
        # Create prediction ID to result mapping
        result_map = {r.prediction_id: r for r in results}
        
        updates = []
        
        for pred in predictions:
            result = result_map.get(pred.prediction_id)
            if not result:
                continue
            
            # Update regime performance
            update = self._update_regime_performance(
                pred.market_regime,
                result.actual_return_percentage,
                result.actual_return_percentage > 0,
            )
            
            if update:
                updates.append(update)
        
        self._logger.info(f"Updated regime learning from {len(predictions)} predictions")
        
        return updates
    
    def _update_regime_performance(
        self,
        regime: str,
        return_percentage: float,
        is_successful: bool,
    ) -> Optional[RegimeLearningUpdate]:
        """
        Update performance for a single regime.
        
        Args:
            regime: Market regime
            return_percentage: Return percentage
            is_successful: Whether prediction was successful
            
        Returns:
            RegimeLearningUpdate if recommendation changed, None otherwise
        """
        # Get or create regime performance
        if regime not in self._regime_performance:
            self._regime_performance[regime] = RegimePerformance(
                regime=regime,
                total_predictions=0,
                successful_predictions=0,
                failed_predictions=0,
                win_rate=0.0,
                average_return=0.0,
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                last_updated=datetime.now(),
            )
        
        perf = self._regime_performance[regime]
        
        # Update counts
        perf.total_predictions += 1
        if is_successful:
            perf.successful_predictions += 1
        else:
            perf.failed_predictions += 1
        
        # Update returns
        perf.total_return += return_percentage
        perf.average_return = perf.total_return / perf.total_predictions
        
        # Update win rate
        perf.win_rate = perf.successful_predictions / perf.total_predictions
        
        # Update max drawdown
        perf.max_drawdown = min(perf.max_drawdown, return_percentage)
        
        # Update Sharpe ratio
        returns = self._get_recent_returns(regime, n=20)
        if len(returns) > 1:
            perf.sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
        
        perf.last_updated = datetime.now()
        
        # Check if regime has enough data for learning
        if perf.total_predictions < self.min_predictions_for_learning:
            return None
        
        # Generate recommendation
        old_accuracy = getattr(perf, 'accuracy_score', None)
        new_accuracy = perf.win_rate
        
        recommendation = self._generate_regime_recommendation(perf)
        
        # Store accuracy for comparison
        perf.accuracy_score = new_accuracy
        
        if old_accuracy is None or abs(new_accuracy - old_accuracy) > 0.05:
            return RegimeLearningUpdate(
                regime=regime,
                old_accuracy=old_accuracy,
                new_accuracy=new_accuracy,
                recommendation=recommendation,
                reason=self._generate_update_reason(perf, old_accuracy, new_accuracy),
            )
        
        return None
    
    def _get_recent_returns(self, regime: str, n: int = 20) -> List[float]:
        """
        Get recent returns for a regime (placeholder - would need storage).
        
        Args:
            regime: Market regime
            n: Number of recent returns
            
        Returns:
            List of recent returns
        """
        # This would need to be implemented with proper storage
        # For now, return empty list
        return []
    
    def _generate_regime_recommendation(self, perf: RegimePerformance) -> str:
        """
        Generate recommendation for regime.
        
        Args:
            perf: RegimePerformance object
            
        Returns:
            Recommendation string
        """
        if perf.win_rate >= 0.7 and perf.average_return >= 2.0:
            return "HIGHLY_FAVORABLE"
        elif perf.win_rate >= 0.6 and perf.average_return >= 1.0:
            return "FAVORABLE"
        elif perf.win_rate >= 0.5:
            return "NEUTRAL"
        elif perf.win_rate >= 0.4:
            return "CAUTION"
        else:
            return "AVOID"
    
    def _generate_update_reason(
        self,
        perf: RegimePerformance,
        old_accuracy: Optional[float],
        new_accuracy: float,
    ) -> str:
        """
        Generate reason for regime update.
        
        Args:
            perf: RegimePerformance object
            old_accuracy: Old accuracy
            new_accuracy: New accuracy
            
        Returns:
            Reason string
        """
        if old_accuracy is None:
            return f"Regime reached minimum sample size ({self.min_predictions_for_learning})"
        
        if new_accuracy > old_accuracy:
            return f"Improved performance: win rate {perf.win_rate:.1%}, avg return {perf.average_return:.2f}%"
        else:
            return f"Declining performance: win rate {perf.win_rate:.1%}, avg return {perf.average_return:.2f}%"
    
    def get_regime_performance(self, regime: str) -> Optional[RegimePerformance]:
        """
        Get performance for a specific regime.
        
        Args:
            regime: Market regime
            
        Returns:
            RegimePerformance if found, None otherwise
        """
        return self._regime_performance.get(regime)
    
    def get_all_regime_performance(self) -> Dict[str, RegimePerformance]:
        """
        Get performance for all regimes.
        
        Returns:
            Dictionary mapping regime to RegimePerformance
        """
        return self._regime_performance.copy()
    
    def get_best_regime(self) -> Optional[str]:
        """
        Get the best performing regime.
        
        Returns:
            Regime name if found, None otherwise
        """
        eligible_regimes = [
            (regime, perf)
            for regime, perf in self._regime_performance.items()
            if perf.total_predictions >= self.min_predictions_for_learning
        ]
        
        if not eligible_regimes:
            return None
        
        best_regime = max(
            eligible_regimes,
            key=lambda x: x[1].win_rate,
        )
        
        return best_regime[0]
    
    def get_worst_regime(self) -> Optional[str]:
        """
        Get the worst performing regime.
        
        Returns:
            Regime name if found, None otherwise
        """
        eligible_regimes = [
            (regime, perf)
            for regime, perf in self._regime_performance.items()
            if perf.total_predictions >= self.min_predictions_for_learning
        ]
        
        if not eligible_regimes:
            return None
        
        worst_regime = min(
            eligible_regimes,
            key=lambda x: x[1].win_rate,
        )
        
        return worst_regime[0]
    
    def get_regime_recommendation(self, regime: str) -> str:
        """
        Get recommendation for a specific regime.
        
        Args:
            regime: Market regime
            
        Returns:
            Recommendation string
        """
        perf = self._regime_performance.get(regime)
        
        if not perf or perf.total_predictions < self.min_predictions_for_learning:
            return "INSUFFICIENT_DATA"
        
        return self._generate_regime_recommendation(perf)
    
    def get_regime_rankings(self) -> List[tuple]:
        """
        Get regimes ranked by performance.
        
        Returns:
            List of (regime, win_rate, average_return) tuples sorted by win rate
        """
        eligible_regimes = [
            (regime, perf)
            for regime, perf in self._regime_performance.items()
            if perf.total_predictions >= self.min_predictions_for_learning
        ]
        
        sorted_regimes = sorted(
            eligible_regimes,
            key=lambda x: x[1].win_rate,
            reverse=True,
        )
        
        return [
            (regime, perf.win_rate, perf.average_return)
            for regime, perf in sorted_regimes
        ]
    
    def get_learning_summary(self) -> Dict:
        """
        Get summary of regime learning.
        
        Returns:
            Dictionary with learning summary
        """
        total_regimes = len(self._regime_performance)
        eligible_regimes = sum(
            1 for perf in self._regime_performance.values()
            if perf.total_predictions >= self.min_predictions_for_learning
        )
        
        if eligible_regimes == 0:
            return {
                "total_regimes": total_regimes,
                "eligible_regimes": eligible_regimes,
                "best_regime": None,
                "worst_regime": None,
                "regime_rankings": [],
            }
        
        best_regime = self.get_best_regime()
        worst_regime = self.get_worst_regime()
        regime_rankings = self.get_regime_rankings()
        
        return {
            "total_regimes": total_regimes,
            "eligible_regimes": eligible_regimes,
            "best_regime": best_regime,
            "worst_regime": worst_regime,
            "regime_rankings": regime_rankings,
        }


def update_regime_learning(
    predictions: List[PredictionMetadata],
    results: List[PredictionResult],
) -> List[RegimeLearningUpdate]:
    """
    Convenience function to update regime learning.
    
    Args:
        predictions: List of predictions
        results: List of results
        
    Returns:
        List of RegimeLearningUpdate objects
    """
    engine = RegimeLearningEngine()
    return engine.update_from_predictions(predictions, results)
