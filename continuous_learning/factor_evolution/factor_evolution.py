"""
Factor Evolution

Tracks lifetime performance of factors over time.
Detects alpha decay by monitoring performance changes.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

from continuous_learning.outcome_engine.trade_outcome import TradeOutcome
from continuous_learning.attribution_engine.factor_attribution import AttributionResult
from utils.logger import get_logger

logger = get_logger("continuous_learning.factor_evolution")


@dataclass
class FactorMetrics:
    """Metrics for a factor at a point in time."""
    factor_name: str
    timestamp: datetime
    information_coefficient: float
    sharpe_ratio: float
    win_rate: float
    avg_return: float
    total_trades: int
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate factor metrics.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check win rate is between 0 and 1
        if not (0.0 <= self.win_rate <= 1.0):
            errors.append(f"Win rate must be between 0 and 1, got {self.win_rate}")
        
        # Check total trades is non-negative
        if self.total_trades < 0:
            errors.append(f"Total trades must be non-negative, got {self.total_trades}")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "factor_name": self.factor_name,
            "timestamp": self.timestamp.isoformat(),
            "information_coefficient": round(self.information_coefficient, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "win_rate": round(self.win_rate, 4),
            "avg_return": round(self.avg_return, 4),
            "total_trades": self.total_trades,
        }


@dataclass
class AlphaDecayDetection:
    """Result of alpha decay detection."""
    factor_name: str
    has_decayed: bool
    decay_score: float  # 0-1, higher means more decay
    current_ic: float
    baseline_ic: float
    ic_change: float
    recommended_action: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "factor_name": self.factor_name,
            "has_decayed": self.has_decayed,
            "decay_score": round(self.decay_score, 4),
            "current_ic": round(self.current_ic, 4),
            "baseline_ic": round(self.baseline_ic, 4),
            "ic_change": round(self.ic_change, 4),
            "recommended_action": self.recommended_action,
        }


class FactorEvolution:
    """
    Tracks lifetime performance of factors over time.
    
    Detects:
    - Alpha decay
    - Performance trends
    - Seasonal patterns
    """
    
    def __init__(
        self,
        decay_threshold: float = 0.3,
        min_trades_for_decay: int = 50,
    ):
        """
        Initialize factor evolution tracker.
        
        Args:
            decay_threshold: IC change threshold for decay detection
            min_trades_for_decay: Minimum trades before detecting decay
        """
        self.decay_threshold = decay_threshold
        self.min_trades_for_decay = min_trades_for_decay
        
        # Factor history: factor_name -> list of FactorMetrics
        self._history: Dict[str, List[FactorMetrics]] = {}
        
        # Factor returns: factor_name -> list of returns
        self._returns: Dict[str, List[float]] = {}
        
        # Factor correctness: factor_name -> list of bool
        self._correctness: Dict[str, List[bool]] = {}
        
        self._logger = get_logger("continuous_learning.factor_evolution")
    
    def update(
        self,
        trade_outcome: TradeOutcome,
        attribution: Optional[AttributionResult] = None,
    ) -> None:
        """
        Update factor evolution with completed trade.
        
        Args:
            trade_outcome: TradeOutcome
            attribution: Optional AttributionResult
        """
        if not attribution:
            return
        
        all_contributors = (
            attribution.positive_contributors +
            attribution.negative_contributors +
            attribution.neutral_contributors
        )
        
        for contributor in all_contributors:
            factor_name = contributor.factor_name
            
            # Initialize factor tracking
            if factor_name not in self._history:
                self._history[factor_name] = []
                self._returns[factor_name] = []
                self._correctness[factor_name] = []
            
            # Store return
            self._returns[factor_name].append(contributor.contribution_score)
            
            # Store correctness
            self._correctness[factor_name].append(contributor.was_correct)
            
            # Calculate current metrics
            if len(self._returns[factor_name]) >= 10:
                metrics = self._calculate_metrics(factor_name)
                self._history[factor_name].append(metrics)
    
    def _calculate_metrics(self, factor_name: str) -> FactorMetrics:
        """
        Calculate current metrics for a factor.
        
        Args:
            factor_name: Factor name
            
        Returns:
            FactorMetrics
        """
        returns = self._returns[factor_name]
        correctness = self._correctness[factor_name]
        
        # Calculate IC (simplified as mean return normalized)
        ic = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
        
        # Calculate Sharpe
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0.0
        
        # Calculate win rate
        win_rate = sum(correctness) / len(correctness) if correctness else 0.0
        
        # Calculate average return
        avg_return = np.mean(returns)
        
        return FactorMetrics(
            factor_name=factor_name,
            timestamp=datetime.now(),
            information_coefficient=ic,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            avg_return=avg_return,
            total_trades=len(returns),
        )
    
    def detect_decay(
        self,
        factor_name: str,
        window_days: int = 90,
    ) -> Optional[AlphaDecayDetection]:
        """
        Detect alpha decay for a factor.
        
        Args:
            factor_name: Factor name
            window_days: Lookback window in days
            
        Returns:
            AlphaDecayDetection or None
        """
        if factor_name not in self._history:
            return None
        
        history = self._history[factor_name]
        
        if len(history) < self.min_trades_for_decay:
            return None
        
        # Get current metrics
        current_metrics = history[-1]
        current_ic = current_metrics.information_coefficient
        
        # Get baseline metrics (first 30% of history)
        baseline_start = 0
        baseline_end = max(1, len(history) // 3)
        baseline_metrics = history[baseline_end - 1]
        baseline_ic = baseline_metrics.information_coefficient
        
        # Calculate IC change
        ic_change = current_ic - baseline_ic
        
        # Calculate decay score
        decay_score = 0.0
        if ic_change < 0:
            decay_score = abs(ic_change) / (abs(baseline_ic) + 0.01)
        
        # Determine if decayed
        has_decayed = decay_score > self.decay_threshold
        
        # Recommend action
        if has_decayed:
            if decay_score > 0.5:
                recommended_action = "REMOVE"
            else:
                recommended_action = "REDUCE_WEIGHT"
        else:
            recommended_action = "KEEP"
        
        return AlphaDecayDetection(
            factor_name=factor_name,
            has_decayed=has_decayed,
            decay_score=decay_score,
            current_ic=current_ic,
            baseline_ic=baseline_ic,
            ic_change=ic_change,
            recommended_action=recommended_action,
        )
    
    def get_factor_history(
        self,
        factor_name: str,
    ) -> List[FactorMetrics]:
        """
        Get performance history for a factor.
        
        Args:
            factor_name: Factor name
            
        Returns:
            List of FactorMetrics
        """
        return self._history.get(factor_name, [])
    
    def get_all_factors(self) -> List[str]:
        """
        Get list of all tracked factors.
        
        Returns:
            List of factor names
        """
        return list(self._history.keys())
    
    def detect_all_decay(self) -> Dict[str, AlphaDecayDetection]:
        """
        Detect decay for all factors.
        
        Returns:
            Dictionary mapping factor names to AlphaDecayDetection
        """
        results = {}
        
        for factor_name in self.get_all_factors():
            detection = self.detect_decay(factor_name)
            if detection:
                results[factor_name] = detection
        
        return results
    
    def get_performance_trend(
        self,
        factor_name: str,
        metric: str = "information_coefficient",
    ) -> str:
        """
        Get performance trend for a factor.
        
        Args:
            factor_name: Factor name
            metric: Metric to analyze
            
        Returns:
            Trend: "IMPROVING", "DECLINING", or "STABLE"
        """
        history = self.get_factor_history(factor_name)
        
        if len(history) < 10:
            return "STABLE"
        
        # Get recent and historical values
        recent_values = [getattr(m, metric) for m in history[-10:]]
        historical_values = [getattr(m, metric) for m in history[:-10]]
        
        if not historical_values:
            return "STABLE"
        
        recent_avg = np.mean(recent_values)
        historical_avg = np.mean(historical_values)
        
        change = (recent_avg - historical_avg) / (abs(historical_avg) + 0.01)
        
        if change > 0.1:
            return "IMPROVING"
        elif change < -0.1:
            return "DECLINING"
        else:
            return "STABLE"
    
    def generate_report(self) -> str:
        """
        Generate human-readable report.
        
        Returns:
            Formatted report string
        """
        lines = []
        
        lines.append("FACTOR EVOLUTION REPORT")
        lines.append("=" * 50)
        
        # Detect decay for all factors
        decay_results = self.detect_all_decay()
        
        if decay_results:
            lines.append("\nALPHA DECAY DETECTION")
            lines.append("-" * 40)
            
            for factor_name, detection in decay_results.items():
                if detection.has_decayed:
                    lines.append(f"  {factor_name}: DECAYED")
                    lines.append(f"    Decay Score: {detection.decay_score:.2%}")
                    lines.append(f"    IC Change: {detection.ic_change:.3f}")
                    lines.append(f"    Action: {detection.recommended_action}")
                else:
                    lines.append(f"  {factor_name}: STABLE")
        
        lines.append("\nFACTOR PERFORMANCE TRENDS")
        lines.append("-" * 40)
        
        for factor_name in self.get_all_factors():
            trend = self.get_performance_trend(factor_name)
            history = self.get_factor_history(factor_name)
            
            if history:
                current = history[-1]
                lines.append(f"  {factor_name}:")
                lines.append(f"    Trend: {trend}")
                lines.append(f"    Current IC: {current.information_coefficient:.3f}")
                lines.append(f"    Win Rate: {current.win_rate:.2%}")
                lines.append(f"    Total Trades: {current.total_trades}")
        
        return "\n".join(lines)
    
    def clear_old_data(self, days: int = 365) -> int:
        """
        Clear old data from history.
        
        Args:
            days: Number of days to keep
            
        Returns:
            Number of records cleared
        """
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(days=days)
        
        cleared = 0
        
        for factor_name in self._history:
            original_length = len(self._history[factor_name])
            self._history[factor_name] = [
                m for m in self._history[factor_name]
                if m.timestamp > cutoff
            ]
            cleared += original_length - len(self._history[factor_name])
        
        self._logger.info(f"Cleared {cleared} old records")
        return cleared


# Global instance
_global_factor_evolution = None


def get_factor_evolution() -> FactorEvolution:
    """
    Get global factor evolution instance.
    
    Returns:
        FactorEvolution instance
    """
    global _global_factor_evolution
    if _global_factor_evolution is None:
        _global_factor_evolution = FactorEvolution()
    return _global_factor_evolution
