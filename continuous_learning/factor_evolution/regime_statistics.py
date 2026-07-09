"""
Regime Statistics

Tracks factor performance across different market regimes.
Updates statistics after every completed trade.
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
class RegimePerformance:
    """Performance of a factor in a specific regime."""
    regime: str
    factor_name: str
    trades: int
    wins: int
    losses: int
    sharpe: float
    information_coefficient: float
    win_rate: float
    avg_return: float
    last_updated: datetime
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate regime performance.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check trades is non-negative
        if self.trades < 0:
            errors.append(f"Trades must be non-negative, got {self.trades}")
        
        # Check win rate is between 0 and 1
        if not (0.0 <= self.win_rate <= 1.0):
            errors.append(f"Win rate must be between 0 and 1, got {self.win_rate}")
        
        # Check wins + losses equals trades
        if self.wins + self.losses != self.trades:
            errors.append(f"Wins + losses must equal trades: {self.wins} + {self.losses} != {self.trades}")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "regime": self.regime,
            "factor_name": self.factor_name,
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "sharpe": round(self.sharpe, 4),
            "information_coefficient": round(self.information_coefficient, 4),
            "win_rate": round(self.win_rate, 4),
            "avg_return": round(self.avg_return, 4),
            "last_updated": self.last_updated.isoformat(),
        }


class RegimeStatistics:
    """
    Tracks factor performance across different market regimes.
    
    Maintains:
    - Trades per regime
    - Wins/losses per regime
    - Sharpe ratio per regime
    - Information Coefficient per regime
    - Win rate per regime
    """
    
    def __init__(self):
        """Initialize regime statistics."""
        self._stats: Dict[str, Dict[str, RegimePerformance]] = {}  # regime -> factor -> performance
        self._returns: Dict[str, Dict[str, List[float]]] = {}  # regime -> factor -> returns
        self._logger = get_logger("continuous_learning.factor_evolution")
    
    def update(
        self,
        trade_outcome: TradeOutcome,
        attribution: Optional[AttributionResult] = None,
        market_regime: str = "unknown",
    ) -> None:
        """
        Update statistics with completed trade.
        
        Args:
            trade_outcome: TradeOutcome
            attribution: Optional AttributionResult
            market_regime: Market regime
        """
        if market_regime not in self._stats:
            self._stats[market_regime] = {}
            self._returns[market_regime] = {}
        
        # Update overall regime statistics
        self._update_regime_stats(trade_outcome, market_regime)
        
        # Update factor-specific statistics if attribution available
        if attribution:
            self._update_factor_stats(attribution, trade_outcome, market_regime)
    
    def _update_regime_stats(
        self,
        trade_outcome: TradeOutcome,
        regime: str,
    ) -> None:
        """
        Update overall regime statistics.
        
        Args:
            trade_outcome: TradeOutcome
            regime: Market regime
        """
        # Store return for Sharpe calculation
        if regime not in self._returns:
            self._returns[regime] = {}
        
        if "overall" not in self._returns[regime]:
            self._returns[regime]["overall"] = []
        
        self._returns[regime]["overall"].append(trade_outcome.resolved_outcome.return_percentage)
        
        # Update regime performance
        if "overall" not in self._stats[regime]:
            self._stats[regime]["overall"] = RegimePerformance(
                regime=regime,
                factor_name="overall",
                trades=0,
                wins=0,
                losses=0,
                sharpe=0.0,
                information_coefficient=0.0,
                win_rate=0.0,
                avg_return=0.0,
                last_updated=datetime.now(),
            )
        
        perf = self._stats[regime]["overall"]
        perf.trades += 1
        
        if trade_outcome.is_successful:
            perf.wins += 1
        else:
            perf.losses += 1
        
        # Recalculate metrics
        perf.win_rate = perf.wins / perf.trades if perf.trades > 0 else 0.0
        perf.avg_return = np.mean(self._returns[regime]["overall"])
        
        if len(self._returns[regime]["overall"]) > 1:
            perf.sharpe = np.mean(self._returns[regime]["overall"]) / np.std(self._returns[regime]["overall"]) * np.sqrt(252)
        
        perf.last_updated = datetime.now()
    
    def _update_factor_stats(
        self,
        attribution: AttributionResult,
        trade_outcome: TradeOutcome,
        regime: str,
    ) -> None:
        """
        Update factor-specific statistics.
        
        Args:
            attribution: AttributionResult
            trade_outcome: TradeOutcome
            regime: Market regime
        """
        all_contributors = (
            attribution.positive_contributors +
            attribution.negative_contributors +
            attribution.neutral_contributors
        )
        
        for contributor in all_contributors:
            factor_name = contributor.factor_name
            
            # Initialize factor stats if needed
            if factor_name not in self._stats[regime]:
                self._stats[regime][factor_name] = RegimePerformance(
                    regime=regime,
                    factor_name=factor_name,
                    trades=0,
                    wins=0,
                    losses=0,
                    sharpe=0.0,
                    information_coefficient=0.0,
                    win_rate=0.0,
                    avg_return=0.0,
                    last_updated=datetime.now(),
                )
            
            if factor_name not in self._returns[regime]:
                self._returns[regime][factor_name] = []
            
            # Store return for this factor
            factor_return = contributor.contribution_score
            self._returns[regime][factor_name].append(factor_return)
            
            # Update performance
            perf = self._stats[regime][factor_name]
            perf.trades += 1
            
            if contributor.was_correct:
                perf.wins += 1
            else:
                perf.losses += 1
            
            # Recalculate metrics
            perf.win_rate = perf.wins / perf.trades if perf.trades > 0 else 0.0
            perf.avg_return = np.mean(self._returns[regime][factor_name])
            
            if len(self._returns[regime][factor_name]) > 1:
                perf.sharpe = np.mean(self._returns[regime][factor_name]) / np.std(self._returns[regime][factor_name]) * np.sqrt(252)
            
            # Calculate IC (simplified as correlation with actual return)
            if len(self._returns[regime][factor_name]) > 10:
                actual_returns = [trade_outcome.resolved_outcome.return_percentage] * len(self._returns[regime][factor_name])
                perf.information_coefficient = np.corrcoef(
                    self._returns[regime][factor_name],
                    actual_returns[:len(self._returns[regime][factor_name])]
                )[0, 1]
            
            perf.last_updated = datetime.now()
    
    def get_regime_performance(
        self,
        regime: str,
        factor_name: Optional[str] = None,
    ) -> Optional[RegimePerformance]:
        """
        Get performance for a regime and factor.
        
        Args:
            regime: Market regime
            factor_name: Optional factor name
            
        Returns:
            RegimePerformance or None
        """
        if regime not in self._stats:
            return None
        
        if factor_name:
            return self._stats[regime].get(factor_name)
        else:
            return self._stats[regime].get("overall")
    
    def get_all_regimes(self) -> List[str]:
        """
        Get list of all tracked regimes.
        
        Returns:
            List of regime names
        """
        return list(self._stats.keys())
    
    def get_factor_performance_by_regime(
        self,
        factor_name: str,
    ) -> Dict[str, RegimePerformance]:
        """
        Get factor performance across all regimes.
        
        Args:
            factor_name: Factor name
            
        Returns:
            Dictionary mapping regime to RegimePerformance
        """
        result = {}
        
        for regime, factors in self._stats.items():
            if factor_name in factors:
                result[regime] = factors[factor_name]
        
        return result
    
    def get_best_regime_for_factor(
        self,
        factor_name: str,
        metric: str = "win_rate",
    ) -> Optional[tuple]:
        """
        Get best regime for a factor by metric.
        
        Args:
            factor_name: Factor name
            metric: Metric to use ("win_rate", "sharpe", "avg_return")
            
        Returns:
            Tuple of (regime, value) or None
        """
        regime_perfs = self.get_factor_performance_by_regime(factor_name)
        
        if not regime_perfs:
            return None
        
        best_regime = None
        best_value = float("-inf")
        
        for regime, perf in regime_perfs.items():
            value = getattr(perf, metric, 0.0)
            if value > best_value:
                best_value = value
                best_regime = regime
        
        return (best_regime, best_value) if best_regime else None
    
    def get_worst_regime_for_factor(
        self,
        factor_name: str,
        metric: str = "win_rate",
    ) -> Optional[tuple]:
        """
        Get worst regime for a factor by metric.
        
        Args:
            factor_name: Factor name
            metric: Metric to use ("win_rate", "sharpe", "avg_return")
            
        Returns:
            Tuple of (regime, value) or None
        """
        regime_perfs = self.get_factor_performance_by_regime(factor_name)
        
        if not regime_perfs:
            return None
        
        worst_regime = None
        worst_value = float("inf")
        
        for regime, perf in regime_perfs.items():
            value = getattr(perf, metric, 0.0)
            if value < worst_value:
                worst_value = value
                worst_regime = regime
        
        return (worst_regime, worst_value) if worst_regime else None
    
    def generate_report(self) -> str:
        """
        Generate human-readable report.
        
        Returns:
            Formatted report string
        """
        lines = []
        
        lines.append("REGIME STATISTICS REPORT")
        lines.append("=" * 50)
        
        for regime in sorted(self._stats.keys()):
            lines.append(f"\nRegime: {regime}")
            lines.append("-" * 40)
            
            # Overall performance
            overall = self._stats[regime].get("overall")
            if overall:
                lines.append(f"  Overall:")
                lines.append(f"    Trades: {overall.trades}")
                lines.append(f"    Win Rate: {overall.win_rate:.2%}")
                lines.append(f"    Sharpe: {overall.sharpe:.2f}")
                lines.append(f"    Avg Return: {overall.avg_return:.2%}")
            
            # Factor performance
            lines.append(f"  Factors:")
            for factor_name, perf in self._stats[regime].items():
                if factor_name != "overall":
                    lines.append(f"    {factor_name}:")
                    lines.append(f"      Trades: {perf.trades}")
                    lines.append(f"      Win Rate: {perf.win_rate:.2%}")
                    lines.append(f"      Sharpe: {perf.sharpe:.2f}")
                    lines.append(f"      IC: {perf.information_coefficient:.3f}")
        
        return "\n".join(lines)
    
    def clear_old_data(self, days: int = 90) -> int:
        """
        Clear old data from statistics.
        
        Args:
            days: Number of days to keep
            
        Returns:
            Number of regimes cleared
        """
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(days=days)
        
        to_remove = []
        
        for regime in self._stats.keys():
            overall = self._stats[regime].get("overall")
            if overall and overall.last_updated < cutoff:
                to_remove.append(regime)
        
        for regime in to_remove:
            del self._stats[regime]
            del self._returns[regime]
        
        self._logger.info(f"Cleared {len(to_remove)} old regimes")
        return len(to_remove)


# Global instance
_global_regime_stats = None


def get_regime_statistics() -> RegimeStatistics:
    """
    Get global regime statistics instance.
    
    Returns:
        RegimeStatistics instance
    """
    global _global_regime_stats
    if _global_regime_stats is None:
        _global_regime_stats = RegimeStatistics()
    return _global_regime_stats
