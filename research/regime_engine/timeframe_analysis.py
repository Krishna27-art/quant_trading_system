"""
Timeframe Analysis Engine

Analyzes factor performance across different timeframes.
Critical for understanding optimal holding periods and signal timing.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("research.timeframe_analysis")


@dataclass
class TimeframePerformance:
    """Performance metrics for a single timeframe."""
    timeframe: str
    mean_ic: float
    mean_rank_ic: float
    hit_rate: float
    sharpe: float
    sample_size: int
    optimal_holding_period: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "timeframe": self.timeframe,
            "mean_ic": round(self.mean_ic, 4),
            "mean_rank_ic": round(self.mean_rank_ic, 4),
            "hit_rate": round(self.hit_rate, 4),
            "sharpe": round(self.sharpe, 4),
            "sample_size": self.sample_size,
            "optimal_holding_period": self.optimal_holding_period,
        }


class TimeframeAnalyzer:
    """
    Analyzes factor performance across different timeframes.
    
    This is critical for understanding:
    - Which timeframes a factor works best in
    - Optimal holding periods
    - Signal timing
    - Multi-timeframe strategies
    """
    
    def __init__(self, timeframes: List[str] = None):
        """
        Initialize timeframe analyzer.
        
        Args:
            timeframes: List of timeframes to analyze (default: common timeframes)
        """
        self.timeframes = timeframes or ["1m", "5m", "15m", "30m", "1h", "1d", "1w"]
        self._logger = get_logger("research.timeframe_analysis")
    
    def analyze_factor_by_timeframe(
        self,
        factor_values: pd.Series,
        future_returns: pd.Series,
        timeframe_mapping: pd.Series,
    ) -> Dict[str, TimeframePerformance]:
        """
        Analyze factor performance by timeframe.
        
        Args:
            factor_values: Series with factor values
            future_returns: Series with future returns
            timeframe_mapping: Series mapping index to timeframe labels
            
        Returns:
            Dictionary mapping timeframe names to TimeframePerformance
        """
        from research.factor_tests.information_coefficient import calculate_ic
        from research.factor_tests.performance_metrics import calculate_performance_metrics
        from research.factor_tests.signal_decay import analyze_signal_decay
        
        # Align all series
        aligned_factor = factor_values.reindex(timeframe_mapping.index)
        aligned_returns = future_returns.reindex(timeframe_mapping.index)
        
        # Remove NaN
        valid_mask = (
            aligned_factor.notna() &
            aligned_returns.notna() &
            timeframe_mapping.notna()
        )
        aligned_factor = aligned_factor[valid_mask]
        aligned_returns = aligned_returns[valid_mask]
        aligned_timeframes = timeframe_mapping[valid_mask]
        
        # Analyze by timeframe
        timeframe_performance = {}
        
        for timeframe in aligned_timeframes.unique():
            timeframe_mask = aligned_timeframes == timeframe
            timeframe_factor = aligned_factor[timeframe_mask]
            timeframe_returns = aligned_returns[timeframe_mask]
            
            if len(timeframe_factor) < 30:
                self._logger.warning(
                    f"Insufficient data for timeframe {timeframe}: "
                    f"{len(timeframe_factor)} samples"
                )
                continue
            
            try:
                # Calculate IC
                ic_result = calculate_ic(timeframe_factor, timeframe_returns)
                
                # Calculate performance metrics
                perf_metrics = calculate_performance_metrics(timeframe_factor, timeframe_returns)
                
                # Calculate optimal holding period using signal decay
                decay_result = analyze_signal_decay(
                    timeframe_factor,
                    timeframe_returns,
                    max_horizon=20,
                    horizon_step=1,
                )
                optimal_holding = decay_result.optimal_horizon
                
                timeframe_performance[timeframe] = TimeframePerformance(
                    timeframe=timeframe,
                    mean_ic=ic_result.mean_ic,
                    mean_rank_ic=ic_result.mean_rank_ic,
                    hit_rate=ic_result.hit_rate,
                    sharpe=perf_metrics.sharpe_ratio,
                    sample_size=len(timeframe_factor),
                    optimal_holding_period=optimal_holding,
                )
                
            except Exception as e:
                self._logger.warning(f"Failed to analyze timeframe {timeframe}: {e}")
                continue
        
        return timeframe_performance
    
    def get_best_timeframe(
        self,
        timeframe_performance: Dict[str, TimeframePerformance],
        metric: str = "mean_ic",
    ) -> Optional[TimeframePerformance]:
        """
        Get best performing timeframe for a factor.
        
        Args:
            timeframe_performance: Dictionary of timeframe performance
            metric: Metric to rank by (mean_ic, sharpe, hit_rate, etc.)
            
        Returns:
            Best TimeframePerformance or None
        """
        timeframes = list(timeframe_performance.values())
        
        if not timeframes:
            return None
        
        # Sort by metric
        if metric == "mean_ic":
            timeframes.sort(key=lambda x: x.mean_ic, reverse=True)
        elif metric == "mean_rank_ic":
            timeframes.sort(key=lambda x: x.mean_rank_ic, reverse=True)
        elif metric == "hit_rate":
            timeframes.sort(key=lambda x: x.hit_rate, reverse=True)
        elif metric == "sharpe":
            timeframes.sort(key=lambda x: x.sharpe, reverse=True)
        else:
            timeframes.sort(key=lambda x: x.mean_ic, reverse=True)
        
        return timeframes[0]
    
    def compare_timeframes(
        self,
        timeframe_performance: Dict[str, TimeframePerformance],
    ) -> pd.DataFrame:
        """
        Compare timeframe performance in a DataFrame.
        
        Args:
            timeframe_performance: Dictionary of timeframe performance
            
        Returns:
            DataFrame with timeframe comparison
        """
        data = []
        
        for timeframe_name, perf in timeframe_performance.items():
            data.append({
                "timeframe": timeframe_name,
                "mean_ic": perf.mean_ic,
                "mean_rank_ic": perf.mean_rank_ic,
                "hit_rate": perf.hit_rate,
                "sharpe": perf.sharpe,
                "sample_size": perf.sample_size,
                "optimal_holding_period": perf.optimal_holding_period,
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values("mean_ic", ascending=False)
        
        return df
    
    def analyze_multi_timeframe_signal(
        self,
        timeframe_performance: Dict[str, TimeframePerformance],
    ) -> Dict[str, str]:
        """
        Analyze multi-timeframe signal strength.
        
        Args:
            timeframe_performance: Dictionary of timeframe performance
            
        Returns:
            Dictionary with multi-timeframe recommendations
        """
        recommendations = {}
        
        # Check if factor works across multiple timeframes
        strong_timeframes = [
            tf for tf, perf in timeframe_performance.items()
            if perf.mean_ic > 0.03 and perf.sharpe > 0.5
        ]
        
        if len(strong_timeframes) >= 3:
            recommendations["signal_strength"] = "STRONG"
            recommendations["strategy"] = "Use across multiple timeframes"
        elif len(strong_timeframes) >= 2:
            recommendations["signal_strength"] = "MODERATE"
            recommendations["strategy"] = "Use in best timeframes"
        elif len(strong_timeframes) == 1:
            recommendations["signal_strength"] = "WEAK"
            recommendations["strategy"] = "Use only in best timeframe"
        else:
            recommendations["signal_strength"] = "NONE"
            recommendations["strategy"] = "Factor not suitable for production"
        
        # Identify best timeframe
        best_tf = self.get_best_timeframe(timeframe_performance)
        if best_tf:
            recommendations["best_timeframe"] = best_tf.timeframe
            recommendations["optimal_holding"] = str(best_tf.optimal_holding_period)
        
        return recommendations
    
    def plot_timeframe_performance(
        self,
        timeframe_performance: Dict[str, TimeframePerformance],
        metric: str = "mean_ic",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot timeframe performance for visualization.
        
        Args:
            timeframe_performance: Dictionary of timeframe performance
            metric: Metric to plot
            save_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt
            
            timeframes = list(timeframe_performance.keys())
            
            if metric == "mean_ic":
                values = [perf.mean_ic for perf in timeframe_performance.values()]
            elif metric == "mean_rank_ic":
                values = [perf.mean_rank_ic for perf in timeframe_performance.values()]
            elif metric == "hit_rate":
                values = [perf.hit_rate for perf in timeframe_performance.values()]
            elif metric == "sharpe":
                values = [perf.sharpe for perf in timeframe_performance.values()]
            else:
                values = [perf.mean_ic for perf in timeframe_performance.values()]
            
            plt.figure(figsize=(10, 6))
            colors = ['green' if v > 0 else 'red' for v in values]
            plt.bar(timeframes, values, color=colors)
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            plt.xlabel('Timeframe')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.title(f'Factor Performance by Timeframe ({metric})')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                self._logger.info(f"Saved timeframe performance plot to {save_path}")
            else:
                plt.show()
            
            plt.close()
        except ImportError:
            self._logger.warning("Matplotlib not available, skipping plot")
        except Exception as e:
            self._logger.error(f"Failed to plot timeframe performance: {e}")


def analyze_factor_by_timeframe(
    factor_values: pd.Series,
    future_returns: pd.Series,
    timeframe_mapping: pd.Series,
    timeframes: List[str] = None,
) -> Dict[str, TimeframePerformance]:
    """
    Convenience function to analyze factor by timeframe.
    
    Args:
        factor_values: Series with factor values
        future_returns: Series with future returns
        timeframe_mapping: Series mapping index to timeframe labels
        timeframes: List of timeframes to analyze
        
    Returns:
        Dictionary mapping timeframe names to TimeframePerformance
    """
    analyzer = TimeframeAnalyzer(timeframes=timeframes)
    return analyzer.analyze_factor_by_timeframe(factor_values, future_returns, timeframe_mapping)
