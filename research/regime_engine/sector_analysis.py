"""
Sector Analysis Engine

Analyzes factor performance across different sectors.
Critical for understanding where factors work and where they fail.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("research.sector_analysis")


@dataclass
class SectorPerformance:
    """Performance metrics for a single sector."""
    sector_name: str
    mean_ic: float
    mean_rank_ic: float
    hit_rate: float
    sharpe: float
    sample_size: int
    significance: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "sector_name": self.sector_name,
            "mean_ic": round(self.mean_ic, 4),
            "mean_rank_ic": round(self.mean_rank_ic, 4),
            "hit_rate": round(self.hit_rate, 4),
            "sharpe": round(self.sharpe, 4),
            "sample_size": self.sample_size,
            "significance": round(self.significance, 4),
        }


class SectorAnalyzer:
    """
    Analyzes factor performance across different sectors.
    
    This is critical for understanding:
    - Which sectors a factor works best in
    - Which sectors a factor fails in
    - Sector rotation opportunities
    - Factor deployment strategy
    """
    
    def __init__(self, min_samples: int = 30):
        """
        Initialize sector analyzer.
        
        Args:
            min_samples: Minimum samples required for sector analysis
        """
        self.min_samples = min_samples
        self._logger = get_logger("research.sector_analysis")
    
    def analyze_factor_by_sector(
        self,
        factor_values: pd.Series,
        future_returns: pd.Series,
        sector_mapping: pd.Series,
    ) -> Dict[str, SectorPerformance]:
        """
        Analyze factor performance by sector.
        
        Args:
            factor_values: Series with factor values
            future_returns: Series with future returns
            sector_mapping: Series mapping index to sector names
            
        Returns:
            Dictionary mapping sector names to SectorPerformance
        """
        from research.factor_tests.information_coefficient import calculate_ic
        from research.factor_tests.performance_metrics import calculate_performance_metrics
        
        # Align all series
        aligned_factor = factor_values.reindex(sector_mapping.index)
        aligned_returns = future_returns.reindex(sector_mapping.index)
        
        # Remove NaN
        valid_mask = (
            aligned_factor.notna() &
            aligned_returns.notna() &
            sector_mapping.notna()
        )
        aligned_factor = aligned_factor[valid_mask]
        aligned_returns = aligned_returns[valid_mask]
        aligned_sectors = sector_mapping[valid_mask]
        
        # Analyze by sector
        sector_performance = {}
        
        for sector_name in aligned_sectors.unique():
            sector_mask = aligned_sectors == sector_name
            sector_factor = aligned_factor[sector_mask]
            sector_returns = aligned_returns[sector_mask]
            
            if len(sector_factor) < self.min_samples:
                self._logger.warning(
                    f"Insufficient data for sector {sector_name}: "
                    f"{len(sector_factor)} samples (need {self.min_samples})"
                )
                continue
            
            try:
                # Calculate IC
                ic_result = calculate_ic(sector_factor, sector_returns)
                
                # Calculate performance metrics
                perf_metrics = calculate_performance_metrics(sector_factor, sector_returns)
                
                # Calculate significance (t-stat)
                significance = ic_result.ic_t_stat
                
                sector_performance[sector_name] = SectorPerformance(
                    sector_name=sector_name,
                    mean_ic=ic_result.mean_ic,
                    mean_rank_ic=ic_result.mean_rank_ic,
                    hit_rate=ic_result.hit_rate,
                    sharpe=perf_metrics.sharpe_ratio,
                    sample_size=len(sector_factor),
                    significance=significance,
                )
                
            except Exception as e:
                self._logger.warning(f"Failed to analyze sector {sector_name}: {e}")
                continue
        
        return sector_performance
    
    def get_best_sectors(
        self,
        sector_performance: Dict[str, SectorPerformance],
        metric: str = "mean_ic",
        top_n: int = 5,
    ) -> List[SectorPerformance]:
        """
        Get best performing sectors for a factor.
        
        Args:
            sector_performance: Dictionary of sector performance
            metric: Metric to rank by (mean_ic, sharpe, hit_rate, etc.)
            top_n: Number of top sectors to return
            
        Returns:
            List of top SectorPerformance
        """
        sectors = list(sector_performance.values())
        
        # Sort by metric
        if metric == "mean_ic":
            sectors.sort(key=lambda x: x.mean_ic, reverse=True)
        elif metric == "mean_rank_ic":
            sectors.sort(key=lambda x: x.mean_rank_ic, reverse=True)
        elif metric == "hit_rate":
            sectors.sort(key=lambda x: x.hit_rate, reverse=True)
        elif metric == "sharpe":
            sectors.sort(key=lambda x: x.sharpe, reverse=True)
        elif metric == "significance":
            sectors.sort(key=lambda x: x.significance, reverse=True)
        else:
            sectors.sort(key=lambda x: x.mean_ic, reverse=True)
        
        return sectors[:top_n]
    
    def get_worst_sectors(
        self,
        sector_performance: Dict[str, SectorPerformance],
        metric: str = "mean_ic",
        bottom_n: int = 5,
    ) -> List[SectorPerformance]:
        """
        Get worst performing sectors for a factor.
        
        Args:
            sector_performance: Dictionary of sector performance
            metric: Metric to rank by (mean_ic, sharpe, hit_rate, etc.)
            bottom_n: Number of worst sectors to return
            
        Returns:
            List of worst SectorPerformance
        """
        sectors = list(sector_performance.values())
        
        # Sort by metric (ascending for worst)
        if metric == "mean_ic":
            sectors.sort(key=lambda x: x.mean_ic)
        elif metric == "mean_rank_ic":
            sectors.sort(key=lambda x: x.mean_rank_ic)
        elif metric == "hit_rate":
            sectors.sort(key=lambda x: x.hit_rate)
        elif metric == "sharpe":
            sectors.sort(key=lambda x: x.sharpe)
        elif metric == "significance":
            sectors.sort(key=lambda x: x.significance)
        else:
            sectors.sort(key=lambda x: x.mean_ic)
        
        return sectors[:bottom_n]
    
    def calculate_sector_rotation(
        self,
        sector_performance: Dict[str, SectorPerformance],
    ) -> Dict[str, str]:
        """
        Calculate sector rotation recommendations.
        
        Args:
            sector_performance: Dictionary of sector performance
            
        Returns:
            Dictionary mapping sector names to recommendations
        """
        recommendations = {}
        
        for sector_name, perf in sector_performance.items():
            if perf.mean_ic > 0.05 and perf.sharpe > 1.0:
                recommendations[sector_name] = "STRONG BUY"
            elif perf.mean_ic > 0.02 and perf.sharpe > 0.5:
                recommendations[sector_name] = "BUY"
            elif perf.mean_ic < -0.02 and perf.sharpe < -0.5:
                recommendations[sector_name] = "SELL"
            elif perf.mean_ic < -0.05 and perf.sharpe < -1.0:
                recommendations[sector_name] = "STRONG SELL"
            else:
                recommendations[sector_name] = "HOLD"
        
        return recommendations
    
    def compare_sectors(
        self,
        sector_performance: Dict[str, SectorPerformance],
    ) -> pd.DataFrame:
        """
        Compare sector performance in a DataFrame.
        
        Args:
            sector_performance: Dictionary of sector performance
            
        Returns:
            DataFrame with sector comparison
        """
        data = []
        
        for sector_name, perf in sector_performance.items():
            data.append({
                "sector": sector_name,
                "mean_ic": perf.mean_ic,
                "mean_rank_ic": perf.mean_rank_ic,
                "hit_rate": perf.hit_rate,
                "sharpe": perf.sharpe,
                "significance": perf.significance,
                "sample_size": perf.sample_size,
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values("mean_ic", ascending=False)
        
        return df
    
    def plot_sector_performance(
        self,
        sector_performance: Dict[str, SectorPerformance],
        metric: str = "mean_ic",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot sector performance for visualization.
        
        Args:
            sector_performance: Dictionary of sector performance
            metric: Metric to plot
            save_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt
            
            sectors = list(sector_performance.keys())
            
            if metric == "mean_ic":
                values = [perf.mean_ic for perf in sector_performance.values()]
            elif metric == "mean_rank_ic":
                values = [perf.mean_rank_ic for perf in sector_performance.values()]
            elif metric == "hit_rate":
                values = [perf.hit_rate for perf in sector_performance.values()]
            elif metric == "sharpe":
                values = [perf.sharpe for perf in sector_performance.values()]
            else:
                values = [perf.mean_ic for perf in sector_performance.values()]
            
            plt.figure(figsize=(12, 6))
            colors = ['green' if v > 0 else 'red' for v in values]
            plt.barh(sectors, values, color=colors)
            plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            plt.xlabel(metric.replace('_', ' ').title())
            plt.ylabel('Sector')
            plt.title(f'Factor Performance by Sector ({metric})')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                self._logger.info(f"Saved sector performance plot to {save_path}")
            else:
                plt.show()
            
            plt.close()
        except ImportError:
            self._logger.warning("Matplotlib not available, skipping plot")
        except Exception as e:
            self._logger.error(f"Failed to plot sector performance: {e}")


def analyze_factor_by_sector(
    factor_values: pd.Series,
    future_returns: pd.Series,
    sector_mapping: pd.Series,
    min_samples: int = 30,
) -> Dict[str, SectorPerformance]:
    """
    Convenience function to analyze factor by sector.
    
    Args:
        factor_values: Series with factor values
        future_returns: Series with future returns
        sector_mapping: Series mapping index to sector names
        min_samples: Minimum samples required
        
    Returns:
        Dictionary mapping sector names to SectorPerformance
    """
    analyzer = SectorAnalyzer(min_samples=min_samples)
    return analyzer.analyze_factor_by_sector(factor_values, future_returns, sector_mapping)
