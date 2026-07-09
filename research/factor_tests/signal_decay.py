"""
Signal Decay Analyzer

Analyzes how long factor predictive power persists over time.
Critical for determining optimal holding periods and signal timing.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("research.signal_decay")


@dataclass
class DecayResult:
    """Result of signal decay analysis."""
    factor_name: str
    decay_horizons: List[int]
    decay_ics: List[float]
    decay_rank_ics: List[float]
    optimal_horizon: int
    half_life: int
    total_decay: float
    persistence_score: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "factor_name": self.factor_name,
            "decay_horizons": self.decay_horizons,
            "decay_ics": [round(ic, 4) for ic in self.decay_ics],
            "decay_rank_ics": [round(ic, 4) for ic in self.decay_rank_ics],
            "optimal_horizon": self.optimal_horizon,
            "half_life": self.half_life,
            "total_decay": round(self.total_decay, 4),
            "persistence_score": round(self.persistence_score, 4),
        }


class SignalDecayAnalyzer:
    """
    Analyzes signal decay over time.
    
    Measures how factor predictive power decays as the prediction horizon increases.
    This is critical for:
    - Determining optimal holding periods
    - Understanding signal timing
    - Avoiding over-holding degraded signals
    """
    
    def __init__(self, max_horizon: int = 20, horizon_step: int = 1):
        """
        Initialize signal decay analyzer.
        
        Args:
            max_horizon: Maximum horizon to test (in days/bars)
            horizon_step: Step size for horizon testing
        """
        self.max_horizon = max_horizon
        self.horizon_step = horizon_step
        self._logger = get_logger("research.signal_decay")
    
    def analyze_decay(
        self,
        factor_values: pd.Series,
        future_returns: pd.Series,
        factor_name: str = "unknown",
    ) -> DecayResult:
        """
        Analyze signal decay over multiple horizons.
        
        Args:
            factor_values: Series with factor values
            future_returns: Series with future returns
            factor_name: Name of factor for reporting
            
        Returns:
            DecayResult with decay analysis
        """
        from research.factor_tests.information_coefficient import calculate_ic
        
        decay_horizons = []
        decay_ics = []
        decay_rank_ics = []
        
        # Test IC at different horizons
        for horizon in range(self.horizon_step, self.max_horizon + 1, self.horizon_step):
            shifted_returns = future_returns.shift(-horizon)
            
            # Align with factor values (remove NaN from shift)
            aligned_factor = factor_values[:len(shifted_returns)]
            aligned_returns = shifted_returns[:len(factor_values)]
            
            # Remove NaN
            valid_mask = aligned_factor.notna() & aligned_returns.notna()
            if valid_mask.sum() < 10:
                continue
            
            try:
                ic_result = calculate_ic(
                    aligned_factor[valid_mask],
                    aligned_returns[valid_mask],
                )
                
                decay_horizons.append(horizon)
                decay_ics.append(ic_result.mean_ic)
                decay_rank_ics.append(ic_result.mean_rank_ic)
            except Exception as e:
                self._logger.warning(f"Failed to calculate IC at horizon {horizon}: {e}")
        
        if not decay_horizons:
            return DecayResult(
                factor_name=factor_name,
                decay_horizons=[],
                decay_ics=[],
                decay_rank_ics=[],
                optimal_horizon=0,
                half_life=0,
                total_decay=0.0,
                persistence_score=0.0,
            )
        
        # Find optimal horizon (maximum absolute IC)
        abs_ics = [abs(ic) for ic in decay_ics]
        optimal_idx = abs_ics.index(max(abs_ics))
        optimal_horizon = decay_horizons[optimal_idx]
        
        # Calculate half-life (horizon where IC drops to half of initial)
        initial_ic = abs(decay_ics[0]) if decay_ics else 0
        half_life = 0
        if initial_ic > 0:
            for i, ic in enumerate(decay_ics):
                if abs(ic) <= initial_ic / 2:
                    half_life = decay_horizons[i]
                    break
        
        # Calculate total decay (from first to last horizon)
        if len(decay_ics) >= 2:
            total_decay = abs(decay_ics[-1]) - abs(decay_ics[0])
        else:
            total_decay = 0.0
        
        # Calculate persistence score (area under decay curve)
        persistence_score = sum(abs(ic) for ic in decay_ics) / len(decay_ics) if decay_ics else 0.0
        
        return DecayResult(
            factor_name=factor_name,
            decay_horizons=decay_horizons,
            decay_ics=decay_ics,
            decay_rank_ics=decay_rank_ics,
            optimal_horizon=optimal_horizon,
            half_life=half_life,
            total_decay=total_decay,
            persistence_score=persistence_score,
        )
    
    def compare_decay(
        self,
        decay_results: List[DecayResult],
    ) -> pd.DataFrame:
        """
        Compare decay across multiple factors.
        
        Args:
            decay_results: List of DecayResult
            
        Returns:
            DataFrame with comparison
        """
        data = []
        
        for result in decay_results:
            data.append({
                "factor_name": result.factor_name,
                "optimal_horizon": result.optimal_horizon,
                "half_life": result.half_life,
                "total_decay": result.total_decay,
                "persistence_score": result.persistence_score,
                "max_ic": max([abs(ic) for ic in result.decay_ics]) if result.decay_ics else 0,
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values("persistence_score", ascending=False)
        
        return df
    
    def plot_decay_curve(
        self,
        decay_result: DecayResult,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot decay curve for visualization.
        
        Args:
            decay_result: DecayResult to plot
            save_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(decay_result.decay_horizons, decay_result.decay_ics, 'b-o', label='IC')
            plt.plot(decay_result.decay_horizons, decay_result.decay_rank_ics, 'r-s', label='Rank IC')
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            plt.axvline(x=decay_result.optimal_horizon, color='g', linestyle='--', alpha=0.5, label=f'Optimal: {decay_result.optimal_horizon}')
            plt.xlabel('Horizon (days/bars)')
            plt.ylabel('Information Coefficient')
            plt.title(f'Signal Decay: {decay_result.factor_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                self._logger.info(f"Saved decay plot to {save_path}")
            else:
                plt.show()
            
            plt.close()
        except ImportError:
            self._logger.warning("Matplotlib not available, skipping plot")
        except Exception as e:
            self._logger.error(f"Failed to plot decay curve: {e}")


def analyze_signal_decay(
    factor_values: pd.Series,
    future_returns: pd.Series,
    factor_name: str = "unknown",
    max_horizon: int = 20,
    horizon_step: int = 1,
) -> DecayResult:
    """
    Convenience function to analyze signal decay.
    
    Args:
        factor_values: Series with factor values
        future_returns: Series with future returns
        factor_name: Name of factor
        max_horizon: Maximum horizon to test
        horizon_step: Step size for horizon testing
        
    Returns:
        DecayResult
    """
    analyzer = SignalDecayAnalyzer(max_horizon=max_horizon, horizon_step=horizon_step)
    return analyzer.analyze_decay(factor_values, future_returns, factor_name=factor_name)
