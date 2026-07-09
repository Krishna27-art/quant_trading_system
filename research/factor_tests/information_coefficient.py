"""
Information Coefficient Calculator

Calculates Pearson IC and Spearman Rank IC for factor evaluation.
These are standard metrics for evaluating predictive factors in quantitative research.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from utils.logger import get_logger

logger = get_logger("research.information_coefficient")


@dataclass
class ICResult:
    """Result of Information Coefficient calculation."""
    mean_ic: float
    std_ic: float
    mean_rank_ic: float
    std_rank_ic: float
    rolling_ic: pd.Series
    rolling_rank_ic: pd.Series
    ic_t_stat: float
    ic_p_value: float
    rank_ic_t_stat: float
    rank_ic_p_value: float
    hit_rate: float
    sample_size: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "mean_ic": round(self.mean_ic, 4),
            "std_ic": round(self.std_ic, 4),
            "mean_rank_ic": round(self.mean_rank_ic, 4),
            "std_rank_ic": round(self.std_rank_ic, 4),
            "ic_t_stat": round(self.ic_t_stat, 4),
            "ic_p_value": round(self.ic_p_value, 4),
            "rank_ic_t_stat": round(self.rank_ic_t_stat, 4),
            "rank_ic_p_value": round(self.rank_ic_p_value, 4),
            "hit_rate": round(self.hit_rate, 4),
            "sample_size": self.sample_size,
        }


class InformationCoefficient:
    """
    Calculate Information Coefficient (IC) and Rank IC for factor evaluation.
    
    IC measures the correlation between factor values and future returns.
    Rank IC uses Spearman correlation which is more robust to outliers.
    
    These are standard metrics used by institutional quant researchers
    to evaluate factor predictive power.
    """
    
    def __init__(self, window: int = 20):
        """
        Initialize IC calculator.
        
        Args:
            window: Rolling window for rolling IC calculation
        """
        self.window = window
        self._logger = get_logger("research.information_coefficient")
    
    def calculate(
        self,
        factor_values: pd.Series,
        future_returns: pd.Series,
        min_periods: int = 10,
    ) -> ICResult:
        """
        Calculate IC metrics.
        
        Args:
            factor_values: Series with factor values
            future_returns: Series with future returns (aligned with factor_values)
            min_periods: Minimum periods for rolling calculation
            
        Returns:
            ICResult with all IC metrics
            
        Raises:
            ValueError: If timestamps don't match or insufficient data
        """
        # Validate inputs
        if len(factor_values) != len(future_returns):
            raise ValueError(
                f"Length mismatch: factor_values ({len(factor_values)}) vs "
                f"future_returns ({len(future_returns)})"
            )
        
        if not factor_values.index.equals(future_returns.index):
            raise ValueError("Timestamps do not match between factor_values and future_returns")
        
        if len(factor_values) < min_periods:
            raise ValueError(
                f"Insufficient data: {len(factor_values)} samples, need at least {min_periods}"
            )
        
        # Remove NaN values
        valid_mask = factor_values.notna() & future_returns.notna()
        factor_clean = factor_values[valid_mask]
        returns_clean = future_returns[valid_mask]
        
        if len(factor_clean) < min_periods:
            raise ValueError(
                f"Insufficient valid data after NaN removal: {len(factor_clean)} samples"
            )
        
        # Calculate Pearson IC
        ic, ic_p_value = pearsonr(factor_clean, returns_clean)
        
        # Calculate Spearman Rank IC
        rank_ic, rank_ic_p_value = spearmanr(factor_clean, returns_clean)
        
        # Calculate rolling IC
        rolling_ic = self._calculate_rolling_ic(factor_clean, returns_clean, min_periods)
        rolling_rank_ic = self._calculate_rolling_rank_ic(
            factor_clean, returns_clean, min_periods
        )
        
        # Calculate t-statistics
        n = len(factor_clean)
        ic_std = rolling_ic.std() if len(rolling_ic) > 0 else 0
        rank_ic_std = rolling_rank_ic.std() if len(rolling_rank_ic) > 0 else 0
        
        ic_t_stat = ic * np.sqrt(n) / ic_std if ic_std > 0 else 0
        rank_ic_t_stat = rank_ic * np.sqrt(n) / rank_ic_std if rank_ic_std > 0 else 0
        
        # Calculate hit rate (percentage of times IC > 0)
        hit_rate = (rolling_ic > 0).mean() if len(rolling_ic) > 0 else 0
        
        return ICResult(
            mean_ic=float(ic),
            std_ic=float(rolling_ic.std()) if len(rolling_ic) > 0 else 0.0,
            mean_rank_ic=float(rank_ic),
            std_rank_ic=float(rolling_rank_ic.std()) if len(rolling_rank_ic) > 0 else 0.0,
            rolling_ic=rolling_ic,
            rolling_rank_ic=rolling_rank_ic,
            ic_t_stat=float(ic_t_stat),
            ic_p_value=float(ic_p_value),
            rank_ic_t_stat=float(rank_ic_t_stat),
            rank_ic_p_value=float(rank_ic_p_value),
            hit_rate=float(hit_rate),
            sample_size=n,
        )
    
    def _calculate_rolling_ic(
        self,
        factor_values: pd.Series,
        future_returns: pd.Series,
        min_periods: int,
    ) -> pd.Series:
        """Calculate rolling Pearson IC."""
        rolling_ic = pd.Series(index=factor_values.index, dtype=float)
        
        for i in range(len(factor_values)):
            start_idx = max(0, i - self.window + 1)
            window_factor = factor_values.iloc[start_idx : i + 1]
            window_returns = future_returns.iloc[start_idx : i + 1]
            
            if len(window_factor) >= min_periods:
                try:
                    ic, _ = pearsonr(window_factor, window_returns)
                    rolling_ic.iloc[i] = ic
                except Exception:
                    rolling_ic.iloc[i] = np.nan
        
        return rolling_ic
    
    def _calculate_rolling_rank_ic(
        self,
        factor_values: pd.Series,
        future_returns: pd.Series,
        min_periods: int,
    ) -> pd.Series:
        """Calculate rolling Spearman Rank IC."""
        rolling_rank_ic = pd.Series(index=factor_values.index, dtype=float)
        
        for i in range(len(factor_values)):
            start_idx = max(0, i - self.window + 1)
            window_factor = factor_values.iloc[start_idx : i + 1]
            window_returns = future_returns.iloc[start_idx : i + 1]
            
            if len(window_factor) >= min_periods:
                try:
                    rank_ic, _ = spearmanr(window_factor, window_returns)
                    rolling_rank_ic.iloc[i] = rank_ic
                except Exception:
                    rolling_rank_ic.iloc[i] = np.nan
        
        return rolling_rank_ic
    
    def calculate_ic_stability(
        self,
        factor_values: pd.Series,
        future_returns: pd.Series,
        n_splits: int = 5,
    ) -> dict:
        """
        Calculate IC stability across time splits.
        
        Args:
            factor_values: Series with factor values
            future_returns: Series with future returns
            n_splits: Number of time splits
            
        Returns:
            Dictionary with IC stability metrics
        """
        valid_mask = factor_values.notna() & future_returns.notna()
        factor_clean = factor_values[valid_mask]
        returns_clean = future_returns[valid_mask]
        
        n = len(factor_clean)
        split_size = n // n_splits
        
        ic_values = []
        rank_ic_values = []
        
        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = start_idx + split_size if i < n_splits - 1 else n
            
            split_factor = factor_clean.iloc[start_idx:end_idx]
            split_returns = returns_clean.iloc[start_idx:end_idx]
            
            if len(split_factor) >= 10:
                ic, _ = pearsonr(split_factor, split_returns)
                rank_ic, _ = spearmanr(split_factor, split_returns)
                
                ic_values.append(ic)
                rank_ic_values.append(rank_ic)
        
        if not ic_values:
            return {
                "ic_stability": 0.0,
                "rank_ic_stability": 0.0,
                "ic_mean": 0.0,
                "ic_std": 0.0,
                "rank_ic_mean": 0.0,
                "rank_ic_std": 0.0,
            }
        
        ic_array = np.array(ic_values)
        rank_ic_array = np.array(rank_ic_values)
        
        # Stability = 1 - (std / abs(mean)) to measure consistency
        ic_stability = 1 - (np.std(ic_array) / abs(np.mean(ic_array))) if np.mean(ic_array) != 0 else 0
        rank_ic_stability = 1 - (np.std(rank_ic_array) / abs(np.mean(rank_ic_array))) if np.mean(rank_ic_array) != 0 else 0
        
        return {
            "ic_stability": max(0.0, ic_stability),
            "rank_ic_stability": max(0.0, rank_ic_stability),
            "ic_mean": float(np.mean(ic_array)),
            "ic_std": float(np.std(ic_array)),
            "rank_ic_mean": float(np.mean(rank_ic_array)),
            "rank_ic_std": float(np.std(rank_ic_array)),
        }


def calculate_ic(
    factor_values: pd.Series,
    future_returns: pd.Series,
    window: int = 20,
    min_periods: int = 10,
) -> ICResult:
    """
    Convenience function to calculate IC metrics.
    
    Args:
        factor_values: Series with factor values
        future_returns: Series with future returns
        window: Rolling window for rolling IC
        min_periods: Minimum periods for calculation
        
    Returns:
        ICResult
    """
    calculator = InformationCoefficient(window=window)
    return calculator.calculate(factor_values, future_returns, min_periods=min_periods)
