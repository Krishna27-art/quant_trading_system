"""
Interaction Stability Analysis

Measures the stability of factor-condition interactions over time.
Distinguishes stable interactions from ones that only appeared due to chance or specific historical periods.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd
import numpy as np

from research.interactions.interaction_engine.interaction_engine import InteractionResult
from research.interactions.condition_engine.condition import Condition
from utils.logger import get_logger

logger = get_logger("research.interactions.interaction_stability")


@dataclass
class StabilityMetrics:
    """Stability metrics for an interaction."""
    rolling_ic_mean: float
    rolling_ic_std: float
    rolling_ic_min: float
    rolling_ic_max: float
    rolling_sharpe_mean: float
    rolling_sharpe_std: float
    rolling_win_rate_mean: float
    rolling_win_rate_std: float
    regime_consistency: float
    stability_score: float
    is_stable: bool
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "rolling_ic_mean": round(self.rolling_ic_mean, 4),
            "rolling_ic_std": round(self.rolling_ic_std, 4),
            "rolling_ic_min": round(self.rolling_ic_min, 4),
            "rolling_ic_max": round(self.rolling_ic_max, 4),
            "rolling_sharpe_mean": round(self.rolling_sharpe_mean, 4),
            "rolling_sharpe_std": round(self.rolling_sharpe_std, 4),
            "rolling_win_rate_mean": round(self.rolling_win_rate_mean, 4),
            "rolling_win_rate_std": round(self.rolling_win_rate_mean, 4),
            "regime_consistency": round(self.regime_consistency, 4),
            "stability_score": round(self.stability_score, 4),
            "is_stable": self.is_stable,
        }


class InteractionStabilityAnalyzer:
    """
    Measures the stability of factor-condition interactions over time.
    
    Analyzes:
    - Rolling Information Coefficient (IC)
    - Rolling Sharpe ratio
    - Rolling win rate
    - Rolling expectancy
    - Performance across different market regimes
    """
    
    def __init__(
        self,
        window_size: int = 252,  # 1 year of trading days
        min_observations: int = 100,
        stability_threshold: float = 0.6,
    ):
        """
        Initialize stability analyzer.
        
        Args:
            window_size: Rolling window size for calculations
            min_observations: Minimum observations for stability assessment
            stability_threshold: Minimum stability score to consider stable
        """
        self.window_size = window_size
        self.min_observations = min_observations
        self.stability_threshold = stability_threshold
        self._logger = get_logger("research.interactions.interaction_stability")
    
    def analyze_stability(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        condition: Condition,
        market_contexts: pd.DataFrame,
    ) -> StabilityMetrics:
        """
        Analyze stability of an interaction over time.
        
        Args:
            factor_values: Series of factor values
            returns: Series of forward returns
            condition: Condition to analyze
            market_contexts: DataFrame with market context
            
        Returns:
            StabilityMetrics
        """
        # Filter by condition
        mask = self._filter_by_condition(condition, market_contexts)
        filtered_factor = factor_values[mask]
        filtered_returns = returns[mask]
        
        if len(filtered_factor) < self.min_observations:
            self._logger.warning(f"Insufficient data for stability analysis: {len(filtered_factor)}")
            return StabilityMetrics(
                rolling_ic_mean=0.0,
                rolling_ic_std=0.0,
                rolling_ic_min=0.0,
                rolling_ic_max=0.0,
                rolling_sharpe_mean=0.0,
                rolling_sharpe_std=0.0,
                rolling_win_rate_mean=0.0,
                rolling_win_rate_std=0.0,
                regime_consistency=0.0,
                stability_score=0.0,
                is_stable=False,
            )
        
        # Calculate rolling metrics
        rolling_ic = self._calculate_rolling_ic(filtered_factor, filtered_returns)
        rolling_sharpe = self._calculate_rolling_sharpe(filtered_returns)
        rolling_win_rate = self._calculate_rolling_win_rate(filtered_returns)
        
        # Calculate statistics
        ic_mean = rolling_ic.mean() if not rolling_ic.empty else 0.0
        ic_std = rolling_ic.std() if not rolling_ic.empty else 0.0
        ic_min = rolling_ic.min() if not rolling_ic.empty else 0.0
        ic_max = rolling_ic.max() if not rolling_ic.empty else 0.0
        
        sharpe_mean = rolling_sharpe.mean() if not rolling_sharpe.empty else 0.0
        sharpe_std = rolling_sharpe.std() if not rolling_sharpe.empty else 0.0
        
        win_rate_mean = rolling_win_rate.mean() if not rolling_win_rate.empty else 0.0
        win_rate_std = rolling_win_rate.std() if not rolling_win_rate.empty else 0.0
        
        # Calculate regime consistency
        regime_consistency = self._calculate_regime_consistency(
            filtered_factor,
            filtered_returns,
            market_contexts[mask],
        )
        
        # Calculate stability score
        stability_score = self._calculate_stability_score(
            ic_mean,
            ic_std,
            sharpe_mean,
            sharpe_std,
            win_rate_mean,
            win_rate_std,
            regime_consistency,
        )
        
        is_stable = stability_score >= self.stability_threshold
        
        return StabilityMetrics(
            rolling_ic_mean=ic_mean,
            rolling_ic_std=ic_std,
            rolling_ic_min=ic_min,
            rolling_ic_max=ic_max,
            rolling_sharpe_mean=sharpe_mean,
            rolling_sharpe_std=sharpe_std,
            rolling_win_rate_mean=win_rate_mean,
            rolling_win_rate_std=win_rate_std,
            regime_consistency=regime_consistency,
            stability_score=stability_score,
            is_stable=is_stable,
        )
    
    def _filter_by_condition(
        self,
        condition: Condition,
        market_contexts: pd.DataFrame,
    ) -> pd.Series:
        """Filter data by condition."""
        mask = pd.Series(True, index=market_contexts.index)
        
        if condition.trend is not None:
            mask &= (market_contexts["trend"] == condition.trend)
        
        if condition.volatility is not None:
            mask &= (market_contexts["volatility"] == condition.volatility)
        
        if condition.liquidity is not None:
            mask &= (market_contexts["liquidity"] == condition.liquidity)
        
        if condition.market_breadth is not None:
            mask &= (market_contexts["market_breadth"] == condition.market_breadth)
        
        if condition.options_sentiment is not None:
            mask &= (market_contexts["options_sentiment"] == condition.options_sentiment)
        
        if condition.sector is not None and "sector" in market_contexts.columns:
            mask &= (market_contexts["sector"] == condition.sector)
        
        return mask
    
    def _calculate_rolling_ic(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
    ) -> pd.Series:
        """Calculate rolling IC."""
        rolling_ic = pd.Series(index=factor_values.index, dtype=float)
        
        for i in range(self.window_size, len(factor_values)):
            window_factor = factor_values.iloc[i - self.window_size:i]
            window_returns = returns.iloc[i - self.window_size:i]
            ic = window_factor.corr(window_returns)
            rolling_ic.iloc[i] = ic if not pd.isna(ic) else 0.0
        
        return rolling_ic.dropna()
    
    def _calculate_rolling_sharpe(
        self,
        returns: pd.Series,
    ) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        rolling_sharpe = pd.Series(index=returns.index, dtype=float)
        
        for i in range(self.window_size, len(returns)):
            window_returns = returns.iloc[i - self.window_size:i]
            sharpe = window_returns.mean() / window_returns.std() * np.sqrt(252) if window_returns.std() > 0 else 0.0
            rolling_sharpe.iloc[i] = sharpe if not pd.isna(sharpe) else 0.0
        
        return rolling_sharpe.dropna()
    
    def _calculate_rolling_win_rate(
        self,
        returns: pd.Series,
    ) -> pd.Series:
        """Calculate rolling win rate."""
        rolling_win_rate = pd.Series(index=returns.index, dtype=float)
        
        for i in range(self.window_size, len(returns)):
            window_returns = returns.iloc[i - self.window_size:i]
            win_rate = (window_returns > 0).mean()
            rolling_win_rate.iloc[i] = win_rate if not pd.isna(win_rate) else 0.0
        
        return rolling_win_rate.dropna()
    
    def _calculate_regime_consistency(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        market_contexts: pd.DataFrame,
    ) -> float:
        """
        Calculate performance consistency across market regimes.
        
        Args:
            factor_values: Series of factor values
            returns: Series of forward returns
            market_contexts: DataFrame with market context
            
        Returns:
            Regime consistency score (0 to 1)
        """
        if len(market_contexts) == 0:
            return 0.0
        
        # Group by trend
        trend_groups = market_contexts.groupby("trend")
        
        regime_scores = []
        
        for trend, group in trend_groups:
            if len(group) < 10:
                continue
            
            # Get corresponding factor and returns
            group_factor = factor_values[group.index]
            group_returns = returns[group.index]
            
            # Calculate IC for this regime
            ic = group_factor.corr(group_returns)
            
            if not pd.isna(ic):
                regime_scores.append(abs(ic))
        
        if not regime_scores:
            return 0.0
        
        # Consistency is inverse of variance
        consistency = 1.0 - np.std(regime_scores) if regime_scores else 0.0
        return max(0.0, consistency)
    
    def _calculate_stability_score(
        self,
        ic_mean: float,
        ic_std: float,
        sharpe_mean: float,
        sharpe_std: float,
        win_rate_mean: float,
        win_rate_std: float,
        regime_consistency: float,
    ) -> float:
        """
        Calculate composite stability score.
        
        Args:
            ic_mean: Mean rolling IC
            ic_std: Std of rolling IC
            sharpe_mean: Mean rolling Sharpe
            sharpe_std: Std of rolling Sharpe
            win_rate_mean: Mean rolling win rate
            win_rate_std: Std of rolling win rate
            regime_consistency: Regime consistency score
            
        Returns:
            Stability score (0 to 1)
        """
        score = 0.0
        
        # IC stability (mean should be positive, std should be low)
        if ic_mean > 0:
            ic_score = min(ic_mean, 0.2) / 0.2  # Normalize to 0-1
            ic_stability = max(0, 1 - ic_std * 10)  # Penalize high std
            score += ic_score * ic_stability * 0.3
        
        # Sharpe stability
        if sharpe_mean > 0:
            sharpe_score = min(sharpe_mean, 2.0) / 2.0
            sharpe_stability = max(0, 1 - sharpe_std * 5)
            score += sharpe_score * sharpe_stability * 0.3
        
        # Win rate stability
        if win_rate_mean > 0.5:
            win_rate_score = (win_rate_mean - 0.5) / 0.5
            win_rate_stability = max(0, 1 - win_rate_std * 10)
            score += win_rate_score * win_rate_stability * 0.2
        
        # Regime consistency
        score += regime_consistency * 0.2
        
        return min(1.0, max(0.0, score))
    
    def batch_analyze_stability(
        self,
        interactions: List[tuple],
        factor_data: Dict[str, pd.Series],
        returns: pd.Series,
        market_contexts: pd.DataFrame,
    ) -> Dict[str, StabilityMetrics]:
        """
        Analyze stability for multiple interactions.
        
        Args:
            interactions: List of (factor_name, condition) tuples
            factor_data: Dictionary mapping factor names to factor values
            returns: Series of forward returns
            market_contexts: DataFrame with market context
            
        Returns:
            Dictionary mapping interaction IDs to StabilityMetrics
        """
        results = {}
        
        for factor_name, condition in interactions:
            if factor_name not in factor_data:
                self._logger.warning(f"Factor data not found: {factor_name}")
                continue
            
            factor_values = factor_data[factor_name]
            
            stability = self.analyze_stability(
                factor_values,
                returns,
                condition,
                market_contexts,
            )
            
            interaction_id = f"{factor_name}_{hash(str(condition.serialize()))}"
            results[interaction_id] = stability
        
        return results
    
    def filter_stable_interactions(
        self,
        stability_results: Dict[str, StabilityMetrics],
    ) -> List[str]:
        """
        Filter for stable interactions.
        
        Args:
            stability_results: Dictionary of stability results
            
        Returns:
            List of stable interaction IDs
        """
        stable = [
            interaction_id
            for interaction_id, metrics in stability_results.items()
            if metrics.is_stable
        ]
        
        return stable


def analyze_interaction_stability(
    factor_values: pd.Series,
    returns: pd.Series,
    condition: Condition,
    market_contexts: pd.DataFrame,
) -> StabilityMetrics:
    """
    Convenience function to analyze interaction stability.
    
    Args:
        factor_values: Series of factor values
        returns: Series of forward returns
        condition: Condition to analyze
        market_contexts: DataFrame with market context
        
    Returns:
        StabilityMetrics
    """
    analyzer = InteractionStabilityAnalyzer()
    return analyzer.analyze_stability(factor_values, returns, condition, market_contexts)
