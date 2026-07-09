"""
Interaction Engine

Tests factor performance under specific market conditions.
Calculates performance metrics for factor-condition combinations.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np

from research.interactions.condition_engine.condition import Condition
from research.performance_metrics.performance_metrics import PerformanceMetrics
from utils.logger import get_logger

logger = get_logger("research.interactions.interaction_engine")


@dataclass
class InteractionResult:
    """Result of factor-condition interaction testing."""
    factor_name: str
    condition: Condition
    ic: float
    rank_ic: float
    sharpe: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    num_trades: int
    decision: str  # "PASS", "FAIL", "NEUTRAL"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "factor_name": self.factor_name,
            "condition": self.condition.serialize(),
            "ic": round(self.ic, 4),
            "rank_ic": round(self.rank_ic, 4),
            "sharpe": round(self.sharpe, 4),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "num_trades": self.num_trades,
            "decision": self.decision,
        }


class InteractionEngine:
    """
    Tests factor performance under specific market conditions.
    
    Input:
    - One factor
    - One condition
    - Historical dataset
    
    Output:
    - Performance metrics for that combination
    """
    
    def __init__(
        self,
        min_trades: int = 100,
        min_ic: float = 0.05,
        min_sharpe: float = 1.0,
        min_win_rate: float = 0.55,
    ):
        """
        Initialize interaction engine.
        
        Args:
            min_trades: Minimum number of trades for valid test
            min_ic: Minimum IC for PASS
            min_sharpe: Minimum Sharpe for PASS
            min_win_rate: Minimum win rate for PASS
        """
        self.min_trades = min_trades
        self.min_ic = min_ic
        self.min_sharpe = min_sharpe
        self.min_win_rate = min_win_rate
        self._logger = get_logger("research.interactions.interaction_engine")
    
    def test_interaction(
        self,
        factor_name: str,
        factor_values: pd.Series,
        returns: pd.Series,
        condition: Condition,
        market_contexts: pd.DataFrame,
    ) -> InteractionResult:
        """
        Test factor performance under specific condition.
        
        Args:
            factor_name: Name of the factor
            factor_values: Series of factor values
            returns: Series of forward returns
            condition: Condition to test
            market_contexts: DataFrame with market context for each timestamp
            
        Returns:
            InteractionResult
        """
        # Filter data by condition
        mask = self._filter_by_condition(condition, market_contexts)
        
        # Apply mask to factor and returns
        filtered_factor = factor_values[mask]
        filtered_returns = returns[mask]
        
        # Check if enough data
        if len(filtered_factor) < self.min_trades:
            self._logger.warning(f"Insufficient data for interaction: {len(filtered_factor)} < {self.min_trades}")
            return InteractionResult(
                factor_name=factor_name,
                condition=condition,
                ic=0.0,
                rank_ic=0.0,
                sharpe=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                max_drawdown=0.0,
                num_trades=len(filtered_factor),
                decision="NEUTRAL",
            )
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(filtered_factor, filtered_returns)
        
        # Determine decision
        decision = self._make_decision(metrics)
        
        return InteractionResult(
            factor_name=factor_name,
            condition=condition,
            ic=metrics["ic"],
            rank_ic=metrics["rank_ic"],
            sharpe=metrics["sharpe"],
            win_rate=metrics["win_rate"],
            profit_factor=metrics["profit_factor"],
            max_drawdown=metrics["max_drawdown"],
            num_trades=len(filtered_factor),
            decision=decision,
        )
    
    def _filter_by_condition(
        self,
        condition: Condition,
        market_contexts: pd.DataFrame,
    ) -> pd.Series:
        """
        Filter data by condition.
        
        Args:
            condition: Condition to filter by
            market_contexts: DataFrame with market context
            
        Returns:
            Boolean mask
        """
        mask = pd.Series(True, index=market_contexts.index)
        
        # Filter by trend
        if condition.trend is not None:
            mask &= (market_contexts["trend"] == condition.trend)
        
        # Filter by volatility
        if condition.volatility is not None:
            mask &= (market_contexts["volatility"] == condition.volatility)
        
        # Filter by liquidity
        if condition.liquidity is not None:
            mask &= (market_contexts["liquidity"] == condition.liquidity)
        
        # Filter by market breadth
        if condition.market_breadth is not None:
            mask &= (market_contexts["market_breadth"] == condition.market_breadth)
        
        # Filter by options sentiment
        if condition.options_sentiment is not None:
            mask &= (market_contexts["options_sentiment"] == condition.options_sentiment)
        
        # Filter by sector (if available)
        if condition.sector is not None and "sector" in market_contexts.columns:
            mask &= (market_contexts["sector"] == condition.sector)
        
        return mask
    
    def _calculate_metrics(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
    ) -> dict:
        """
        Calculate performance metrics.
        
        Args:
            factor_values: Series of factor values
            returns: Series of forward returns
            
        Returns:
            Dictionary of metrics
        """
        # Calculate IC
        ic = factor_values.corr(returns)
        
        # Calculate Rank IC
        rank_ic = factor_values.rank().corr(returns.rank())
        
        # Calculate Sharpe
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0
        
        # Calculate win rate
        win_rate = (returns > 0).mean()
        
        # Calculate profit factor
        wins = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = wins / losses if losses > 0 else 0.0
        
        # Calculate max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            "ic": ic if not pd.isna(ic) else 0.0,
            "rank_ic": rank_ic if not pd.isna(rank_ic) else 0.0,
            "sharpe": sharpe if not pd.isna(sharpe) else 0.0,
            "win_rate": win_rate if not pd.isna(win_rate) else 0.0,
            "profit_factor": profit_factor if not pd.isna(profit_factor) else 0.0,
            "max_drawdown": max_drawdown if not pd.isna(max_drawdown) else 0.0,
        }
    
    def _make_decision(self, metrics: dict) -> str:
        """
        Make PASS/FAIL/NEUTRAL decision based on metrics.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Decision string
        """
        # Check if metrics meet thresholds
        ic_pass = metrics["ic"] >= self.min_ic
        sharpe_pass = metrics["sharpe"] >= self.min_sharpe
        win_rate_pass = metrics["win_rate"] >= self.min_win_rate
        
        if ic_pass and sharpe_pass and win_rate_pass:
            return "PASS"
        elif metrics["ic"] < 0 or metrics["sharpe"] < 0:
            return "FAIL"
        else:
            return "NEUTRAL"


def test_factor_interaction(
    factor_name: str,
    factor_values: pd.Series,
    returns: pd.Series,
    condition: Condition,
    market_contexts: pd.DataFrame,
) -> InteractionResult:
    """
    Convenience function to test factor interaction.
    
    Args:
        factor_name: Name of the factor
        factor_values: Series of factor values
        returns: Series of forward returns
        condition: Condition to test
        market_contexts: DataFrame with market context
        
    Returns:
        InteractionResult
    """
    engine = InteractionEngine()
    return engine.test_interaction(factor_name, factor_values, returns, condition, market_contexts)
