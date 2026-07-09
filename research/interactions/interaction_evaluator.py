"""
Interaction Evaluator

Evaluates whether factors improve under specific conditions.
Compares factor performance with and without conditions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from research.interactions.interaction_engine.interaction_engine import InteractionResult
from research.interactions.condition_engine.condition import Condition
from utils.logger import get_logger

logger = get_logger("research.interactions.interaction_evaluator")


@dataclass
class EvaluationResult:
    """Result of condition evaluation."""
    factor_name: str
    condition: Condition
    baseline_ic: float
    conditional_ic: float
    ic_improvement: float
    baseline_sharpe: float
    conditional_sharpe: float
    sharpe_improvement: float
    baseline_win_rate: float
    conditional_win_rate: float
    win_rate_improvement: float
    is_improvement: bool
    significance: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "factor_name": self.factor_name,
            "condition": self.condition.serialize(),
            "baseline_ic": round(self.baseline_ic, 4),
            "conditional_ic": round(self.conditional_ic, 4),
            "ic_improvement": round(self.ic_improvement, 4),
            "baseline_sharpe": round(self.baseline_sharpe, 4),
            "conditional_sharpe": round(self.conditional_sharpe, 4),
            "sharpe_improvement": round(self.sharpe_improvement, 4),
            "baseline_win_rate": round(self.baseline_win_rate, 4),
            "conditional_win_rate": round(self.conditional_win_rate, 4),
            "win_rate_improvement": round(self.win_rate_improvement, 4),
            "is_improvement": self.is_improvement,
            "significance": self.significance,
        }


class InteractionEvaluator:
    """
    Evaluates whether factors improve under specific conditions.
    
    Compares:
    - Baseline performance (without condition)
    - Conditional performance (with condition)
    - Improvement metrics
    - Statistical significance
    """
    
    def __init__(
        self,
        improvement_threshold: float = 0.02,
        significance_threshold: float = 0.05,
    ):
        """
        Initialize interaction evaluator.
        
        Args:
            improvement_threshold: Minimum improvement to consider significant
            significance_threshold: P-value threshold for significance
        """
        self.improvement_threshold = improvement_threshold
        self.significance_threshold = significance_threshold
        self._logger = get_logger("research.interactions.interaction_evaluator")
    
    def evaluate_condition(
        self,
        factor_name: str,
        factor_values: pd.Series,
        returns: pd.Series,
        condition: Condition,
        market_contexts: pd.DataFrame,
    ) -> EvaluationResult:
        """
        Evaluate if factor improves under specific condition.
        
        Args:
            factor_name: Name of the factor
            factor_values: Series of factor values
            returns: Series of forward returns
            condition: Condition to evaluate
            market_contexts: DataFrame with market context
            
        Returns:
            EvaluationResult
        """
        # Calculate baseline metrics (without condition)
        baseline_metrics = self._calculate_baseline_metrics(factor_values, returns)
        
        # Calculate conditional metrics (with condition)
        conditional_metrics = self._calculate_conditional_metrics(
            factor_values,
            returns,
            condition,
            market_contexts,
        )
        
        # Calculate improvements
        ic_improvement = conditional_metrics["ic"] - baseline_metrics["ic"]
        sharpe_improvement = conditional_metrics["sharpe"] - baseline_metrics["sharpe"]
        win_rate_improvement = conditional_metrics["win_rate"] - baseline_metrics["win_rate"]
        
        # Determine if improvement
        is_improvement = self._is_improvement(ic_improvement, sharpe_improvement, win_rate_improvement)
        
        # Determine significance
        significance = self._determine_significance(
            ic_improvement,
            conditional_metrics["num_trades"],
        )
        
        return EvaluationResult(
            factor_name=factor_name,
            condition=condition,
            baseline_ic=baseline_metrics["ic"],
            conditional_ic=conditional_metrics["ic"],
            ic_improvement=ic_improvement,
            baseline_sharpe=baseline_metrics["sharpe"],
            conditional_sharpe=conditional_metrics["sharpe"],
            sharpe_improvement=sharpe_improvement,
            baseline_win_rate=baseline_metrics["win_rate"],
            conditional_win_rate=conditional_metrics["win_rate"],
            win_rate_improvement=win_rate_improvement,
            is_improvement=is_improvement,
            significance=significance,
        )
    
    def _calculate_baseline_metrics(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
    ) -> Dict:
        """Calculate baseline metrics without condition."""
        ic = factor_values.corr(returns)
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0
        win_rate = (returns > 0).mean()
        
        return {
            "ic": ic if not pd.isna(ic) else 0.0,
            "sharpe": sharpe if not pd.isna(sharpe) else 0.0,
            "win_rate": win_rate if not pd.isna(win_rate) else 0.0,
            "num_trades": len(factor_values),
        }
    
    def _calculate_conditional_metrics(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        condition: Condition,
        market_contexts: pd.DataFrame,
    ) -> Dict:
        """Calculate conditional metrics with condition."""
        # Filter by condition
        mask = self._filter_by_condition(condition, market_contexts)
        
        filtered_factor = factor_values[mask]
        filtered_returns = returns[mask]
        
        if len(filtered_factor) < 10:
            return {
                "ic": 0.0,
                "sharpe": 0.0,
                "win_rate": 0.0,
                "num_trades": len(filtered_factor),
            }
        
        ic = filtered_factor.corr(filtered_returns)
        sharpe = filtered_returns.mean() / filtered_returns.std() * np.sqrt(252) if filtered_returns.std() > 0 else 0.0
        win_rate = (filtered_returns > 0).mean()
        
        return {
            "ic": ic if not pd.isna(ic) else 0.0,
            "sharpe": sharpe if not pd.isna(sharpe) else 0.0,
            "win_rate": win_rate if not pd.isna(win_rate) else 0.0,
            "num_trades": len(filtered_factor),
        }
    
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
    
    def _is_improvement(
        self,
        ic_improvement: float,
        sharpe_improvement: float,
        win_rate_improvement: float,
    ) -> bool:
        """Determine if condition represents improvement."""
        # Improvement if at least one metric improves significantly
        return (
            ic_improvement > self.improvement_threshold or
            sharpe_improvement > self.improvement_threshold or
            win_rate_improvement > self.improvement_threshold
        )
    
    def _determine_significance(
        self,
        ic_improvement: float,
        num_trades: int,
    ) -> str:
        """Determine significance level."""
        if num_trades < 50:
            return "INSUFFICIENT_DATA"
        elif abs(ic_improvement) < 0.01:
            return "NEGLIGIBLE"
        elif abs(ic_improvement) < 0.05:
            return "MODERATE"
        else:
            return "SIGNIFICANT"
    
    def evaluate_hierarchical(
        self,
        factor_name: str,
        factor_values: pd.Series,
        returns: pd.Series,
        conditions: List[Condition],
        market_contexts: pd.DataFrame,
    ) -> List[EvaluationResult]:
        """
        Evaluate factor under hierarchical conditions.
        
        Args:
            factor_name: Name of the factor
            factor_values: Series of factor values
            returns: Series of forward returns
            conditions: List of conditions to evaluate
            market_contexts: DataFrame with market context
            
        Returns:
            List of EvaluationResult
        """
        results = []
        
        for condition in conditions:
            result = self.evaluate_condition(
                factor_name,
                factor_values,
                returns,
                condition,
                market_contexts,
            )
            results.append(result)
        
        return results
    
    def find_best_conditions(
        self,
        evaluations: List[EvaluationResult],
        metric: str = "ic_improvement",
        n: int = 5,
    ) -> List[EvaluationResult]:
        """
        Find conditions that provide best improvement.
        
        Args:
            evaluations: List of EvaluationResult
            metric: Metric to sort by
            n: Number of top conditions to return
            
        Returns:
            List of top EvaluationResult
        """
        sorted_evals = sorted(
            evaluations,
            key=lambda x: getattr(x, metric),
            reverse=True,
        )
        
        return sorted_evals[:n]


def evaluate_factor_condition(
    factor_name: str,
    factor_values: pd.Series,
    returns: pd.Series,
    condition: Condition,
    market_contexts: pd.DataFrame,
) -> EvaluationResult:
    """
    Convenience function to evaluate factor under condition.
    
    Args:
        factor_name: Name of the factor
        factor_values: Series of factor values
        returns: Series of forward returns
        condition: Condition to evaluate
        market_contexts: DataFrame with market context
        
    Returns:
        EvaluationResult
    """
    evaluator = InteractionEvaluator()
    return evaluator.evaluate_condition(factor_name, factor_values, returns, condition, market_contexts)
