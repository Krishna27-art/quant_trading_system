"""
Interaction Runner

Runs batch interaction tests for multiple factor-condition combinations.
Orchestrates the testing process and aggregates results.
"""

from typing import List, Dict

import pandas as pd

from research.interactions.interaction_engine.interaction_builder import Interaction
from research.interactions.interaction_engine.interaction_engine import InteractionEngine, InteractionResult
from utils.logger import get_logger

logger = get_logger("research.interactions.interaction_engine")


class InteractionRunner:
    """
    Runs batch interaction tests for multiple factor-condition combinations.
    
    Orchestrates:
    - Multiple factor tests
    - Multiple condition tests
    - Result aggregation
    - Progress tracking
    """
    
    def __init__(self, engine: Optional[InteractionEngine] = None):
        """
        Initialize interaction runner.
        
        Args:
            engine: Optional InteractionEngine instance
        """
        self.engine = engine or InteractionEngine()
        self._logger = get_logger("research.interactions.interaction_engine")
    
    def run_batch(
        self,
        interactions: List[Interaction],
        factor_data: Dict[str, pd.Series],
        returns: pd.Series,
        market_contexts: pd.DataFrame,
    ) -> Dict[str, InteractionResult]:
        """
        Run batch interaction tests.
        
        Args:
            interactions: List of Interaction objects to test
            factor_data: Dictionary mapping factor names to factor values
            returns: Series of forward returns
            market_contexts: DataFrame with market context
            
        Returns:
            Dictionary mapping interaction IDs to results
        """
        results = {}
        
        for interaction in interactions:
            # Get factor data
            if interaction.factor_name not in factor_data:
                self._logger.warning(f"Factor data not found: {interaction.factor_name}")
                continue
            
            factor_values = factor_data[interaction.factor_name]
            
            # Test interaction
            result = self.engine.test_interaction(
                interaction.factor_name,
                factor_values,
                returns,
                interaction.condition,
                market_contexts,
            )
            
            # Store result
            interaction_id = interaction.get_id()
            results[interaction_id] = result
            
            self._logger.info(f"Tested interaction {interaction_id}: {result.decision}")
        
        return results
    
    def run_factor_conditions(
        self,
        factor_name: str,
        factor_values: pd.Series,
        conditions: List,
        returns: pd.Series,
        market_contexts: pd.DataFrame,
    ) -> Dict[str, InteractionResult]:
        """
        Run tests for one factor under multiple conditions.
        
        Args:
            factor_name: Name of the factor
            factor_values: Series of factor values
            conditions: List of Condition objects
            returns: Series of forward returns
            market_contexts: DataFrame with market context
            
        Returns:
            Dictionary mapping condition IDs to results
        """
        results = {}
        
        for condition in conditions:
            # Test interaction
            result = self.engine.test_interaction(
                factor_name,
                factor_values,
                returns,
                condition,
                market_contexts,
            )
            
            # Store result
            condition_id = f"{factor_name}_{condition.serialize()}"
            results[condition_id] = result
            
            self._logger.info(f"Tested {factor_name} under {condition.get_description()}: {result.decision}")
        
        return results
    
    def run_condition_factors(
        self,
        condition,
        factor_data: Dict[str, pd.Series],
        returns: pd.Series,
        market_contexts: pd.DataFrame,
    ) -> Dict[str, InteractionResult]:
        """
        Run tests for multiple factors under one condition.
        
        Args:
            condition: Condition to test
            factor_data: Dictionary mapping factor names to factor values
            returns: Series of forward returns
            market_contexts: DataFrame with market context
            
        Returns:
            Dictionary mapping factor names to results
        """
        results = {}
        
        for factor_name, factor_values in factor_data.items():
            # Test interaction
            result = self.engine.test_interaction(
                factor_name,
                factor_values,
                returns,
                condition,
                market_contexts,
            )
            
            # Store result
            results[factor_name] = result
            
            self._logger.info(f"Tested {factor_name} under condition: {result.decision}")
        
        return results
    
    def summarize_results(self, results: Dict[str, InteractionResult]) -> Dict:
        """
        Summarize interaction test results.
        
        Args:
            results: Dictionary of interaction results
            
        Returns:
            Summary statistics
        """
        total = len(results)
        passed = sum(1 for r in results.values() if r.decision == "PASS")
        failed = sum(1 for r in results.values() if r.decision == "FAIL")
        neutral = sum(1 for r in results.values() if r.decision == "NEUTRAL")
        
        # Calculate average metrics
        avg_ic = sum(r.ic for r in results.values()) / total if total > 0 else 0.0
        avg_sharpe = sum(r.sharpe for r in results.values()) / total if total > 0 else 0.0
        avg_win_rate = sum(r.win_rate for r in results.values()) / total if total > 0 else 0.0
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "neutral": neutral,
            "pass_rate": passed / total if total > 0 else 0.0,
            "avg_ic": avg_ic,
            "avg_sharpe": avg_sharpe,
            "avg_win_rate": avg_win_rate,
        }
    
    def get_top_performers(
        self,
        results: Dict[str, InteractionResult],
        metric: str = "ic",
        n: int = 10,
    ) -> List[tuple]:
        """
        Get top performing interactions.
        
        Args:
            results: Dictionary of interaction results
            metric: Metric to sort by (ic, sharpe, win_rate)
            n: Number of top performers to return
            
        Returns:
            List of (interaction_id, result) tuples
        """
        sorted_results = sorted(
            results.items(),
            key=lambda x: getattr(x[1], metric),
            reverse=True,
        )
        
        return sorted_results[:n]


def run_interaction_batch(
    interactions: List[Interaction],
    factor_data: Dict[str, pd.Series],
    returns: pd.Series,
    market_contexts: pd.DataFrame,
) -> Dict[str, InteractionResult]:
    """
    Convenience function to run batch interaction tests.
    
    Args:
        interactions: List of Interaction objects
        factor_data: Dictionary mapping factor names to factor values
        returns: Series of forward returns
        market_contexts: DataFrame with market context
        
    Returns:
        Dictionary mapping interaction IDs to results
    """
    runner = InteractionRunner()
    return runner.run_batch(interactions, factor_data, returns, market_contexts)
