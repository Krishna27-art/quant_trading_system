"""
Context-Aware Signal Engine

Integrates conditional alpha into the signal generation pipeline.
Adjusts signal probabilities based on historical factor performance under current market conditions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from research.interactions.market_context.market_context import MarketContext
from research.interactions.condition_engine.condition import Condition
from research.interactions.interaction_engine.interaction_engine import InteractionResult
from research.interactions.interaction_database.database import InteractionDatabase
from utils.logger import get_logger

logger = get_logger("research.interactions.context_aware_signal")


@dataclass
class ContextAwareSignal:
    """Signal adjusted for market context."""
    symbol: str
    original_probability: float
    adjusted_probability: float
    confidence_boost: float
    matching_interactions: List[dict]
    current_context: MarketContext
    adjustment_reason: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "original_probability": round(self.original_probability, 4),
            "adjusted_probability": round(self.adjusted_probability, 4),
            "confidence_boost": round(self.confidence_boost, 4),
            "matching_interactions": self.matching_interactions,
            "current_context": self.current_context.to_dict(),
            "adjustment_reason": self.adjustment_reason,
        }


class ContextAwareSignalEngine:
    """
    Integrates conditional alpha into signal generation.
    
    Process:
    1. Get current market context
    2. Search database for matching interactions
    3. Adjust signal probability based on historical performance
    4. Generate context-aware signal
    """
    
    def __init__(
        self,
        database: Optional[InteractionDatabase] = None,
        min_matches: int = 3,
        confidence_boost_factor: float = 0.2,
    ):
        """
        Initialize context-aware signal engine.
        
        Args:
            database: Optional InteractionDatabase instance
            min_matches: Minimum number of matching interactions required
            confidence_boost_factor: Factor for confidence boost calculation
        """
        self.database = database or InteractionDatabase()
        self.min_matches = min_matches
        self.confidence_boost_factor = confidence_boost_factor
        self._logger = get_logger("research.interactions.context_aware_signal")
    
    def generate_signal(
        self,
        symbol: str,
        factor_name: str,
        original_probability: float,
        current_context: MarketContext,
    ) -> ContextAwareSignal:
        """
        Generate context-aware signal.
        
        Args:
            symbol: Stock symbol
            factor_name: Name of the factor
            original_probability: Original signal probability
            current_context: Current market context
            
        Returns:
            ContextAwareSignal
        """
        # Build condition from current context
        condition = self._context_to_condition(current_context)
        
        # Search database for matching interactions
        matching_interactions = self._find_matching_interactions(factor_name, condition)
        
        # Calculate adjustment
        adjustment = self._calculate_adjustment(matching_interactions)
        
        # Apply adjustment
        adjusted_probability = self._apply_adjustment(
            original_probability,
            adjustment,
        )
        
        # Generate reason
        reason = self._generate_adjustment_reason(matching_interactions, adjustment)
        
        return ContextAwareSignal(
            symbol=symbol,
            original_probability=original_probability,
            adjusted_probability=adjusted_probability,
            confidence_boost=adjustment,
            matching_interactions=matching_interactions,
            current_context=current_context,
            adjustment_reason=reason,
        )
    
    def _context_to_condition(self, context: MarketContext) -> Condition:
        """
        Convert market context to condition.
        
        Args:
            context: MarketContext
            
        Returns:
            Condition
        """
        return Condition(
            trend=context.trend,
            volatility=context.volatility,
            liquidity=context.liquidity,
            market_breadth=context.market_breadth,
            options_sentiment=context.options_sentiment,
        )
    
    def _find_matching_interactions(
        self,
        factor_name: str,
        condition: Condition,
    ) -> List[dict]:
        """
        Find interactions matching the condition.
        
        Args:
            factor_name: Name of the factor
            condition: Condition to match
            
        Returns:
            List of matching interaction records
        """
        # Query database for factor
        factor_results = self.database.query_by_factor(factor_name)
        
        # Filter by condition
        matching = []
        for record in factor_results:
            if self._condition_matches(record.condition, condition):
                matching.append(record.to_dict())
        
        return matching
    
    def _condition_matches(self, db_condition: dict, target_condition: Condition) -> bool:
        """
        Check if database condition matches target condition.
        
        Args:
            db_condition: Condition from database
            target_condition: Target condition to match
            
        Returns:
            True if conditions match
        """
        target_dict = target_condition.serialize()
        
        # Check each non-None field
        for key, value in target_dict.items():
            if value is not None:
                if db_condition.get(key) != value:
                    return False
        
        return True
    
    def _calculate_adjustment(self, matching_interactions: List[dict]) -> float:
        """
        Calculate probability adjustment based on matching interactions.
        
        Args:
            matching_interactions: List of matching interaction records
            
        Returns:
            Adjustment value (-1 to 1)
        """
        if len(matching_interactions) < self.min_matches:
            return 0.0
        
        # Calculate average IC of matching interactions
        avg_ic = sum(r["ic"] for r in matching_interactions) / len(matching_interactions)
        
        # Calculate average Sharpe
        avg_sharpe = sum(r["sharpe"] for r in matching_interactions) / len(matching_interactions)
        
        # Calculate pass rate
        pass_rate = sum(1 for r in matching_interactions if r["decision"] == "PASS") / len(matching_interactions)
        
        # Calculate adjustment
        # Positive IC and high pass rate -> boost
        # Negative IC or low pass rate -> reduce
        adjustment = 0.0
        
        if avg_ic > 0.05 and pass_rate > 0.6:
            adjustment = min(avg_ic * self.confidence_boost_factor, 0.3)
        elif avg_ic < -0.05 or pass_rate < 0.4:
            adjustment = max(avg_ic * self.confidence_boost_factor, -0.3)
        
        return adjustment
    
    def _apply_adjustment(
        self,
        original_probability: float,
        adjustment: float,
    ) -> float:
        """
        Apply adjustment to original probability.
        
        Args:
            original_probability: Original probability
            adjustment: Adjustment value
            
        Returns:
            Adjusted probability
        """
        adjusted = original_probability + adjustment
        
        # Clamp to [0, 1]
        adjusted = max(0.0, min(1.0, adjusted))
        
        return adjusted
    
    def _generate_adjustment_reason(
        self,
        matching_interactions: List[dict],
        adjustment: float,
    ) -> str:
        """
        Generate human-readable reason for adjustment.
        
        Args:
            matching_interactions: List of matching interactions
            adjustment: Adjustment value
            
        Returns:
            Reason string
        """
        if len(matching_interactions) < self.min_matches:
            return f"Insufficient matching interactions ({len(matching_interactions)} < {self.min_matches})"
        
        avg_ic = sum(r["ic"] for r in matching_interactions) / len(matching_interactions)
        pass_rate = sum(1 for r in matching_interactions if r["decision"] == "PASS") / len(matching_interactions)
        
        if adjustment > 0:
            return f"Factor historically performs well in current context (Avg IC: {avg_ic:.3f}, Pass Rate: {pass_rate:.2%})"
        elif adjustment < 0:
            return f"Factor historically performs poorly in current context (Avg IC: {avg_ic:.3f}, Pass Rate: {pass_rate:.2%})"
        else:
            return "Factor performance neutral in current context"
    
    def batch_generate_signals(
        self,
        signals: List[dict],
        current_context: MarketContext,
    ) -> List[ContextAwareSignal]:
        """
        Generate context-aware signals for multiple signals.
        
        Args:
            signals: List of signal dictionaries with symbol, factor_name, probability
            current_context: Current market context
            
        Returns:
            List of ContextAwareSignal
        """
        context_aware_signals = []
        
        for signal in signals:
            context_signal = self.generate_signal(
                symbol=signal["symbol"],
                factor_name=signal["factor_name"],
                original_probability=signal["probability"],
                current_context=current_context,
            )
            context_aware_signals.append(context_signal)
        
        return context_aware_signals


def generate_context_aware_signal(
    symbol: str,
    factor_name: str,
    original_probability: float,
    current_context: MarketContext,
    database: Optional[InteractionDatabase] = None,
) -> ContextAwareSignal:
    """
    Convenience function to generate context-aware signal.
    
    Args:
        symbol: Stock symbol
        factor_name: Name of the factor
        original_probability: Original signal probability
        current_context: Current market context
        database: Optional InteractionDatabase instance
        
    Returns:
        ContextAwareSignal
    """
    engine = ContextAwareSignalEngine(database=database)
    return engine.generate_signal(symbol, factor_name, original_probability, current_context)
