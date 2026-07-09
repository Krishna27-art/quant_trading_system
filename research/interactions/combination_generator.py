"""
Combination Generator

Automatically generates factor-condition combinations.
Creates all valid combinations for systematic testing.
"""

from typing import List, Dict, Optional
from itertools import product

from research.interactions.condition_engine.condition import Condition
from research.interactions.condition_engine.condition_builder import ConditionFactory
from research.interactions.interaction_engine.interaction_builder import Interaction, InteractionBuilder
from utils.logger import get_logger

logger = get_logger("research.interactions.combination_generator")


class CombinationGenerator:
    """
    Automatically generates factor-condition combinations.
    
    Generates:
    - All valid combinations
    - No duplicates
    - Structured objects
    - Extensible design
    """
    
    def __init__(self):
        """Initialize combination generator."""
        self._logger = get_logger("research.interactions.combination_generator")
    
    def generate_condition_combinations(
        self,
        trends: Optional[List[str]] = None,
        volatilities: Optional[List[str]] = None,
        sectors: Optional[List[str]] = None,
        liquidities: Optional[List[str]] = None,
        market_breadths: Optional[List[str]] = None,
        options_sentiments: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        holding_periods: Optional[List[int]] = None,
    ) -> List[Condition]:
        """
        Generate all condition combinations.
        
        Args:
            trends: List of trend values
            volatilities: List of volatility values
            sectors: List of sector values
            liquidities: List of liquidity values
            market_breadths: List of market breadth values
            options_sentiments: List of options sentiment values
            timeframes: List of timeframe values
            holding_periods: List of holding period values
            
        Returns:
            List of Condition objects
        """
        # Default values
        trends = trends or ["bull", "bear", "sideways"]
        volatilities = volatilities or ["low", "medium", "high"]
        liquidities = liquidities or ["low", "medium", "high"]
        market_breadths = market_breadths or ["weak", "neutral", "strong"]
        options_sentiments = options_sentiments or ["bearish", "neutral", "bullish"]
        
        # Generate all combinations
        combinations = []
        
        for trend in trends:
            for volatility in volatilities:
                for liquidity in liquidities:
                    for market_breadth in market_breadths:
                        for options_sentiment in options_sentiments:
                            # Base combination
                            base_combination = {
                                "trend": trend,
                                "volatility": volatility,
                                "liquidity": liquidity,
                                "market_breadth": market_breadth,
                                "options_sentiment": options_sentiment,
                            }
                            
                            # Add sector if provided
                            if sectors:
                                for sector in sectors:
                                    combo = base_combination.copy()
                                    combo["sector"] = sector
                                    combinations.append(combo)
                            else:
                                combinations.append(base_combination)
        
        # Convert to Condition objects
        conditions = []
        seen = set()
        
        for combo in combinations:
            # Create hash for deduplication
            combo_hash = hash(str(combo))
            
            if combo_hash not in seen:
                seen.add(combo_hash)
                condition = Condition.deserialize(combo)
                conditions.append(condition)
        
        self._logger.info(f"Generated {len(conditions)} unique condition combinations")
        return conditions
    
    def generate_interaction_combinations(
        self,
        factor_names: List[str],
        conditions: List[Condition],
    ) -> List[Interaction]:
        """
        Generate factor-condition interaction combinations.
        
        Args:
            factor_names: List of factor names
            conditions: List of Condition objects
            
        Returns:
            List of Interaction objects
        """
        interactions = []
        
        for factor_name in factor_names:
            for condition in conditions:
                interaction = Interaction(
                    factor_name=factor_name,
                    condition=condition,
                )
                interactions.append(interaction)
        
        self._logger.info(f"Generated {len(interactions)} interaction combinations")
        return interactions
    
    def generate_hierarchical_combinations(
        self,
        factor_names: List[str],
        max_depth: int = 3,
    ) -> List[Interaction]:
        """
        Generate hierarchical combinations (adding conditions incrementally).
        
        Args:
            factor_names: List of factor names
            max_depth: Maximum number of condition dimensions
            
        Returns:
            List of Interaction objects
        """
        interactions = []
        
        # Dimension order
        dimensions = [
            ("trend", ["bull", "bear", "sideways"]),
            ("volatility", ["low", "medium", "high"]),
            ("liquidity", ["low", "medium", "high"]),
            ("market_breadth", ["weak", "neutral", "strong"]),
        ]
        
        for factor_name in factor_names:
            # Generate combinations for each depth
            for depth in range(1, max_depth + 1):
                # Select first 'depth' dimensions
                selected_dimensions = dimensions[:depth]
                
                # Generate all combinations for this depth
                dimension_values = [values for _, values in selected_dimensions]
                dimension_names = [name for name, _ in selected_dimensions]
                
                for combo_values in product(*dimension_values):
                    condition_dict = dict(zip(dimension_names, combo_values))
                    condition = Condition.deserialize(condition_dict)
                    
                    interaction = Interaction(
                        factor_name=factor_name,
                        condition=condition,
                    )
                    interactions.append(interaction)
        
        self._logger.info(f"Generated {len(interactions)} hierarchical interaction combinations")
        return interactions
    
    def generate_sector_specific_combinations(
        self,
        factor_names: List[str],
        sectors: List[str],
        base_conditions: Optional[List[Condition]] = None,
    ) -> List[Interaction]:
        """
        Generate sector-specific interaction combinations.
        
        Args:
            factor_names: List of factor names
            sectors: List of sector names
            base_conditions: Optional base conditions to combine with sectors
            
        Returns:
            List of Interaction objects
        """
        interactions = []
        
        if base_conditions:
            # Combine base conditions with sectors
            for factor_name in factor_names:
                for base_condition in base_conditions:
                    for sector in sectors:
                        # Create new condition with sector
                        condition_dict = base_condition.serialize()
                        condition_dict["sector"] = sector
                        condition = Condition.deserialize(condition_dict)
                        
                        interaction = Interaction(
                            factor_name=factor_name,
                            condition=condition,
                        )
                        interactions.append(interaction)
        else:
            # Generate simple sector conditions
            for factor_name in factor_names:
                for sector in sectors:
                    condition = Condition(sector=sector)
                    interaction = Interaction(
                        factor_name=factor_name,
                        condition=condition,
                    )
                    interactions.append(interaction)
        
        self._logger.info(f"Generated {len(interactions)} sector-specific interaction combinations")
        return interactions
    
    def filter_by_complexity(
        self,
        interactions: List[Interaction],
        min_complexity: int = 1,
        max_complexity: int = 5,
    ) -> List[Interaction]:
        """
        Filter interactions by complexity (number of condition dimensions).
        
        Args:
            interactions: List of Interaction objects
            min_complexity: Minimum complexity
            max_complexity: Maximum complexity
            
        Returns:
            Filtered list of Interaction objects
        """
        filtered = []
        
        for interaction in interactions:
            complexity = sum(1 for v in interaction.condition.serialize().values() if v is not None)
            
            if min_complexity <= complexity <= max_complexity:
                filtered.append(interaction)
        
        self._logger.info(f"Filtered to {len(filtered)} interactions (complexity {min_complexity}-{max_complexity})")
        return filtered


def generate_all_combinations(
    factor_names: List[str],
    include_sectors: bool = True,
    sectors: Optional[List[str]] = None,
) -> List[Interaction]:
    """
    Convenience function to generate all combinations.
    
    Args:
        factor_names: List of factor names
        include_sectors: Whether to include sector combinations
        sectors: Optional list of sector names
        
    Returns:
        List of Interaction objects
    """
    generator = CombinationGenerator()
    
    # Generate conditions
    if include_sectors and sectors:
        conditions = generator.generate_condition_combinations(sectors=sectors)
    else:
        conditions = generator.generate_condition_combinations()
    
    # Generate interactions
    interactions = generator.generate_interaction_combinations(factor_names, conditions)
    
    return interactions
