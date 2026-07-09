"""
Condition Validator

Validates condition objects and checks for logical consistency.
Ensures conditions are well-formed and meaningful.
"""

from dataclasses import dataclass
from typing import List, Optional

from research.interactions.condition_engine.condition import Condition
from research.interactions.market_context.market_state import MarketStateValidator
from utils.logger import get_logger

logger = get_logger("research.interactions.condition_engine")


@dataclass
class ConditionValidationResult:
    """Result of condition validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class ConditionValidator:
    """
    Validates condition objects and checks for logical consistency.
    
    Ensures:
    - All field values are valid
    - No contradictory conditions
    - Conditions are meaningful (not all wildcards)
    """
    
    def __init__(self):
        """Initialize condition validator."""
        self._logger = get_logger("research.interactions.condition_engine")
    
    def validate(self, condition: Condition) -> ConditionValidationResult:
        """
        Validate a condition.
        
        Args:
            condition: Condition to validate
            
        Returns:
            ConditionValidationResult
        """
        errors = []
        warnings = []
        
        # Validate individual fields
        is_valid, field_errors = condition.validate()
        errors.extend(field_errors)
        
        # Check for contradictions
        contradiction_errors = self._check_contradictions(condition)
        errors.extend(contradiction_errors)
        
        # Check if condition is meaningful
        if not condition.is_specific():
            warnings.append("Condition has no specific requirements (all wildcards)")
        
        # Check for unlikely combinations
        unlikely_warnings = self._check_unlikely_combinations(condition)
        warnings.extend(unlikely_warnings)
        
        return ConditionValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
    
    def _check_contradictions(self, condition: Condition) -> List[str]:
        """
        Check for contradictory conditions.
        
        Args:
            condition: Condition to check
            
        Returns:
            List of contradiction errors
        """
        errors = []
        
        # Bear market + bullish options is unlikely
        if condition.trend == "bear" and condition.options_sentiment == "bullish":
            errors.append("Contradictory: bear market with bullish options sentiment")
        
        # Bull market + bearish options is unlikely
        if condition.trend == "bull" and condition.options_sentiment == "bearish":
            errors.append("Contradictory: bull market with bearish options sentiment")
        
        # High volatility + strong breadth is unusual
        if condition.volatility == "high" and condition.market_breadth == "strong":
            errors.append("Unlikely: high volatility with strong market breadth")
        
        return errors
    
    def _check_unlikely_combinations(self, condition: Condition) -> List[str]:
        """
        Check for unlikely (but not impossible) combinations.
        
        Args:
            condition: Condition to check
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Very specific conditions may have few matches
        specific_count = sum([
            condition.trend is not None,
            condition.volatility is not None,
            condition.sector is not None,
            condition.liquidity is not None,
            condition.market_breadth is not None,
            condition.options_sentiment is not None,
        ])
        
        if specific_count >= 5:
            warnings.append("Very specific condition may have few historical matches")
        
        return warnings
    
    def validate_batch(self, conditions: List[Condition]) -> dict:
        """
        Validate multiple conditions.
        
        Args:
            conditions: List of Condition to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "total": len(conditions),
            "valid": 0,
            "invalid": 0,
            "errors": [],
        }
        
        for i, condition in enumerate(conditions):
            validation = self.validate(condition)
            
            if validation.is_valid:
                results["valid"] += 1
            else:
                results["invalid"] += 1
                results["errors"].append({
                    "index": i,
                    "description": condition.get_description(),
                    "errors": validation.errors,
                })
        
        return results
    
    def check_duplicates(self, conditions: List[Condition]) -> List[tuple]:
        """
        Check for duplicate conditions.
        
        Args:
            conditions: List of Condition
            
        Returns:
            List of tuples (index1, index2) for duplicates
        """
        duplicates = []
        
        for i in range(len(conditions)):
            for j in range(i + 1, len(conditions)):
                if conditions[i].serialize() == conditions[j].serialize():
                    duplicates.append((i, j))
        
        return duplicates
    
    def check_coverage(self, conditions: List[Condition]) -> dict:
        """
        Check how well conditions cover the market state space.
        
        Args:
            conditions: List of Condition
            
        Returns:
            Dictionary with coverage statistics
        """
        # Count unique values for each dimension
        trends = set()
        volatilities = set()
        sectors = set()
        
        for condition in conditions:
            if condition.trend:
                trends.add(condition.trend)
            if condition.volatility:
                volatilities.add(condition.volatility)
            if condition.sector:
                sectors.add(condition.sector)
        
        return {
            "unique_trends": len(trends),
            "unique_volatilities": len(volatilities),
            "unique_sectors": len(sectors),
            "total_conditions": len(conditions),
        }


def validate_condition(condition: Condition) -> ConditionValidationResult:
    """
    Convenience function to validate a condition.
    
    Args:
        condition: Condition to validate
        
    Returns:
        ConditionValidationResult
    """
    validator = ConditionValidator()
    return validator.validate(condition)
