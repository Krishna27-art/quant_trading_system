"""
Context Validator

Validates market context objects.
Ensures all required fields are present and valid.
"""

from dataclasses import dataclass
from typing import List, Optional

from research.interactions.market_context.market_context import MarketContext
from research.interactions.market_context.market_state import MarketStateValidator
from utils.logger import get_logger

logger = get_logger("research.interactions.market_context")


@dataclass
class ValidationResult:
    """Result of context validation."""
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


class ContextValidator:
    """
    Validates market context objects.
    
    Ensures:
    - All required fields are present
    - All field values are valid
    - Timestamp is valid
    - Optional fields are handled correctly
    """
    
    def __init__(self):
        """Initialize context validator."""
        self._logger = get_logger("research.interactions.market_context")
    
    def validate(self, context: MarketContext) -> ValidationResult:
        """
        Validate market context.
        
        Args:
            context: MarketContext to validate
            
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        
        # Validate timestamp
        if context.timestamp is None:
            errors.append("Timestamp is required")
        
        # Validate trend
        if not MarketStateValidator.validate_trend(context.trend):
            errors.append(f"Invalid trend: {context.trend}")
        
        # Validate volatility
        if not MarketStateValidator.validate_volatility(context.volatility):
            errors.append(f"Invalid volatility: {context.volatility}")
        
        # Validate liquidity
        if not MarketStateValidator.validate_liquidity(context.liquidity):
            errors.append(f"Invalid liquidity: {context.liquidity}")
        
        # Validate volume
        if not MarketStateValidator.validate_volume(context.volume):
            errors.append(f"Invalid volume: {context.volume}")
        
        # Validate sector strength
        if not MarketStateValidator.validate_sector_strength(context.sector_strength):
            errors.append(f"Invalid sector_strength: {context.sector_strength}")
        
        # Validate market breadth
        if not MarketStateValidator.validate_market_breadth(context.market_breadth):
            errors.append(f"Invalid market_breadth: {context.market_breadth}")
        
        # Validate options sentiment
        if not MarketStateValidator.validate_options_sentiment(context.options_sentiment):
            errors.append(f"Invalid options_sentiment: {context.options_sentiment}")
        
        # Check optional fields
        if context.vix_level is not None and context.vix_level < 0:
            warnings.append("VIX level should be non-negative")
        
        if context.nifty_level is not None and context.nifty_level < 0:
            errors.append("NIFTY level should be non-negative")
        
        if context.banknifty_level is not None and context.banknifty_level < 0:
            errors.append("BANKNIFTY level should be non-negative")
        
        if context.advance_decline_ratio is not None and (context.advance_decline_ratio < 0 or context.advance_decline_ratio > 1):
            warnings.append("Advance/decline ratio should be between 0 and 1")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
        )
    
    def validate_batch(self, contexts: List[MarketContext]) -> dict:
        """
        Validate multiple market contexts.
        
        Args:
            contexts: List of MarketContext to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "total": len(contexts),
            "valid": 0,
            "invalid": 0,
            "errors": [],
        }
        
        for i, context in enumerate(contexts):
            validation = self.validate(context)
            
            if validation.is_valid:
                results["valid"] += 1
            else:
                results["invalid"] += 1
                results["errors"].append({
                    "index": i,
                    "timestamp": context.timestamp.isoformat() if context.timestamp else None,
                    "errors": validation.errors,
                })
        
        return results
    
    def check_consistency(self, contexts: List[MarketContext]) -> List[str]:
        """
        Check consistency across multiple contexts.
        
        Args:
            contexts: List of MarketContext
            
        Returns:
            List of consistency warnings
        """
        warnings = []
        
        if len(contexts) < 2:
            return warnings
        
        # Check for rapid regime changes
        for i in range(1, len(contexts)):
            prev = contexts[i - 1]
            curr = contexts[i]
            
            # Trend changes
            if prev.trend != curr.trend:
                warnings.append(f"Trend changed from {prev.trend} to {curr.trend} at {curr.timestamp}")
            
            # Volatility spikes
            if prev.volatility == "low" and curr.volatility == "high":
                warnings.append(f"Volatility spike detected at {curr.timestamp}")
        
        return warnings


def validate_market_context(context: MarketContext) -> ValidationResult:
    """
    Convenience function to validate market context.
    
    Args:
        context: MarketContext to validate
        
    Returns:
        ValidationResult
    """
    validator = ContextValidator()
    return validator.validate(context)
