"""
Interaction Validator

Validates interaction results and checks for statistical significance.
Ensures interaction results are reliable and meaningful.
"""

from dataclasses import dataclass
from typing import List, Dict

from research.interactions.interaction_engine.interaction_engine import InteractionResult
from utils.logger import get_logger

logger = get_logger("research.interactions.interaction_engine")


@dataclass
class InteractionValidationResult:
    """Result of interaction validation."""
    is_valid: bool
    is_significant: bool
    errors: List[str]
    warnings: List[str]
    confidence_level: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "is_valid": self.is_valid,
            "is_significant": self.is_significant,
            "errors": self.errors,
            "warnings": self.warnings,
            "confidence_level": self.confidence_level,
        }


class InteractionValidator:
    """
    Validates interaction results and checks for statistical significance.
    
    Ensures:
    - Sufficient sample size
    - Statistical significance
    - Reasonable metrics
    - No data issues
    """
    
    def __init__(
        self,
        min_trades: int = 100,
        min_ic: float = 0.03,
        min_sharpe: float = 0.5,
        ic_significance_threshold: float = 0.05,
    ):
        """
        Initialize interaction validator.
        
        Args:
            min_trades: Minimum number of trades
            min_ic: Minimum IC for validity
            min_sharpe: Minimum Sharpe for validity
            ic_significance_threshold: P-value threshold for IC significance
        """
        self.min_trades = min_trades
        self.min_ic = min_ic
        self.min_sharpe = min_sharpe
        self.ic_significance_threshold = ic_significance_threshold
        self._logger = get_logger("research.interactions.interaction_engine")
    
    def validate(self, result: InteractionResult) -> InteractionValidationResult:
        """
        Validate an interaction result.
        
        Args:
            result: InteractionResult to validate
            
        Returns:
            InteractionValidationResult
        """
        errors = []
        warnings = []
        
        # Check sample size
        if result.num_trades < self.min_trades:
            errors.append(f"Insufficient trades: {result.num_trades} < {self.min_trades}")
        
        # Check IC
        if abs(result.ic) < self.min_ic:
            warnings.append(f"Low IC: {result.ic:.4f} < {self.min_ic}")
        
        # Check Sharpe
        if result.sharpe < self.min_sharpe:
            warnings.append(f"Low Sharpe: {result.sharpe:.4f} < {self.min_sharpe}")
        
        # Check for extreme values
        if result.ic > 1.0 or result.ic < -1.0:
            errors.append(f"Invalid IC: {result.ic:.4f}")
        
        if result.sharpe > 10.0:
            warnings.append(f"Suspiciously high Sharpe: {result.sharpe:.4f}")
        
        if result.win_rate > 0.95:
            warnings.append(f"Suspiciously high win rate: {result.win_rate:.4f}")
        
        # Check max drawdown
        if result.max_drawdown > -0.5:
            warnings.append(f"High drawdown: {result.max_drawdown:.4f}")
        
        # Determine significance (simplified)
        is_significant = self._check_significance(result)
        
        # Determine confidence level
        confidence_level = self._determine_confidence_level(result)
        
        is_valid = len(errors) == 0
        
        return InteractionValidationResult(
            is_valid=is_valid,
            is_significant=is_significant,
            errors=errors,
            warnings=warnings,
            confidence_level=confidence_level,
        )
    
    def _check_significance(self, result: InteractionResult) -> bool:
        """
        Check if result is statistically significant.
        
        Args:
            result: InteractionResult
            
        Returns:
            True if significant
        """
        # Simplified significance check
        # In practice, would use proper statistical tests
        
        # IC significance based on sample size
        if result.num_trades < 100:
            return False
        
        # IC should be reasonably high
        if abs(result.ic) < 0.05:
            return False
        
        # Sharpe should be positive
        if result.sharpe <= 0:
            return False
        
        return True
    
    def _determine_confidence_level(self, result: InteractionResult) -> str:
        """
        Determine confidence level based on metrics.
        
        Args:
            result: InteractionResult
            
        Returns:
            Confidence level: "HIGH", "MEDIUM", "LOW"
        """
        score = 0
        
        # Sample size score
        if result.num_trades >= 500:
            score += 3
        elif result.num_trades >= 200:
            score += 2
        elif result.num_trades >= 100:
            score += 1
        
        # IC score
        if abs(result.ic) >= 0.1:
            score += 3
        elif abs(result.ic) >= 0.05:
            score += 2
        elif abs(result.ic) >= 0.03:
            score += 1
        
        # Sharpe score
        if result.sharpe >= 2.0:
            score += 3
        elif result.sharpe >= 1.0:
            score += 2
        elif result.sharpe >= 0.5:
            score += 1
        
        # Win rate score
        if result.win_rate >= 0.6:
            score += 2
        elif result.win_rate >= 0.55:
            score += 1
        
        # Determine level
        if score >= 8:
            return "HIGH"
        elif score >= 5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def validate_batch(self, results: Dict[str, InteractionResult]) -> Dict:
        """
        Validate multiple interaction results.
        
        Args:
            results: Dictionary of interaction results
            
        Returns:
            Summary of validation results
        """
        summary = {
            "total": len(results),
            "valid": 0,
            "invalid": 0,
            "significant": 0,
            "not_significant": 0,
            "high_confidence": 0,
            "medium_confidence": 0,
            "low_confidence": 0,
            "errors": [],
        }
        
        for interaction_id, result in results.items():
            validation = self.validate(result)
            
            if validation.is_valid:
                summary["valid"] += 1
            else:
                summary["invalid"] += 1
                summary["errors"].append({
                    "interaction_id": interaction_id,
                    "errors": validation.errors,
                })
            
            if validation.is_significant:
                summary["significant"] += 1
            else:
                summary["not_significant"] += 1
            
            if validation.confidence_level == "HIGH":
                summary["high_confidence"] += 1
            elif validation.confidence_level == "MEDIUM":
                summary["medium_confidence"] += 1
            else:
                summary["low_confidence"] += 1
        
        return summary


def validate_interaction_result(result: InteractionResult) -> InteractionValidationResult:
    """
    Convenience function to validate an interaction result.
    
    Args:
        result: InteractionResult to validate
        
    Returns:
        InteractionValidationResult
    """
    validator = InteractionValidator()
    return validator.validate(result)
