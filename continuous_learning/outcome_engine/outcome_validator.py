"""
Outcome Validator

Validates trade outcomes and checks for data quality.
Ensures outcomes are well-formed and meaningful for learning.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional

from continuous_learning.outcome_engine.trade_outcome import TradeOutcome
from continuous_learning.outcome_engine.outcome_resolver import ResolvedOutcome
from utils.logger import get_logger

logger = get_logger("continuous_learning.outcome_engine")


@dataclass
class OutcomeValidationResult:
    """Result of outcome validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    quality_score: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "quality_score": round(self.quality_score, 4),
        }


class OutcomeValidator:
    """
    Validates trade outcomes and checks for data quality.
    
    Ensures:
    - All required fields are present
    - Field values are valid
    - No contradictory information
    - Outcome is meaningful for learning
    """
    
    def __init__(self):
        """Initialize outcome validator."""
        self._logger = get_logger("continuous_learning.outcome_engine")
    
    def validate(self, trade_outcome: TradeOutcome) -> OutcomeValidationResult:
        """
        Validate a trade outcome.
        
        Args:
            trade_outcome: TradeOutcome to validate
            
        Returns:
            OutcomeValidationResult
        """
        errors = []
        warnings = []
        
        # Use trade outcome's built-in validation
        is_valid, field_errors = trade_outcome.validate()
        errors.extend(field_errors)
        
        # Validate resolved outcome
        outcome_valid, outcome_errors = trade_outcome.resolved_outcome.validate()
        if not outcome_valid:
            errors.extend(outcome_errors)
        
        # Check for data quality issues
        quality_warnings = self._check_data_quality(trade_outcome)
        warnings.extend(quality_warnings)
        
        # Check for learning value
        learning_warnings = self._check_learning_value(trade_outcome)
        warnings.extend(learning_warnings)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(trade_outcome, errors, warnings)
        
        return OutcomeValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            quality_score=quality_score,
        )
    
    def _check_data_quality(self, trade_outcome: TradeOutcome) -> List[str]:
        """
        Check for data quality issues.
        
        Args:
            trade_outcome: TradeOutcome to check
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Check for missing target or stop loss
        if trade_outcome.target_price is None and trade_outcome.stop_loss is None:
            warnings.append("No target or stop loss specified - limited learning value")
        
        # Check for very short holding period
        if trade_outcome.resolved_outcome.holding_period_days is not None:
            if trade_outcome.resolved_outcome.holding_period_days < 1:
                warnings.append("Very short holding period - may be data error")
        
        # Check for extreme returns
        if abs(trade_outcome.resolved_outcome.return_percentage) > 0.5:
            warnings.append(f"Extreme return: {trade_outcome.resolved_outcome.return_percentage:.2%}")
        
        # Check for missing exit price
        if trade_outcome.resolved_outcome.exit_price is None:
            warnings.append("Missing exit price - outcome may be pending")
        
        return warnings
    
    def _check_learning_value(self, trade_outcome: TradeOutcome) -> List[str]:
        """
        Check for learning value issues.
        
        Args:
            trade_outcome: TradeOutcome to check
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Check if outcome is pending
        if trade_outcome.resolved_outcome.outcome_type == "pending":
            warnings.append("Outcome is pending - not yet suitable for learning")
        
        # Check if outcome is cancelled
        if trade_outcome.resolved_outcome.outcome_type == "cancelled":
            warnings.append("Outcome was cancelled - limited learning value")
        
        # Check for low confidence predictions
        if trade_outcome.predicted_confidence == "LOW":
            warnings.append("Low confidence prediction - may not be representative")
        
        # Check for HOLD actions
        if trade_outcome.action == "HOLD":
            warnings.append("HOLD action provides limited learning value")
        
        return warnings
    
    def _calculate_quality_score(
        self,
        trade_outcome: TradeOutcome,
        errors: List[str],
        warnings: List[str],
    ) -> float:
        """
        Calculate quality score for learning.
        
        Args:
            trade_outcome: TradeOutcome
            errors: List of errors
            warnings: List of warnings
            
        Returns:
            Quality score (0-1)
        """
        score = 1.0
        
        # Penalize errors heavily
        score -= len(errors) * 0.3
        
        # Penalize warnings lightly
        score -= len(warnings) * 0.1
        
        # Boost for complete outcomes
        if trade_outcome.resolved_outcome.outcome_type in ["target_hit", "stop_hit", "timeout"]:
            score += 0.2
        
        # Boost for high confidence
        if trade_outcome.predicted_confidence == "HIGH":
            score += 0.1
        
        # Boost for successful trades
        if trade_outcome.is_successful:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def validate_batch(self, trade_outcomes: List[TradeOutcome]) -> Dict:
        """
        Validate multiple trade outcomes.
        
        Args:
            trade_outcomes: List of TradeOutcome
            
        Returns:
            Summary of validation results
        """
        results = {
            "total": len(trade_outcomes),
            "valid": 0,
            "invalid": 0,
            "high_quality": 0,
            "medium_quality": 0,
            "low_quality": 0,
            "errors": [],
        }
        
        for i, trade_outcome in enumerate(trade_outcomes):
            validation = self.validate(trade_outcome)
            
            if validation.is_valid:
                results["valid"] += 1
            else:
                results["invalid"] += 1
                results["errors"].append({
                    "index": i,
                    "prediction_id": trade_outcome.prediction_id,
                    "symbol": trade_outcome.symbol,
                    "errors": validation.errors,
                })
            
            # Categorize by quality
            if validation.quality_score >= 0.7:
                results["high_quality"] += 1
            elif validation.quality_score >= 0.4:
                results["medium_quality"] += 1
            else:
                results["low_quality"] += 1
        
        return results
    
    def filter_for_learning(
        self,
        trade_outcomes: List[TradeOutcome],
        min_quality_score: float = 0.5,
    ) -> List[TradeOutcome]:
        """
        Filter outcomes suitable for learning.
        
        Args:
            trade_outcomes: List of TradeOutcome
            min_quality_score: Minimum quality score
            
        Returns:
            List of TradeOutcome suitable for learning
        """
        suitable = []
        
        for trade_outcome in trade_outcomes:
            validation = self.validate(trade_outcome)
            
            if validation.is_valid and validation.quality_score >= min_quality_score:
                suitable.append(trade_outcome)
        
        return suitable
    
    def check_outcome_distribution(self, trade_outcomes: List[TradeOutcome]) -> Dict:
        """
        Check distribution of outcome types.
        
        Args:
            trade_outcomes: List of TradeOutcome
            
        Returns:
            Dictionary with outcome distribution statistics
        """
        outcome_counts = {}
        success_by_action = {"BUY": {"success": 0, "total": 0}, "SELL": {"success": 0, "total": 0}}
        
        for trade_outcome in trade_outcomes:
            # Count outcome types
            outcome_type = trade_outcome.resolved_outcome.outcome_type
            outcome_counts[outcome_type] = outcome_counts.get(outcome_type, 0) + 1
            
            # Track success by action
            if trade_outcome.action in success_by_action:
                success_by_action[trade_outcome.action]["total"] += 1
                if trade_outcome.is_successful:
                    success_by_action[trade_outcome.action]["success"] += 1
        
        # Calculate success rates
        success_rates = {}
        for action, counts in success_by_action.items():
            if counts["total"] > 0:
                success_rates[action] = counts["success"] / counts["total"]
            else:
                success_rates[action] = 0.0
        
        return {
            "outcome_counts": outcome_counts,
            "success_rates": success_rates,
            "total_outcomes": len(trade_outcomes),
        }


def validate_trade_outcome(trade_outcome: TradeOutcome) -> OutcomeValidationResult:
    """
    Convenience function to validate trade outcome.
    
    Args:
        trade_outcome: TradeOutcome to validate
        
    Returns:
        OutcomeValidationResult
    """
    validator = OutcomeValidator()
    return validator.validate(trade_outcome)
