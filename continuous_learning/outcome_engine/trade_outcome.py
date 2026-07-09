"""
Trade Outcome

Represents the complete outcome of a trade.
Combines prediction with resolved outcome for analysis.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime

from continuous_learning.outcome_engine.outcome_resolver import ResolvedOutcome
from utils.logger import get_logger

logger = get_logger("continuous_learning.outcome_engine")


@dataclass
class TradeOutcome:
    """Complete outcome of a trade."""
    prediction_id: str
    symbol: str
    action: str
    predicted_probability: float
    predicted_confidence: str
    expected_return: float
    entry_price: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    entry_timestamp: datetime
    resolved_outcome: ResolvedOutcome
    is_successful: bool
    lessons_learned: Optional[str]
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate trade outcome.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check action is valid
        valid_actions = ["BUY", "SELL", "HOLD"]
        if self.action not in valid_actions:
            errors.append(f"Invalid action: {self.action}")
        
        # Check probability is between 0 and 1
        if not (0.0 <= self.predicted_probability <= 1.0):
            errors.append(f"Probability must be between 0 and 1, got {self.predicted_probability}")
        
        # Check confidence level is valid
        valid_confidences = ["HIGH", "MEDIUM", "LOW"]
        if self.predicted_confidence not in valid_confidences:
            errors.append(f"Invalid confidence level: {self.predicted_confidence}")
        
        # Validate resolved outcome
        outcome_valid, outcome_errors = self.resolved_outcome.validate()
        if not outcome_valid:
            errors.extend(outcome_errors)
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "prediction_id": self.prediction_id,
            "symbol": self.symbol,
            "action": self.action,
            "predicted_probability": round(self.predicted_probability, 4),
            "predicted_confidence": self.predicted_confidence,
            "expected_return": round(self.expected_return, 4),
            "entry_price": round(self.entry_price, 2),
            "target_price": round(self.target_price, 2) if self.target_price else None,
            "stop_loss": round(self.stop_loss, 2) if self.stop_loss else None,
            "entry_timestamp": self.entry_timestamp.isoformat(),
            "resolved_outcome": self.resolved_outcome.to_dict(),
            "is_successful": self.is_successful,
            "lessons_learned": self.lessons_learned,
        }
    
    def calculate_prediction_accuracy(self) -> float:
        """
        Calculate prediction accuracy.
        
        Returns:
            Accuracy score (0-1)
        """
        if self.action == "HOLD":
            return 1.0  # HOLD is always "correct"
        
        if self.resolved_outcome.outcome_type == "target_hit":
            return 1.0
        elif self.resolved_outcome.outcome_type == "stop_hit":
            return 0.0
        elif self.resolved_outcome.outcome_type == "timeout":
            # For timeout, check if return was positive
            return 1.0 if self.resolved_outcome.return_percentage > 0 else 0.0
        else:
            return 0.5  # Neutral for pending/cancelled
    
    def get_risk_reward_ratio(self) -> Optional[float]:
        """
        Calculate risk/reward ratio.
        
        Returns:
            Risk/reward ratio or None
        """
        if self.target_price and self.stop_loss and self.entry_price:
            if self.action == "BUY":
                reward = abs(self.target_price - self.entry_price)
                risk = abs(self.entry_price - self.stop_loss)
            else:  # SELL
                reward = abs(self.entry_price - self.target_price)
                risk = abs(self.stop_loss - self.entry_price)
            
            return reward / risk if risk > 0 else None
        return None
    
    def get_holding_efficiency(self) -> Optional[float]:
        """
        Calculate holding efficiency (actual return / max possible return).
        
        Returns:
            Holding efficiency or None
        """
        if self.resolved_outcome.max_favorable_excursion > 0:
            return self.resolved_outcome.return_percentage / self.resolved_outcome.max_favorable_excursion
        return None


class TradeOutcomeBuilder:
    """Builder for creating TradeOutcome objects."""
    
    def __init__(self):
        """Initialize trade outcome builder."""
        self._prediction_id: Optional[str] = None
        self._symbol: Optional[str] = None
        self._action: Optional[str] = None
        self._predicted_probability: Optional[float] = None
        self._predicted_confidence: Optional[str] = None
        self._expected_return: Optional[float] = None
        self._entry_price: Optional[float] = None
        self._target_price: Optional[float] = None
        self._stop_loss: Optional[float] = None
        self._entry_timestamp: Optional[datetime] = None
        self._resolved_outcome: Optional[ResolvedOutcome] = None
        self._lessons_learned: Optional[str] = None
        self._logger = get_logger("continuous_learning.outcome_engine")
    
    def prediction_id(self, value: str) -> "TradeOutcomeBuilder":
        """Set prediction ID."""
        self._prediction_id = value
        return self
    
    def symbol(self, value: str) -> "TradeOutcomeBuilder":
        """Set symbol."""
        self._symbol = value
        return self
    
    def action(self, value: str) -> "TradeOutcomeBuilder":
        """Set action."""
        self._action = value
        return self
    
    def predicted_probability(self, value: float) -> "TradeOutcomeBuilder":
        """Set predicted probability."""
        self._predicted_probability = value
        return self
    
    def predicted_confidence(self, value: str) -> "TradeOutcomeBuilder":
        """Set predicted confidence."""
        self._predicted_confidence = value
        return self
    
    def expected_return(self, value: float) -> "TradeOutcomeBuilder":
        """Set expected return."""
        self._expected_return = value
        return self
    
    def entry_price(self, value: float) -> "TradeOutcomeBuilder":
        """Set entry price."""
        self._entry_price = value
        return self
    
    def target_price(self, value: float) -> "TradeOutcomeBuilder":
        """Set target price."""
        self._target_price = value
        return self
    
    def stop_loss(self, value: float) -> "TradeOutcomeBuilder":
        """Set stop loss."""
        self._stop_loss = value
        return self
    
    def entry_timestamp(self, value: datetime) -> "TradeOutcomeBuilder":
        """Set entry timestamp."""
        self._entry_timestamp = value
        return self
    
    def resolved_outcome(self, value: ResolvedOutcome) -> "TradeOutcomeBuilder":
        """Set resolved outcome."""
        self._resolved_outcome = value
        return self
    
    def lessons_learned(self, value: str) -> "TradeOutcomeBuilder":
        """Set lessons learned."""
        self._lessons_learned = value
        return self
    
    def build(self) -> TradeOutcome:
        """
        Build the trade outcome.
        
        Returns:
            TradeOutcome
        """
        if self._prediction_id is None:
            raise ValueError("Prediction ID is required")
        if self._symbol is None:
            raise ValueError("Symbol is required")
        if self._action is None:
            raise ValueError("Action is required")
        if self._predicted_probability is None:
            raise ValueError("Predicted probability is required")
        if self._predicted_confidence is None:
            raise ValueError("Predicted confidence is required")
        if self._expected_return is None:
            raise ValueError("Expected return is required")
        if self._entry_price is None:
            raise ValueError("Entry price is required")
        if self._entry_timestamp is None:
            raise ValueError("Entry timestamp is required")
        if self._resolved_outcome is None:
            raise ValueError("Resolved outcome is required")
        
        # Determine if successful
        is_successful = self._determine_success()
        
        return TradeOutcome(
            prediction_id=self._prediction_id,
            symbol=self._symbol,
            action=self._action,
            predicted_probability=self._predicted_probability,
            predicted_confidence=self._predicted_confidence,
            expected_return=self._expected_return,
            entry_price=self._entry_price,
            target_price=self._target_price,
            stop_loss=self._stop_loss,
            entry_timestamp=self._entry_timestamp,
            resolved_outcome=self._resolved_outcome,
            is_successful=is_successful,
            lessons_learned=self._lessons_learned,
        )
    
    def _determine_success(self) -> bool:
        """
        Determine if trade was successful.
        
        Returns:
            True if successful, False otherwise
        """
        if self._action == "HOLD":
            return True
        
        if self._resolved_outcome.outcome_type == "target_hit":
            return True
        elif self._resolved_outcome.outcome_type == "stop_hit":
            return False
        elif self._resolved_outcome.outcome_type == "timeout":
            return self._resolved_outcome.return_percentage > 0
        else:
            return False
    
    def reset(self) -> "TradeOutcomeBuilder":
        """Reset builder to initial state."""
        self._prediction_id = None
        self._symbol = None
        self._action = None
        self._predicted_probability = None
        self._predicted_confidence = None
        self._expected_return = None
        self._entry_price = None
        self._target_price = None
        self._stop_loss = None
        self._entry_timestamp = None
        self._resolved_outcome = None
        self._lessons_learned = None
        return self


def create_trade_outcome(
    prediction_id: str,
    symbol: str,
    action: str,
    predicted_probability: float,
    predicted_confidence: str,
    expected_return: float,
    entry_price: float,
    entry_timestamp: datetime,
    resolved_outcome: ResolvedOutcome,
    target_price: Optional[float] = None,
    stop_loss: Optional[float] = None,
) -> TradeOutcome:
    """
    Convenience function to create trade outcome.
    
    Args:
        prediction_id: Prediction ID
        symbol: Stock symbol
        action: Trading action
        predicted_probability: Predicted probability
        predicted_confidence: Predicted confidence level
        expected_return: Expected return
        entry_price: Entry price
        entry_timestamp: Entry timestamp
        resolved_outcome: Resolved outcome
        target_price: Target price
        stop_loss: Stop loss price
        
    Returns:
        TradeOutcome
    """
    builder = TradeOutcomeBuilder()
    return (
        builder.prediction_id(prediction_id)
        .symbol(symbol)
        .action(action)
        .predicted_probability(predicted_probability)
        .predicted_confidence(predicted_confidence)
        .expected_return(expected_return)
        .entry_price(entry_price)
        .entry_timestamp(entry_timestamp)
        .resolved_outcome(resolved_outcome)
        .target_price(target_price)
        .stop_loss(stop_loss)
        .build()
    )
