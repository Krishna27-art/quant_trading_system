"""
Recommendation Engine

Generates final trading recommendations (BUY/SELL/HOLD).
Includes entry, stop, target, and reasoning.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime

from meta_alpha.probability_engine.probability_engine import ProbabilityResult
from meta_alpha.confidence_engine.confidence_engine import ConfidenceResult
from meta_alpha.return_engine.return_distribution import ReturnDistribution
from utils.logger import get_logger

logger = get_logger("meta_alpha.recommendation_engine")


@dataclass
class Recommendation:
    """Final trading recommendation."""
    symbol: str
    action: str  # "BUY", "SELL", "HOLD"
    probability: float
    confidence: str
    expected_return: float
    risk_level: str
    entry_price: Optional[float]
    stop_loss: Optional[float]
    target_price: Optional[float]
    holding_period: int  # days
    reason: str
    timestamp: datetime
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate recommendation.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check action is valid
        valid_actions = ["BUY", "SELL", "HOLD"]
        if self.action not in valid_actions:
            errors.append(f"Invalid action: {self.action}")
        
        # Check probability is between 0 and 1
        if not (0.0 <= self.probability <= 1.0):
            errors.append(f"Probability must be between 0 and 1, got {self.probability}")
        
        # Check confidence level is valid
        valid_confidences = ["HIGH", "MEDIUM", "LOW"]
        if self.confidence not in valid_confidences:
            errors.append(f"Invalid confidence level: {self.confidence}")
        
        # Check risk level is valid
        valid_risks = ["LOW", "MEDIUM", "HIGH"]
        if self.risk_level not in valid_risks:
            errors.append(f"Invalid risk level: {self.risk_level}")
        
        # Check price relationships
        if self.entry_price and self.stop_loss:
            if self.action == "BUY" and self.stop_loss >= self.entry_price:
                errors.append("For BUY, stop loss must be below entry price")
            elif self.action == "SELL" and self.stop_loss <= self.entry_price:
                errors.append("For SELL, stop loss must be above entry price")
        
        if self.entry_price and self.target_price:
            if self.action == "BUY" and self.target_price <= self.entry_price:
                errors.append("For BUY, target must be above entry price")
            elif self.action == "SELL" and self.target_price >= self.entry_price:
                errors.append("For SELL, target must be below entry price")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "action": self.action,
            "probability": round(self.probability, 4),
            "confidence": self.confidence,
            "expected_return": round(self.expected_return, 4),
            "risk_level": self.risk_level,
            "entry_price": round(self.entry_price, 2) if self.entry_price else None,
            "stop_loss": round(self.stop_loss, 2) if self.stop_loss else None,
            "target_price": round(self.target_price, 2) if self.target_price else None,
            "holding_period": self.holding_period,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }


class RecommendationEngine:
    """
    Generates final trading recommendations.
    
    Process:
    - Takes probability, confidence, and return distribution
    - Applies risk thresholds
    - Generates entry/stop/target
    - Provides reasoning
    """
    
    def __init__(
        self,
        buy_threshold: float = 0.6,
        sell_threshold: float = 0.4,
        min_confidence: str = "MEDIUM",
    ):
        """
        Initialize recommendation engine.
        
        Args:
            buy_threshold: Probability threshold for BUY
            sell_threshold: Probability threshold for SELL
            min_confidence: Minimum confidence level
        """
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_confidence = min_confidence
        self._logger = get_logger("meta_alpha.recommendation_engine")
    
    def generate_recommendation(
        self,
        symbol: str,
        current_price: float,
        probability_result: ProbabilityResult,
        confidence_result: ConfidenceResult,
        return_distribution: ReturnDistribution,
    ) -> Recommendation:
        """
        Generate trading recommendation.
        
        Args:
            symbol: Stock symbol
            current_price: Current price
            probability_result: Probability result
            confidence_result: Confidence result
            return_distribution: Return distribution
            
        Returns:
            Recommendation
        """
        # Determine action
        action = self._determine_action(
            probability_result.probability,
            confidence_result.confidence_level,
        )
        
        # Calculate risk level
        risk_level = self._calculate_risk_level(return_distribution)
        
        # Calculate entry, stop, target
        entry_price, stop_loss, target_price = self._calculate_price_levels(
            action,
            current_price,
            return_distribution,
        )
        
        # Calculate holding period
        holding_period = self._calculate_holding_period(
            action,
            return_distribution,
        )
        
        # Generate reason
        reason = self._generate_reason(
            action,
            probability_result,
            confidence_result,
            return_distribution,
        )
        
        return Recommendation(
            symbol=symbol,
            action=action,
            probability=probability_result.probability,
            confidence=confidence_result.confidence_level,
            expected_return=return_distribution.expected_return,
            risk_level=risk_level,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            holding_period=holding_period,
            reason=reason,
            timestamp=datetime.now(),
        )
    
    def _determine_action(
        self,
        probability: float,
        confidence_level: str,
    ) -> str:
        """
        Determine action from probability and confidence.
        
        Args:
            probability: Probability of success
            confidence_level: Confidence level
            
        Returns:
            Action: "BUY", "SELL", or "HOLD"
        """
        # Check minimum confidence
        confidence_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        min_level = confidence_order.get(self.min_confidence, 2)
        current_level = confidence_order.get(confidence_level, 0)
        
        if current_level < min_level:
            return "HOLD"
        
        # Determine action based on probability
        if probability >= self.buy_threshold:
            return "BUY"
        elif probability <= self.sell_threshold:
            return "SELL"
        else:
            return "HOLD"
    
    def _calculate_risk_level(self, return_distribution: ReturnDistribution) -> str:
        """
        Calculate risk level from return distribution.
        
        Args:
            return_distribution: Return distribution
            
        Returns:
            Risk level: "LOW", "MEDIUM", or "HIGH"
        """
        downside_risk = return_distribution.risk_metrics.get("downside_risk", 0.0)
        
        if downside_risk < 0.02:
            return "LOW"
        elif downside_risk < 0.05:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _calculate_price_levels(
        self,
        action: str,
        current_price: float,
        return_distribution: ReturnDistribution,
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calculate entry, stop loss, and target price.
        
        Args:
            action: Trading action
            current_price: Current price
            return_distribution: Return distribution
            
        Returns:
            Tuple of (entry_price, stop_loss, target_price)
        """
        if action == "HOLD":
            return None, None, None
        
        entry_price = current_price
        
        # Calculate stop loss based on worst case
        worst_case = return_distribution.worst_case
        if action == "BUY":
            stop_loss = current_price * (1.0 + worst_case)
        else:
            stop_loss = current_price * (1.0 - worst_case)
        
        # Calculate target based on best case
        best_case = return_distribution.best_case
        if action == "BUY":
            target_price = current_price * (1.0 + best_case)
        else:
            target_price = current_price * (1.0 - best_case)
        
        return entry_price, stop_loss, target_price
    
    def _calculate_holding_period(
        self,
        action: str,
        return_distribution: ReturnDistribution,
    ) -> int:
        """
        Calculate optimal holding period.
        
        Args:
            action: Trading action
            return_distribution: Return distribution
            
        Returns:
            Holding period in days
        """
        if action == "HOLD":
            return 0
        
        # Base holding period
        base_period = 5
        
        # Adjust based on expected return
        if abs(return_distribution.expected_return) > 0.05:
            base_period = 7
        elif abs(return_distribution.expected_return) > 0.03:
            base_period = 5
        else:
            base_period = 3
        
        return base_period
    
    def _generate_reason(
        self,
        action: str,
        probability_result: ProbabilityResult,
        confidence_result: ConfidenceResult,
        return_distribution: ReturnDistribution,
    ) -> str:
        """
        Generate reason for recommendation.
        
        Args:
            action: Trading action
            probability_result: Probability result
            confidence_result: Confidence result
            return_distribution: Return distribution
            
        Returns:
            Reason string
        """
        parts = []
        
        parts.append(f"Action: {action}")
        parts.append(f"Probability: {probability_result.probability:.1%}")
        parts.append(f"Confidence: {confidence_result.confidence_level}")
        parts.append(f"Expected Return: {return_distribution.expected_return:.2%}")
        parts.append(f"Risk/Reward: {return_distribution.risk_metrics.get('risk_reward_ratio', 0):.2f}")
        
        return " | ".join(parts)
    
    def batch_generate_recommendations(
        self,
        signals: list[Dict[str, Any]],
        probability_results: list[ProbabilityResult],
        confidence_results: list[ConfidenceResult],
        return_distributions: list[ReturnDistribution],
    ) -> list[Recommendation]:
        """
        Generate recommendations for multiple signals.
        
        Args:
            signals: List of signal dictionaries
            probability_results: List of probability results
            confidence_results: List of confidence results
            return_distributions: List of return distributions
            
        Returns:
            List of Recommendation
        """
        recommendations = []
        
        for i, signal in enumerate(signals):
            try:
                recommendation = self.generate_recommendation(
                    symbol=signal["symbol"],
                    current_price=signal["current_price"],
                    probability_result=probability_results[i],
                    confidence_result=confidence_results[i],
                    return_distribution=return_distributions[i],
                )
                recommendations.append(recommendation)
            except Exception as e:
                self._logger.error(f"Failed to generate recommendation for {signal['symbol']}: {e}")
        
        return recommendations


def generate_recommendation(
    symbol: str,
    current_price: float,
    probability_result: ProbabilityResult,
    confidence_result: ConfidenceResult,
    return_distribution: ReturnDistribution,
) -> Recommendation:
    """
    Convenience function to generate recommendation.
    
    Args:
        symbol: Stock symbol
        current_price: Current price
        probability_result: Probability result
        confidence_result: Confidence result
        return_distribution: Return distribution
        
    Returns:
        Recommendation
    """
    engine = RecommendationEngine()
    return engine.generate_recommendation(
        symbol,
        current_price,
        probability_result,
        confidence_result,
        return_distribution,
    )
