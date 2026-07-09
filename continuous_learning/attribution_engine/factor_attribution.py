"""
Factor Attribution

Attributes trade outcomes to individual factors.
Answers "WHY" a trade succeeded or failed.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

from meta_alpha.evidence_engine.evidence import Evidence
from continuous_learning.outcome_engine.trade_outcome import TradeOutcome
from utils.logger import get_logger

logger = get_logger("continuous_learning.attribution_engine")


@dataclass
class FactorContribution:
    """Contribution of a single factor to trade outcome."""
    factor_name: str
    source: str
    category: str
    direction: str
    contribution_score: float
    weight: float
    was_correct: bool
    attribution: str  # "positive", "negative", "neutral"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "factor_name": self.factor_name,
            "source": self.source,
            "category": self.category,
            "direction": self.direction,
            "contribution_score": round(self.contribution_score, 4),
            "weight": round(self.weight, 4),
            "was_correct": self.was_correct,
            "attribution": self.attribution,
        }


@dataclass
class AttributionResult:
    """Result of factor attribution analysis."""
    prediction_id: str
    symbol: str
    outcome_type: str
    actual_return: float
    positive_contributors: List[FactorContribution]
    negative_contributors: List[FactorContribution]
    neutral_contributors: List[FactorContribution]
    total_attribution_score: float
    attribution_accuracy: float
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate attribution result.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check total attribution is reasonable
        if abs(self.total_attribution_score) > 1.0:
            errors.append(f"Total attribution score too extreme: {self.total_attribution_score}")
        
        # Check attribution accuracy is between 0 and 1
        if not (0.0 <= self.attribution_accuracy <= 1.0):
            errors.append(f"Attribution accuracy must be between 0 and 1, got {self.attribution_accuracy}")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "prediction_id": self.prediction_id,
            "symbol": self.symbol,
            "outcome_type": self.outcome_type,
            "actual_return": round(self.actual_return, 4),
            "positive_contributors": [c.to_dict() for c in self.positive_contributors],
            "negative_contributors": [c.to_dict() for c in self.negative_contributors],
            "neutral_contributors": [c.to_dict() for c in self.neutral_contributors],
            "total_attribution_score": round(self.total_attribution_score, 4),
            "attribution_accuracy": round(self.attribution_accuracy, 4),
        }


class FactorAttributor:
    """
    Attributes trade outcomes to individual factors.
    
    Process:
    - Analyze each factor's contribution
    - Determine if factor was correct
    - Calculate attribution score
    - Classify as positive/negative/neutral
    """
    
    def __init__(self):
        """Initialize factor attributor."""
        self._logger = get_logger("continuous_learning.attribution_engine")
    
    def attribute(
        self,
        trade_outcome: TradeOutcome,
        evidence_list: List[Evidence],
        weights: Optional[Dict[str, float]] = None,
    ) -> AttributionResult:
        """
        Attribute trade outcome to factors.
        
        Args:
            trade_outcome: TradeOutcome to attribute
            evidence_list: List of Evidence used in prediction
            weights: Optional weights for evidence
            
        Returns:
            AttributionResult
        """
        contributions = []
        
        for evidence in evidence_list:
            evidence_id = f"{evidence.source}_{evidence.factor_name}"
            weight = weights.get(evidence_id, 1.0) if weights else 1.0
            
            # Calculate contribution score
            contribution_score = self._calculate_contribution_score(
                evidence,
                trade_outcome,
                weight,
            )
            
            # Determine if factor was correct
            was_correct = self._was_factor_correct(evidence, trade_outcome)
            
            # Determine attribution
            attribution = self._determine_attribution(contribution_score, was_correct)
            
            contribution = FactorContribution(
                factor_name=evidence.factor_name,
                source=evidence.source,
                category=evidence.category,
                direction=evidence.signal_direction,
                contribution_score=contribution_score,
                weight=weight,
                was_correct=was_correct,
                attribution=attribution,
            )
            
            contributions.append(contribution)
        
        # Separate by attribution
        positive = [c for c in contributions if c.attribution == "positive"]
        negative = [c for c in contributions if c.attribution == "negative"]
        neutral = [c for c in contributions if c.attribution == "neutral"]
        
        # Sort by contribution score
        positive.sort(key=lambda x: x.contribution_score, reverse=True)
        negative.sort(key=lambda x: x.contribution_score)
        
        # Calculate total attribution score
        total_attribution = sum(c.contribution_score for c in contributions)
        
        # Calculate attribution accuracy
        attribution_accuracy = self._calculate_attribution_accuracy(contributions)
        
        return AttributionResult(
            prediction_id=trade_outcome.prediction_id,
            symbol=trade_outcome.symbol,
            outcome_type=trade_outcome.resolved_outcome.outcome_type,
            actual_return=trade_outcome.resolved_outcome.return_percentage,
            positive_contributors=positive,
            negative_contributors=negative,
            neutral_contributors=neutral,
            total_attribution_score=total_attribution,
            attribution_accuracy=attribution_accuracy,
        )
    
    def _calculate_contribution_score(
        self,
        evidence: Evidence,
        trade_outcome: TradeOutcome,
        weight: float,
    ) -> float:
        """
        Calculate contribution score for evidence.
        
        Args:
            evidence: Evidence
            trade_outcome: TradeOutcome
            weight: Evidence weight
            
        Returns:
            Contribution score
        """
        # Get bullish score from evidence
        bullish_score = evidence.get_bullish_score()
        
        # Get actual return
        actual_return = trade_outcome.resolved_outcome.return_percentage
        
        # Calculate correlation between evidence direction and actual return
        # If evidence was bullish and return was positive -> positive contribution
        # If evidence was bearish and return was negative -> positive contribution
        contribution = bullish_score * actual_return * weight
        
        return contribution
    
    def _was_factor_correct(
        self,
        evidence: Evidence,
        trade_outcome: TradeOutcome,
    ) -> bool:
        """
        Determine if factor was correct.
        
        Args:
            evidence: Evidence
            trade_outcome: TradeOutcome
            
        Returns:
            True if factor was correct
        """
        actual_return = trade_outcome.resolved_outcome.return_percentage
        
        # Factor is correct if its direction matches actual return
        if evidence.is_bullish():
            return actual_return > 0
        elif evidence.is_bearish():
            return actual_return < 0
        else:
            return True  # Neutral is always "correct"
    
    def _determine_attribution(
        self,
        contribution_score: float,
        was_correct: bool,
    ) -> str:
        """
        Determine attribution category.
        
        Args:
            contribution_score: Contribution score
            was_correct: Whether factor was correct
            
        Returns:
            Attribution: "positive", "negative", or "neutral"
        """
        if contribution_score > 0.01:
            return "positive"
        elif contribution_score < -0.01:
            return "negative"
        else:
            return "neutral"
    
    def _calculate_attribution_accuracy(self, contributions: List[FactorContribution]) -> float:
        """
        Calculate attribution accuracy.
        
        Args:
            contributions: List of FactorContribution
            
        Returns:
            Attribution accuracy (0-1)
        """
        if not contributions:
            return 0.0
        
        correct_count = sum(1 for c in contributions if c.was_correct)
        return correct_count / len(contributions)
    
    def batch_attribute(
        self,
        trade_outcomes: List[TradeOutcome],
        evidence_by_prediction: Dict[str, List[Evidence]],
        weights_by_prediction: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> List[AttributionResult]:
        """
        Attribute multiple trade outcomes.
        
        Args:
            trade_outcomes: List of TradeOutcome
            evidence_by_prediction: Dictionary mapping prediction IDs to evidence lists
            weights_by_prediction: Optional dictionary mapping prediction IDs to weights
            
        Returns:
            List of AttributionResult
        """
        results = []
        
        for trade_outcome in trade_outcomes:
            evidence_list = evidence_by_prediction.get(trade_outcome.prediction_id, [])
            weights = weights_by_prediction.get(trade_outcome.prediction_id) if weights_by_prediction else None
            
            try:
                attribution = self.attribute(trade_outcome, evidence_list, weights)
                results.append(attribution)
            except Exception as e:
                self._logger.error(f"Failed to attribute {trade_outcome.prediction_id}: {e}")
        
        return results
    
    def aggregate_factor_performance(
        self,
        attributions: List[AttributionResult],
    ) -> Dict[str, Dict]:
        """
        Aggregate factor performance across multiple attributions.
        
        Args:
            attributions: List of AttributionResult
            
        Returns:
            Dictionary mapping factor names to performance metrics
        """
        factor_stats = {}
        
        for attribution in attributions:
            all_contributors = (
                attribution.positive_contributors +
                attribution.negative_contributors +
                attribution.neutral_contributors
            )
            
            for contributor in all_contributors:
                factor_name = contributor.factor_name
                
                if factor_name not in factor_stats:
                    factor_stats[factor_name] = {
                        "total_trades": 0,
                        "correct_trades": 0,
                        "total_contribution": 0.0,
                        "positive_count": 0,
                        "negative_count": 0,
                        "neutral_count": 0,
                    }
                
                stats = factor_stats[factor_name]
                stats["total_trades"] += 1
                stats["total_contribution"] += contributor.contribution_score
                
                if contributor.was_correct:
                    stats["correct_trades"] += 1
                
                if contributor.attribution == "positive":
                    stats["positive_count"] += 1
                elif contributor.attribution == "negative":
                    stats["negative_count"] += 1
                else:
                    stats["neutral_count"] += 1
        
        # Calculate accuracy for each factor
        for factor_name, stats in factor_stats.items():
            if stats["total_trades"] > 0:
                stats["accuracy"] = stats["correct_trades"] / stats["total_trades"]
                stats["avg_contribution"] = stats["total_contribution"] / stats["total_trades"]
            else:
                stats["accuracy"] = 0.0
                stats["avg_contribution"] = 0.0
        
        return factor_stats


def attribute_trade_outcome(
    trade_outcome: TradeOutcome,
    evidence_list: List[Evidence],
) -> AttributionResult:
    """
    Convenience function to attribute trade outcome.
    
    Args:
        trade_outcome: TradeOutcome to attribute
        evidence_list: List of Evidence
        
    Returns:
        AttributionResult
    """
    attributor = FactorAttributor()
    return attributor.attribute(trade_outcome, evidence_list)
