"""
Interaction Ranker

Ranks interactions by performance and quality.
Returns top N interactions for production use.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from research.interactions.interaction_engine.interaction_engine import InteractionResult
from research.interactions.interaction_engine.interaction_validator import InteractionValidationResult
from utils.logger import get_logger

logger = get_logger("research.interactions.interaction_ranker")


@dataclass
class RankedInteraction:
    """Ranked interaction with score."""
    interaction_id: str
    factor_name: str
    condition: dict
    ic: float
    sharpe: float
    win_rate: float
    num_trades: int
    decision: str
    rank_score: float
    confidence_level: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "interaction_id": self.interaction_id,
            "factor_name": self.factor_name,
            "condition": self.condition,
            "ic": round(self.ic, 4),
            "sharpe": round(self.sharpe, 4),
            "win_rate": round(self.win_rate, 4),
            "num_trades": self.num_trades,
            "decision": self.decision,
            "rank_score": round(self.rank_score, 4),
            "confidence_level": self.confidence_level,
        }


class InteractionRanker:
    """
    Ranks interactions by performance and quality.
    
    Ranks based on:
    - IC (Information Coefficient)
    - Sharpe ratio
    - Win rate
    - Sample size
    - Validation status
    """
    
    def __init__(
        self,
        ic_weight: float = 0.4,
        sharpe_weight: float = 0.3,
        win_rate_weight: float = 0.2,
        sample_size_weight: float = 0.1,
    ):
        """
        Initialize interaction ranker.
        
        Args:
            ic_weight: Weight for IC in ranking
            sharpe_weight: Weight for Sharpe in ranking
            win_rate_weight: Weight for win rate in ranking
            sample_size_weight: Weight for sample size in ranking
        """
        self.ic_weight = ic_weight
        self.sharpe_weight = sharpe_weight
        self.win_rate_weight = win_rate_weight
        self.sample_size_weight = sample_size_weight
        self._logger = get_logger("research.interactions.interaction_ranker")
    
    def rank_interactions(
        self,
        results: Dict[str, InteractionResult],
        validations: Optional[Dict[str, InteractionValidationResult]] = None,
    ) -> List[RankedInteraction]:
        """
        Rank interactions by performance.
        
        Args:
            results: Dictionary mapping interaction IDs to results
            validations: Optional dictionary mapping IDs to validation results
            
        Returns:
            List of RankedInteraction sorted by rank
        """
        ranked = []
        
        for interaction_id, result in results.items():
            # Calculate rank score
            rank_score = self._calculate_rank_score(result)
            
            # Get confidence level from validation if available
            confidence_level = "MEDIUM"
            if validations and interaction_id in validations:
                confidence_level = validations[interaction_id].confidence_level
            
            ranked.append(RankedInteraction(
                interaction_id=interaction_id,
                factor_name=result.factor_name,
                condition=result.condition.serialize(),
                ic=result.ic,
                sharpe=result.sharpe,
                win_rate=result.win_rate,
                num_trades=result.num_trades,
                decision=result.decision,
                rank_score=rank_score,
                confidence_level=confidence_level,
            ))
        
        # Sort by rank score descending
        ranked.sort(key=lambda x: x.rank_score, reverse=True)
        
        # Assign ranks
        for i, item in enumerate(ranked):
            item.interaction_id = f"rank_{i+1}_{item.interaction_id}"
        
        return ranked
    
    def _calculate_rank_score(self, result: InteractionResult) -> float:
        """
        Calculate composite rank score.
        
        Args:
            result: InteractionResult
            
        Returns:
            Composite score
        """
        # Normalize IC to 0-1 (assuming -0.2 to 0.2 range)
        norm_ic = (result.ic + 0.2) / 0.4
        norm_ic = max(0.0, min(1.0, norm_ic))
        
        # Normalize Sharpe to 0-1 (assuming -2 to 4 range)
        norm_sharpe = (result.sharpe + 2) / 6
        norm_sharpe = max(0.0, min(1.0, norm_sharpe))
        
        # Normalize win rate to 0-1
        norm_win_rate = result.win_rate
        
        # Normalize sample size to 0-1 (assuming 0 to 1000 range)
        norm_sample_size = min(result.num_trades / 1000.0, 1.0)
        
        # Weighted sum
        score = (
            norm_ic * self.ic_weight +
            norm_sharpe * self.sharpe_weight +
            norm_win_rate * self.win_rate_weight +
            norm_sample_size * self.sample_size_weight
        )
        
        # Boost for PASS decisions
        if result.decision == "PASS":
            score *= 1.2
        elif result.decision == "FAIL":
            score *= 0.5
        
        return score
    
    def get_top_n(
        self,
        ranked: List[RankedInteraction],
        n: int = 10,
        min_confidence: Optional[str] = None,
    ) -> List[RankedInteraction]:
        """
        Get top N ranked interactions.
        
        Args:
            ranked: List of RankedInteraction
            n: Number of top interactions to return
            min_confidence: Minimum confidence level (HIGH, MEDIUM, LOW)
            
        Returns:
            List of top N RankedInteraction
        """
        filtered = ranked
        
        # Filter by confidence if specified
        if min_confidence:
            confidence_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
            min_level = confidence_order.get(min_confidence, 0)
            filtered = [
                item for item in ranked
                if confidence_order.get(item.confidence_level, 0) >= min_level
            ]
        
        return filtered[:n]
    
    def filter_by_decision(
        self,
        ranked: List[RankedInteraction],
        decision: str,
    ) -> List[RankedInteraction]:
        """
        Filter interactions by decision.
        
        Args:
            ranked: List of RankedInteraction
            decision: Decision to filter by (PASS, FAIL, NEUTRAL)
            
        Returns:
            Filtered list of RankedInteraction
        """
        return [item for item in ranked if item.decision == decision]
    
    def get_factor_rankings(
        self,
        ranked: List[RankedInteraction],
        factor_name: str,
    ) -> List[RankedInteraction]:
        """
        Get rankings for a specific factor.
        
        Args:
            ranked: List of RankedInteraction
            factor_name: Factor name to filter by
            
        Returns:
            List of RankedInteraction for the factor
        """
        return [item for item in ranked if item.factor_name == factor_name]
    
    def summarize_rankings(self, ranked: List[RankedInteraction]) -> Dict:
        """
        Summarize ranking results.
        
        Args:
            ranked: List of RankedInteraction
            
        Returns:
            Summary statistics
        """
        total = len(ranked)
        passed = sum(1 for item in ranked if item.decision == "PASS")
        failed = sum(1 for item in ranked if item.decision == "FAIL")
        neutral = sum(1 for item in ranked if item.decision == "NEUTRAL")
        
        high_confidence = sum(1 for item in ranked if item.confidence_level == "HIGH")
        medium_confidence = sum(1 for item in ranked if item.confidence_level == "MEDIUM")
        low_confidence = sum(1 for item in ranked if item.confidence_level == "LOW")
        
        # Average metrics
        avg_ic = sum(item.ic for item in ranked) / total if total > 0 else 0.0
        avg_sharpe = sum(item.sharpe for item in ranked) / total if total > 0 else 0.0
        avg_win_rate = sum(item.win_rate for item in ranked) / total if total > 0 else 0.0
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "neutral": neutral,
            "high_confidence": high_confidence,
            "medium_confidence": medium_confidence,
            "low_confidence": low_confidence,
            "avg_ic": avg_ic,
            "avg_sharpe": avg_sharpe,
            "avg_win_rate": avg_win_rate,
        }


def rank_interactions(
    results: Dict[str, InteractionResult],
    validations: Optional[Dict[str, InteractionValidationResult]] = None,
    n: int = 10,
) -> List[RankedInteraction]:
    """
    Convenience function to rank and filter interactions.
    
    Args:
        results: Dictionary mapping interaction IDs to results
        validations: Optional dictionary mapping IDs to validation results
        n: Number of top interactions to return
        
    Returns:
        List of top N RankedInteraction
    """
    ranker = InteractionRanker()
    ranked = ranker.rank_interactions(results, validations)
    top_n = ranker.get_top_n(ranked, n)
    return top_n
