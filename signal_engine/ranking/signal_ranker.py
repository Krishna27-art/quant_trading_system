"""
Signal Ranking Engine

Ranks stocks by signal quality.
Only top N stocks proceed to prediction models.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from signal_engine.base import Signal, SignalCategory, SignalDirection, SignalSet, SignalRanking
from signal_engine.scoring import SignalScorer, ScoringConfig
from utils.logger import get_logger

logger = get_logger("signal_engine.ranking")


@dataclass
class RankingCriteria:
    """Criteria for ranking signals."""
    weights: Dict[str, float]
    require_minimum_categories: int = 2
    require_direction_agreement: bool = True


class SignalRanker:
    """
    Signal Ranking Engine.
    
    Ranks signals by:
    1. Overall signal score
    2. Category diversity
    3. Direction agreement
    4. Signal confidence
    
    Returns ranked list of stocks for prediction.
    """
    
    def __init__(
        self,
        criteria: Optional[RankingCriteria] = None,
        scorer: Optional[SignalScorer] = None,
    ):
        """
        Initialize signal ranker.
        
        Args:
            criteria: Ranking criteria
            scorer: Signal scorer for calculating scores
        """
        self.criteria = criteria or self._default_criteria()
        self.scorer = scorer or SignalScorer()
        self._logger = get_logger("signal_engine.ranking")
    
    def _default_criteria(self) -> RankingCriteria:
        """Create default ranking criteria."""
        return RankingCriteria(
            weights={
                'overall_score': 0.5,
                'category_diversity': 0.2,
                'direction_agreement': 0.15,
                'average_confidence': 0.15,
            },
            require_minimum_categories=2,
            require_direction_agreement=True,
        )
    
    def rank_signal_sets(
        self,
        signal_sets: Dict[str, SignalSet],
    ) -> List[SignalRanking]:
        """
        Rank multiple signal sets.
        
        Args:
            signal_sets: Dictionary mapping symbols to SignalSets
            
        Returns:
            List of SignalRanking objects sorted by rank
        """
        rankings = []
        
        for symbol, signal_set in signal_sets.items():
            # Calculate ranking score
            ranking_score = self._calculate_ranking_score(signal_set)
            
            rankings.append(
                SignalRanking(
                    symbol=symbol,
                    signal_set=signal_set,
                    rank=0,  # Will be assigned after sorting
                    overall_score=ranking_score,
                )
            )
        
        # Sort by ranking score descending
        rankings.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Assign ranks
        for i, ranking in enumerate(rankings):
            ranking.rank = i + 1
        
        return rankings
    
    def _calculate_ranking_score(self, signal_set: SignalSet) -> float:
        """
        Calculate ranking score for a signal set.
        
        Args:
            signal_set: SignalSet to score
            
        Returns:
            Ranking score (0-100)
        """
        # Get scoring result
        scoring_result = self.scorer.score_signal_set(signal_set)
        
        # Calculate individual components
        overall_score = scoring_result['overall_score']
        category_diversity = self._calculate_category_diversity(signal_set)
        direction_agreement = scoring_result['direction_agreement']
        average_confidence = self._calculate_average_confidence(signal_set)
        
        # Apply weights
        ranking_score = (
            overall_score * self.criteria.weights['overall_score'] +
            category_diversity * self.criteria.weights['category_diversity'] +
            direction_agreement * 100 * self.criteria.weights['direction_agreement'] +
            average_confidence * self.criteria.weights['average_confidence']
        )
        
        return ranking_score
    
    def _calculate_category_diversity(self, signal_set: SignalSet) -> float:
        """
        Calculate category diversity score.
        
        More diverse categories = higher score.
        
        Returns:
            Diversity score (0-100)
        """
        num_categories = len(signal_set.signals)
        max_categories = len(SignalCategory)
        
        diversity = (num_categories / max_categories) * 100
        return diversity
    
    def _calculate_average_confidence(self, signal_set: SignalSet) -> float:
        """Calculate average confidence across all signals."""
        if not signal_set.signals:
            return 0.0
        
        total_confidence = sum(signal.confidence for signal in signal_set.signals.values())
        return total_confidence / len(signal_set.signals)
    
    def get_top_n(
        self,
        signal_sets: Dict[str, SignalSet],
        n: int = 10,
    ) -> List[SignalRanking]:
        """
        Get top N ranked signal sets.
        
        Args:
            signal_sets: Dictionary mapping symbols to SignalSets
            n: Number of top signals to return
            
        Returns:
            List of top N SignalRanking objects
        """
        rankings = self.rank_signal_sets(signal_sets)
        return rankings[:n]
    
    def get_ranking_summary(
        self,
        rankings: List[SignalRanking],
    ) -> Dict:
        """
        Get summary of ranking results.
        
        Args:
            rankings: List of SignalRanking objects
            
        Returns:
            Dictionary with ranking summary
        """
        if not rankings:
            return {
                'total_ranked': 0,
                'top_score': 0.0,
                'bottom_score': 0.0,
                'average_score': 0.0,
            }
        
        scores = [ranking.overall_score for ranking in rankings]
        
        return {
            'total_ranked': len(rankings),
            'top_score': scores[0],
            'bottom_score': scores[-1],
            'average_score': sum(scores) / len(scores),
            'top_10': [r.symbol for r in rankings[:10]],
            'top_20': [r.symbol for r in rankings[:20]],
        }
    
    def filter_by_minimum_categories(
        self,
        signal_sets: Dict[str, SignalSet],
    ) -> Dict[str, SignalSet]:
        """
        Filter signal sets by minimum number of categories.
        
        Args:
            signal_sets: Dictionary mapping symbols to SignalSets
            
        Returns:
            Filtered dictionary
        """
        filtered = {}
        
        for symbol, signal_set in signal_sets.items():
            if len(signal_set.signals) >= self.criteria.require_minimum_categories:
                filtered[symbol] = signal_set
        
        return filtered
    
    def filter_by_direction_agreement(
        self,
        signal_sets: Dict[str, SignalSet],
    ) -> Dict[str, SignalSet]:
        """
        Filter signal sets by direction agreement.
        
        Args:
            signal_sets: Dictionary mapping symbols to SignalSets
            
        Returns:
            Filtered dictionary
        """
        if not self.criteria.require_direction_agreement:
            return signal_sets
        
        filtered = {}
        
        for symbol, signal_set in signal_sets.items():
            scoring_result = self.scorer.score_signal_set(signal_set)
            if scoring_result['direction_agreement'] >= 0.6:
                filtered[symbol] = signal_set
        
        return filtered
