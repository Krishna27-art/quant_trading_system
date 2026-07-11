"""
Signal Scoring Engine

Scores and aggregates signals from different categories.
Calculates overall signal score with weighted categories.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from signal_engine.base import Signal, SignalCategory, SignalDirection, SignalSet
from utils.logger import get_logger

logger = get_logger("signal_engine.scoring")


@dataclass
class ScoringConfig:
    """Configuration for signal scoring."""
    category_weights: Dict[SignalCategory, float]
    min_confidence_threshold: float = 50.0
    require_direction_agreement: bool = True
    direction_agreement_threshold: float = 0.6


class SignalScorer:
    """
    Signal Scoring Engine.
    
    Scores signals by:
    1. Applying category weights
    2. Checking confidence thresholds
    3. Validating direction agreement
    4. Calculating overall score
    
    Default weights (can be customized):
    - Technical: 0.25
    - Volume: 0.20
    - Options: 0.20
    - Fundamental: 0.20
    - Sentiment: 0.15
    """
    
    def __init__(self, config: Optional[ScoringConfig] = None):
        """
        Initialize signal scorer.
        
        Args:
            config: Optional scoring configuration
        """
        self.config = config or self._default_config()
        self._logger = get_logger("signal_engine.scoring")
    
    def _default_config(self) -> ScoringConfig:
        """Create default scoring configuration."""
        return ScoringConfig(
            category_weights={
                SignalCategory.TECHNICAL: 0.25,
                SignalCategory.VOLUME: 0.20,
                SignalCategory.OPTIONS: 0.20,
                SignalCategory.FUNDAMENTAL: 0.20,
                SignalCategory.SENTIMENT: 0.15,
                SignalCategory.MACRO: 0.10,
                SignalCategory.SECTOR: 0.10,
                SignalCategory.MARKET: 0.15,
            },
            min_confidence_threshold=50.0,
            require_direction_agreement=True,
            direction_agreement_threshold=0.6,
        )
    
    def score_signal_set(self, signal_set: SignalSet) -> Dict:
        """
        Score a signal set.
        
        Args:
            signal_set: SignalSet containing signals from multiple categories
            
        Returns:
            Dictionary with scoring results:
            - overall_score: Overall signal score (0-100)
            - weighted_score: Weighted score considering category weights
            - direction_agreement: Agreement level among directions
            - passes_filters: Whether signal passes all filters
            - breakdown: Score breakdown by category
        """
        if not signal_set.signals:
            self._logger.warning("Empty signal set")
            return {
                'overall_score': 50.0,
                'weighted_score': 50.0,
                'direction_agreement': 0.0,
                'passes_filters': False,
                'breakdown': {},
                'reason': 'No signals available',
            }
        
        # Calculate weighted score
        weighted_score = self._calculate_weighted_score(signal_set)
        
        # Calculate direction agreement
        direction_agreement = self._calculate_direction_agreement(signal_set)
        
        # Check confidence threshold
        passes_confidence = self._check_confidence_threshold(signal_set)
        
        # Check direction agreement
        passes_direction = True
        if self.config.require_direction_agreement:
            passes_direction = direction_agreement >= self.config.direction_agreement_threshold
        
        # Overall pass/fail
        passes_filters = passes_confidence and passes_direction
        
        # Build breakdown
        breakdown = {}
        for category, signal in signal_set.signals.items():
            weight = self.config.category_weights.get(category, 0.0)
            breakdown[category.value] = {
                'score': signal.score,
                'weight': weight,
                'weighted_score': signal.score * weight,
                'direction': signal.direction.value,
                'confidence': signal.confidence,
            }
        
        return {
            'overall_score': weighted_score,
            'weighted_score': weighted_score,
            'direction_agreement': direction_agreement,
            'passes_filters': passes_filters,
            'breakdown': breakdown,
            'reason': self._build_reason(passes_filters, passes_confidence, passes_direction),
        }
    
    def _calculate_weighted_score(self, signal_set: SignalSet) -> float:
        """Calculate weighted score from signal set."""
        total_weight = 0.0
        weighted_sum = 0.0
        
        for category, signal in signal_set.signals.items():
            weight = self.config.category_weights.get(category, 0.0)
            
            # Only include signals that meet confidence threshold
            if signal.confidence >= self.config.min_confidence_threshold:
                weighted_sum += signal.score * weight
                total_weight += weight
        
        if total_weight == 0:
            return 50.0
        
        return weighted_sum / total_weight
    
    def _calculate_direction_agreement(self, signal_set: SignalSet) -> float:
        """
        Calculate agreement level among signal directions.
        
        Returns:
            Agreement score (0 to 1)
        """
        if not signal_set.signals:
            return 0.0
        
        directions = [signal.direction for signal in signal_set.signals.values()]
        
        # Count each direction
        bullish_count = sum(1 for d in directions if d == SignalDirection.BULLISH)
        bearish_count = sum(1 for d in directions if d == SignalDirection.BEARISH)
        neutral_count = sum(1 for d in directions if d == SignalDirection.NEUTRAL)
        
        total = len(directions)
        
        # Agreement is the maximum proportion of any single direction
        max_count = max(bullish_count, bearish_count, neutral_count)
        agreement = max_count / total
        
        return agreement
    
    def _check_confidence_threshold(self, signal_set: SignalSet) -> bool:
        """Check if signals meet minimum confidence threshold."""
        for signal in signal_set.signals.values():
            if signal.confidence < self.config.min_confidence_threshold:
                return False
        return True
    
    def _build_reason(self, passes_filters: bool, passes_confidence: bool, passes_direction: bool) -> str:
        """Build reason string for scoring result."""
        if passes_filters:
            return "Signal passes all filters"
        
        reasons = []
        if not passes_confidence:
            reasons.append(f"Below confidence threshold ({self.config.min_confidence_threshold}%)")
        if not passes_direction:
            reasons.append(f"Insufficient direction agreement ({self.config.direction_agreement_threshold})")
        
        return "; ".join(reasons)
    
    def get_top_signals(
        self,
        signal_sets: Dict[str, SignalSet],
        top_n: int = 10,
    ) -> list:
        """
        Get top N signal sets by score.
        
        Args:
            signal_sets: Dictionary mapping symbols to SignalSets
            top_n: Number of top signals to return
            
        Returns:
            List of (symbol, score) tuples sorted by score
        """
        scored = []
        
        for symbol, signal_set in signal_sets.items():
            scoring_result = self.score_signal_set(signal_set)
            scored.append((symbol, scoring_result['overall_score']))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored[:top_n]
