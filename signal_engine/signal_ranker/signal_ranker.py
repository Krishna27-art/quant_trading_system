"""
Signal Ranker Engine

Ranks trading signals by quality and expected value.
Returns top N opportunities from candidate signals.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger("signal_engine.signal_ranker")


@dataclass
class RankedSignal:
    """Ranked trading signal."""
    symbol: str
    rank: int
    probability: float
    expected_value: float
    confidence: float
    quality_score: float
    risk_reward_ratio: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "rank": self.rank,
            "probability": round(self.probability, 4),
            "expected_value": round(self.expected_value, 4),
            "confidence": round(self.confidence, 4),
            "quality_score": round(self.quality_score, 2),
            "risk_reward_ratio": round(self.risk_reward_ratio, 4),
        }


class SignalRanker:
    """
    Ranks trading signals by quality and expected value.
    
    Ranks signals based on:
    - Probability
    - Expected value
    - Confidence
    - Quality score
    - Risk/reward ratio
    """
    
    def __init__(
        self,
        ranking_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize signal ranker.
        
        Args:
            ranking_weights: Optional weights for ranking criteria
        """
        self.ranking_weights = ranking_weights or {
            "probability": 0.3,
            "expected_value": 0.25,
            "confidence": 0.2,
            "quality": 0.15,
            "risk_reward": 0.1,
        }
        self._logger = get_logger("signal_engine.signal_ranker")
    
    def rank_signals(
        self,
        signals: Dict[str, Dict],
    ) -> List[RankedSignal]:
        """
        Rank trading signals.
        
        Args:
            signals: Dictionary mapping symbols to signal data
            
        Returns:
            List of RankedSignal sorted by rank
        """
        ranked = []
        
        for symbol, signal_data in signals.items():
            # Calculate composite score
            score = self._calculate_ranking_score(signal_data)
            
            ranked.append({
                "symbol": symbol,
                "score": score,
                "probability": signal_data.get("probability", 0.5),
                "expected_value": signal_data.get("expected_value", 0.0),
                "confidence": signal_data.get("confidence", 0.5),
                "quality_score": signal_data.get("quality_score", 50.0),
                "risk_reward_ratio": signal_data.get("risk_reward_ratio", 1.0),
            })
        
        # Sort by score descending
        ranked.sort(key=lambda x: x["score"], reverse=True)
        
        # Convert to RankedSignal with ranks
        ranked_signals = []
        for i, item in enumerate(ranked):
            ranked_signals.append(RankedSignal(
                symbol=item["symbol"],
                rank=i + 1,
                probability=item["probability"],
                expected_value=item["expected_value"],
                confidence=item["confidence"],
                quality_score=item["quality_score"],
                risk_reward_ratio=item["risk_reward_ratio"],
            ))
        
        return ranked_signals
    
    def _calculate_ranking_score(self, signal_data: Dict) -> float:
        """
        Calculate composite ranking score.
        
        Args:
            signal_data: Signal data dictionary
            
        Returns:
            Composite score
        """
        probability = signal_data.get("probability", 0.5)
        expected_value = signal_data.get("expected_value", 0.0)
        confidence = signal_data.get("confidence", 0.5)
        quality_score = signal_data.get("quality_score", 50.0)
        risk_reward_ratio = signal_data.get("risk_reward_ratio", 1.0)
        
        # Normalize to 0-1 scale
        norm_probability = probability
        norm_expected_value = min(max(expected_value / 0.1, 0), 1)  # Assume 10% is max
        norm_confidence = confidence
        norm_quality = quality_score / 100.0
        norm_risk_reward = min(risk_reward_ratio / 3.0, 1)  # Assume 3:1 is max
        
        # Weighted sum
        score = (
            norm_probability * self.ranking_weights["probability"] +
            norm_expected_value * self.ranking_weights["expected_value"] +
            norm_confidence * self.ranking_weights["confidence"] +
            norm_quality * self.ranking_weights["quality"] +
            norm_risk_reward * self.ranking_weights["risk_reward"]
        )
        
        return score
    
    def get_top_n(
        self,
        ranked_signals: List[RankedSignal],
        n: int = 10,
    ) -> List[RankedSignal]:
        """
        Get top N ranked signals.
        
        Args:
            ranked_signals: List of RankedSignal
            n: Number of top signals to return
            
        Returns:
            List of top N RankedSignal
        """
        return ranked_signals[:n]
    
    def filter_by_threshold(
        self,
        ranked_signals: List[RankedSignal],
        min_probability: float = 0.6,
        min_confidence: float = 0.5,
        min_quality: float = 70.0,
    ) -> List[RankedSignal]:
        """
        Filter signals by minimum thresholds.
        
        Args:
            ranked_signals: List of RankedSignal
            min_probability: Minimum probability threshold
            min_confidence: Minimum confidence threshold
            min_quality: Minimum quality threshold
            
        Returns:
            Filtered list of RankedSignal
        """
        filtered = [
            signal for signal in ranked_signals
            if signal.probability >= min_probability
            and signal.confidence >= min_confidence
            and signal.quality_score >= min_quality
        ]
        
        return filtered


def rank_signals(
    signals: Dict[str, Dict],
    n: int = 10,
    min_probability: float = 0.6,
    min_confidence: float = 0.5,
    min_quality: float = 70.0,
) -> List[RankedSignal]:
    """
    Convenience function to rank and filter signals.
    
    Args:
        signals: Dictionary mapping symbols to signal data
        n: Number of top signals to return
        min_probability: Minimum probability threshold
        min_confidence: Minimum confidence threshold
        min_quality: Minimum quality threshold
        
    Returns:
        List of top N RankedSignal
    """
    ranker = SignalRanker()
    ranked = ranker.rank_signals(signals)
    filtered = ranker.filter_by_threshold(ranked, min_probability, min_confidence, min_quality)
    top_n = ranker.get_top_n(filtered, n)
    
    return top_n
