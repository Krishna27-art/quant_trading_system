"""
Signal Ranking Engine

Ranks predictions by quality score and probability.
Returns top N signals for execution consideration.
"""

from typing import List, Optional
from dataclasses import dataclass
from .signal_generator import TradingSignal, SignalDirection


@dataclass
class RankedSignal:
    """Ranked signal with position."""
    signal: TradingSignal
    rank: int
    score: float


class SignalRankingEngine:
    """
    Ranks trading signals by multiple criteria.
    
    Ranking criteria:
    - Overall quality score
    - Probability/Confidence
    - Expected return
    - Risk-reward ratio
    - Liquidity score
    """
    
    def __init__(self, min_score: float = 60.0, min_probability: float = 0.60):
        self.min_score = min_score
        self.min_probability = min_probability
    
    def rank_signals(
        self,
        signals: List[TradingSignal],
        top_n: Optional[int] = None
    ) -> List[RankedSignal]:
        """
        Rank signals and return top N.
        
        Args:
            signals: List of trading signals
            top_n: Number of top signals to return (None = all above threshold)
            
        Returns:
            List of ranked signals
        """
        # Filter by minimum thresholds
        filtered = [
            s for s in signals
            if s.overall_score >= self.min_score
            and s.probability >= self.min_probability
            and s.direction != SignalDirection.HOLD
        ]
        
        if not filtered:
            return []
        
        # Calculate composite score
        scored = []
        for signal in filtered:
            composite_score = self._calculate_composite_score(signal)
            scored.append((signal, composite_score))
        
        # Sort by composite score (descending)
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Apply top_n limit
        if top_n:
            scored = scored[:top_n]
        
        # Create ranked signals
        ranked = [
            RankedSignal(signal=signal, rank=i+1, score=score)
            for i, (signal, score) in enumerate(scored)
        ]
        
        return ranked
    
    def _calculate_composite_score(self, signal: TradingSignal) -> float:
        """
        Calculate composite ranking score.
        
        Weights:
        - Overall quality score: 40%
        - Probability: 25%
        - Expected return: 15%
        - Risk-reward (target/stop): 10%
        - Liquidity score: 10%
        """
        # Risk-reward ratio
        risk = abs(signal.entry_price - signal.stop_loss)
        reward = abs(signal.target_1 - signal.entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        # Normalize components
        quality_norm = signal.overall_score / 100
        prob_norm = signal.probability
        return_norm = min(signal.expected_return_pct / 10, 1)  # Cap at 10%
        rr_norm = min(rr_ratio / 3, 1)  # Cap at 3:1
        liquidity_norm = signal.liquidity_score / 100
        
        # Weighted composite
        composite = (
            quality_norm * 0.40 +
            prob_norm * 0.25 +
            return_norm * 0.15 +
            rr_norm * 0.10 +
            liquidity_norm * 0.10
        )
        
        return composite * 100  # Scale to 0-100
    
    def get_top_buy_signals(
        self,
        signals: List[TradingSignal],
        top_n: int = 10
    ) -> List[RankedSignal]:
        """Get top N BUY signals."""
        buy_signals = [s for s in signals if s.direction == SignalDirection.BUY]
        return self.rank_signals(buy_signals, top_n)
    
    def get_top_sell_signals(
        self,
        signals: List[TradingSignal],
        top_n: int = 10
    ) -> List[RankedSignal]:
        """Get top N SELL signals."""
        sell_signals = [s for s in signals if s.direction == SignalDirection.SELL]
        return self.rank_signals(sell_signals, top_n)
