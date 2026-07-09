"""
Multi-Timeframe Confirmation Engine

Combines signals from multiple timeframes for confirmation.
Increases confidence when signals agree across timeframes.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger("signal_engine.multi_timeframe")


@dataclass
class TimeframeSignal:
    """Signal for a specific timeframe."""
    timeframe: str
    signal: str
    probability: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "timeframe": self.timeframe,
            "signal": self.signal,
            "probability": round(self.probability, 4),
        }


@dataclass
class MultiTimeframeResult:
    """Result of multi-timeframe analysis."""
    signals: List[TimeframeSignal]
    agreement_score: float
    dominant_signal: str
    combined_probability: float
    confidence_boost: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "signals": [s.to_dict() for s in self.signals],
            "agreement_score": round(self.agreement_score, 4),
            "dominant_signal": self.dominant_signal,
            "combined_probability": round(self.combined_probability, 4),
            "confidence_boost": round(self.confidence_boost, 4),
        }


class MultiTimeframeEngine:
    """
    Combines signals from multiple timeframes for confirmation.
    
    Timeframes:
    - 5m
    - 15m
    - 30m
    - 1h
    - 1d
    - 1w
    """
    
    def __init__(
        self,
        timeframes: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize multi-timeframe engine.
        
        Args:
            timeframes: List of timeframes to analyze
            weights: Optional weights for each timeframe
        """
        self.timeframes = timeframes or ["5m", "15m", "30m", "1h", "1d"]
        self.weights = weights or {
            "5m": 0.1,
            "15m": 0.15,
            "30m": 0.2,
            "1h": 0.25,
            "1d": 0.3,
            "1w": 0.35,
        }
        self._logger = get_logger("signal_engine.multi_timeframe")
    
    def analyze_multi_timeframe(
        self,
        timeframe_signals: Dict[str, Dict],
    ) -> MultiTimeframeResult:
        """
        Analyze signals across multiple timeframes.
        
        Args:
            timeframe_signals: Dictionary mapping timeframes to signal data
            
        Returns:
            MultiTimeframeResult
        """
        signals = []
        
        for timeframe, signal_data in timeframe_signals.items():
            signal = signal_data.get("signal", "HOLD")
            probability = signal_data.get("probability", 0.5)
            
            signals.append(TimeframeSignal(
                timeframe=timeframe,
                signal=signal,
                probability=probability,
            ))
        
        # Calculate agreement score
        agreement_score = self._calculate_agreement(signals)
        
        # Determine dominant signal
        dominant_signal = self._determine_dominant_signal(signals)
        
        # Calculate combined probability
        combined_probability = self._calculate_combined_probability(signals)
        
        # Calculate confidence boost
        confidence_boost = self._calculate_confidence_boost(agreement_score)
        
        return MultiTimeframeResult(
            signals=signals,
            agreement_score=agreement_score,
            dominant_signal=dominant_signal,
            combined_probability=combined_probability,
            confidence_boost=confidence_boost,
        )
    
    def _calculate_agreement(self, signals: List[TimeframeSignal]) -> float:
        """
        Calculate agreement across timeframes.
        
        Args:
            signals: List of TimeframeSignal
            
        Returns:
            Agreement score (0 to 1)
        """
        if not signals:
            return 0.0
        
        # Count bullish, bearish, neutral
        bullish = sum(1 for s in signals if s.signal in ["BUY", "STRONG_BUY"])
        bearish = sum(1 for s in signals if s.signal in ["SELL", "STRONG_SELL"])
        neutral = sum(1 for s in signals if s.signal == "HOLD")
        
        total = len(signals)
        
        # Agreement is the proportion of signals in the dominant category
        max_count = max(bullish, bearish, neutral)
        agreement = max_count / total if total > 0 else 0.0
        
        return agreement
    
    def _determine_dominant_signal(self, signals: List[TimeframeSignal]) -> str:
        """
        Determine dominant signal across timeframes.
        
        Args:
            signals: List of TimeframeSignal
            
        Returns:
            Dominant signal
        """
        if not signals:
            return "HOLD"
        
        # Count signals by type
        bullish = sum(1 for s in signals if s.signal in ["BUY", "STRONG_BUY"])
        bearish = sum(1 for s in signals if s.signal in ["SELL", "STRONG_SELL"])
        neutral = sum(1 for s in signals if s.signal == "HOLD")
        
        # Return dominant
        if bullish >= bearish and bullish >= neutral:
            return "BUY"
        elif bearish >= bullish and bearish >= neutral:
            return "SELL"
        else:
            return "HOLD"
    
    def _calculate_combined_probability(self, signals: List[TimeframeSignal]) -> float:
        """
        Calculate combined probability across timeframes.
        
        Args:
            signals: List of TimeframeSignal
            
        Returns:
            Combined probability
        """
        if not signals:
            return 0.5
        
        # Weighted average of probabilities
        weighted_sum = 0.0
        total_weight = 0.0
        
        for signal in signals:
            weight = self.weights.get(signal.timeframe, 1.0)
            weighted_sum += signal.probability * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        return weighted_sum / total_weight
    
    def _calculate_confidence_boost(self, agreement_score: float) -> float:
        """
        Calculate confidence boost from multi-timeframe agreement.
        
        Args:
            agreement_score: Agreement score
            
        Returns:
            Confidence boost (0 to 0.5)
        """
        # Higher agreement = higher confidence boost
        boost = agreement_score * 0.5
        return boost
    
    def get_timeframe_summary(self, result: MultiTimeframeResult) -> str:
        """
        Get human-readable summary of multi-timeframe analysis.
        
        Args:
            result: MultiTimeframeResult
            
        Returns:
            Summary string
        """
        lines = []
        
        lines.append("Multi-Timeframe Analysis:")
        lines.append("")
        
        for signal in result.signals:
            lines.append(f"{signal.timeframe}: {signal.signal} ({signal.probability:.1%})")
        
        lines.append("")
        lines.append(f"Agreement: {result.agreement_score:.1%}")
        lines.append(f"Dominant Signal: {result.dominant_signal}")
        lines.append(f"Combined Probability: {result.combined_probability:.1%}")
        lines.append(f"Confidence Boost: +{result.confidence_boost:.1%}")
        
        return "\n".join(lines)


def analyze_multi_timeframe(
    timeframe_signals: Dict[str, Dict],
    timeframes: Optional[List[str]] = None,
) -> MultiTimeframeResult:
    """
    Convenience function to analyze multi-timeframe signals.
    
    Args:
        timeframe_signals: Dictionary mapping timeframes to signal data
        timeframes: Optional list of timeframes
        
    Returns:
        MultiTimeframeResult
    """
    engine = MultiTimeframeEngine(timeframes=timeframes)
    return engine.analyze_multi_timeframe(timeframe_signals)
