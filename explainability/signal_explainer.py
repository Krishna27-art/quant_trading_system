"""
Signal Explainer Module

Explains which signals fired for a prediction with ratings.
Shows signal strength using star ratings.

Example output:
- Trend: ★★★★★
- Volume: ★★★★★
- Options: ★★★★☆
- Fundamental: ★★★★☆
- Macro: ★★★☆☆
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from utils.logger import get_logger

logger = get_logger("signal_explainer")


@dataclass
class SignalRating:
    """Represents a signal with its rating."""
    
    signal_name: str
    signal_type: str  # trend, volume, options, fundamental, macro, sentiment
    strength: float  # 0-100
    fired: bool
    contribution: str  # "bullish", "bearish", or "neutral"
    
    @property
    def star_rating(self) -> str:
        """Convert strength to star rating."""
        if self.strength >= 90:
            return "★★★★★"
        elif self.strength >= 70:
            return "★★★★☆"
        elif self.strength >= 50:
            return "★★★☆☆"
        elif self.strength >= 30:
            return "★★☆☆☆"
        else:
            return "★☆☆☆☆"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal_name,
            "type": self.signal_type,
            "strength": round(self.strength, 1),
            "fired": self.fired,
            "contribution": self.contribution,
            "star_rating": self.star_rating,
        }


class SignalExplainer:
    """
    Explains which signals fired for a prediction.
    
    Shows signal strength using intuitive star ratings.
    """
    
    def __init__(self):
        """Initialize the signal explainer."""
        self.logger = logger
    
    def explain_signals(
        self,
        signals: dict[str, Any],
    ) -> list[SignalRating]:
        """
        Explain which signals fired.
        
        Args:
            signals: Dict of signal names to signal data
            
        Returns:
            List of SignalRating objects
        """
        self.logger.info("Explaining signals...")
        
        ratings = []
        
        for signal_name, signal_data in signals.items():
            # Extract signal information
            strength = signal_data.get('strength', 0)
            fired = signal_data.get('fired', False)
            contribution = signal_data.get('contribution', 'neutral')
            signal_type = signal_data.get('type', 'unknown')
            
            ratings.append(SignalRating(
                signal_name=signal_name,
                signal_type=signal_type,
                strength=strength,
                fired=fired,
                contribution=contribution,
            ))
        
        # Sort by strength
        ratings.sort(key=lambda x: x.strength, reverse=True)
        
        self.logger.info(f"Explained {len(ratings)} signals")
        return ratings
    
    def explain_prediction_signals(
        self,
        prediction_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Explain signals for a specific prediction.
        
        Args:
            prediction_data: Dict with prediction signals
            
        Returns:
            Dict with signal explanation
        """
        signals = prediction_data.get('signals', {})
        
        ratings = self.explain_signals(signals)
        
        # Group by signal type
        grouped = {}
        for rating in ratings:
            if rating.signal_type not in grouped:
                grouped[rating.signal_type] = []
            grouped[rating.signal_type].append(rating)
        
        return {
            "symbol": prediction_data.get('symbol'),
            "prediction": prediction_data.get('prediction'),
            "signals": [r.to_dict() for r in ratings],
            "grouped_signals": {
                k: [r.to_dict() for r in v] for k, v in grouped.items()
            },
            "total_signals": len(ratings),
            "fired_signals": sum(1 for r in ratings if r.fired),
        }
    
    def format_explanation(self, ratings: list[SignalRating]) -> str:
        """
        Format signal explanation as human-readable string.
        
        Args:
            ratings: List of SignalRating objects
            
        Returns:
            Formatted string
        """
        lines = ["Signal Analysis:"]
        
        for rating in ratings:
            status = "FIRED" if rating.fired else "NOT FIRED"
            contribution_symbol = "↑" if rating.contribution == "bullish" else ("↓" if rating.contribution == "bearish" else "→")
            lines.append(
                f"{rating.signal_name}: {rating.star_rating} ({status}) "
                f"{contribution_symbol} {rating.strength:.1f}"
            )
        
        return "\n".join(lines)
    
    def get_signal_summary(
        self,
        ratings: list[SignalRating],
    ) -> dict[str, Any]:
        """
        Get summary of signals.
        
        Args:
            ratings: List of SignalRating objects
            
        Returns:
            Dict with signal summary
        """
        fired = [r for r in ratings if r.fired]
        bullish = [r for r in fired if r.contribution == "bullish"]
        bearish = [r for r in fired if r.contribution == "bearish"]
        
        avg_strength = sum(r.strength for r in fired) / len(fired) if fired else 0
        
        return {
            "total_signals": len(ratings),
            "fired_signals": len(fired),
            "bullish_signals": len(bullish),
            "bearish_signals": len(bearish),
            "avg_strength": round(avg_strength, 1),
            "strongest_signal": fired[0].signal_name if fired else None,
        }


def explain_signals(
    signals: dict[str, Any],
) -> dict[str, Any]:
    """
    Convenience function to explain signals.
    
    Args:
        signals: Dict of signal names to signal data
        
    Returns:
        Signal explanation
    """
    explainer = SignalExplainer()
    ratings = explainer.explain_signals(signals)
    
    return {
        "signals": [r.to_dict() for r in ratings],
        "formatted": explainer.format_explanation(ratings),
        "summary": explainer.get_signal_summary(ratings),
    }
