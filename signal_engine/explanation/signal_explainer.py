"""
Signal Explainer Engine

Generates structured explanations for trading signals.
Provides positive and negative factor contributions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger("signal_engine.explanation")


@dataclass
class FactorContribution:
    """Contribution of a single factor to the signal."""
    factor_name: str
    category: str
    contribution: float
    direction: str  # "positive" or "negative"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "factor_name": self.factor_name,
            "category": self.category,
            "contribution": round(self.contribution, 4),
            "direction": self.direction,
        }


@dataclass
class SignalExplanation:
    """Structured explanation for a trading signal."""
    signal: str
    probability: float
    positive_contributors: List[FactorContribution]
    negative_contributors: List[FactorContribution]
    summary: str
    key_factors: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "signal": self.signal,
            "probability": round(self.probability, 4),
            "positive_contributors": [c.to_dict() for c in self.positive_contributors],
            "negative_contributors": [c.to_dict() for c in self.negative_contributors],
            "summary": self.summary,
            "key_factors": self.key_factors,
        }


class SignalExplainer:
    """
    Generates structured explanations for trading signals.
    
    Provides:
    - Positive factor contributions
    - Negative factor contributions
    - Summary explanation
    - Key driving factors
    """
    
    def __init__(self):
        """Initialize signal explainer."""
        self._logger = get_logger("signal_engine.explanation")
    
    def explain_signal(
        self,
        factor_scores: Dict[str, float],
        factor_categories: Dict[str, str],
        probability: float,
    ) -> SignalExplanation:
        """
        Generate explanation for a trading signal.
        
        Args:
            factor_scores: Dictionary mapping factor names to scores
            factor_categories: Dictionary mapping factor names to categories
            probability: Signal probability
            
        Returns:
            SignalExplanation
        """
        # Determine signal direction
        signal = self._classify_signal(probability)
        
        # Separate positive and negative contributors
        positive_contributors = []
        negative_contributors = []
        
        for factor_name, score in factor_scores.items():
            category = factor_categories.get(factor_name, "unknown")
            
            if score > 0.1:
                positive_contributors.append(FactorContribution(
                    factor_name=factor_name,
                    category=category,
                    contribution=score,
                    direction="positive",
                ))
            elif score < -0.1:
                negative_contributors.append(FactorContribution(
                    factor_name=factor_name,
                    category=category,
                    contribution=abs(score),
                    direction="negative",
                ))
        
        # Sort by contribution magnitude
        positive_contributors.sort(key=lambda x: x.contribution, reverse=True)
        negative_contributors.sort(key=lambda x: x.contribution, reverse=True)
        
        # Generate summary
        summary = self._generate_summary(
            signal,
            probability,
            positive_contributors,
            negative_contributors,
        )
        
        # Identify key factors
        key_factors = self._identify_key_factors(
            positive_contributors,
            negative_contributors,
        )
        
        return SignalExplanation(
            signal=signal,
            probability=probability,
            positive_contributors=positive_contributors,
            negative_contributors=negative_contributors,
            summary=summary,
            key_factors=key_factors,
        )
    
    def _classify_signal(self, probability: float) -> str:
        """
        Classify signal based on probability.
        
        Args:
            probability: Signal probability
            
        Returns:
            Signal classification
        """
        if probability >= 0.8:
            return "STRONG_BUY"
        elif probability >= 0.6:
            return "BUY"
        elif probability <= 0.2:
            return "STRONG_SELL"
        elif probability <= 0.4:
            return "SELL"
        else:
            return "HOLD"
    
    def _generate_summary(
        self,
        signal: str,
        probability: float,
        positive_contributors: List[FactorContribution],
        negative_contributors: List[FactorContribution],
    ) -> str:
        """
        Generate human-readable summary.
        
        Args:
            signal: Signal classification
            probability: Signal probability
            positive_contributors: List of positive contributors
            negative_contributors: List of negative contributors
            
        Returns:
            Summary string
        """
        parts = []
        
        parts.append(f"{signal} signal with {probability:.1%} probability.")
        
        if positive_contributors:
            top_positive = positive_contributors[:3]
            positive_names = [c.factor_name for c in top_positive]
            parts.append(f"Supported by: {', '.join(positive_names)}.")
        
        if negative_contributors:
            top_negative = negative_contributors[:3]
            negative_names = [c.factor_name for c in top_negative]
            parts.append(f"Weakened by: {', '.join(negative_names)}.")
        
        return " ".join(parts)
    
    def _identify_key_factors(
        self,
        positive_contributors: List[FactorContribution],
        negative_contributors: List[FactorContribution],
        top_n: int = 5,
    ) -> List[str]:
        """
        Identify key driving factors.
        
        Args:
            positive_contributors: List of positive contributors
            negative_contributors: List of negative contributors
            top_n: Number of top factors to return
            
        Returns:
            List of key factor names
        """
        all_contributors = positive_contributors + negative_contributors
        all_contributors.sort(key=lambda x: x.contribution, reverse=True)
        
        key_factors = [c.factor_name for c in all_contributors[:top_n]]
        
        return key_factors
    
    def format_for_display(self, explanation: SignalExplanation) -> str:
        """
        Format explanation for display.
        
        Args:
            explanation: SignalExplanation
            
        Returns:
            Formatted string
        """
        lines = []
        
        lines.append(f"Signal: {explanation.signal}")
        lines.append(f"Probability: {explanation.probability:.1%}")
        lines.append("")
        
        if explanation.positive_contributors:
            lines.append("Positive Contributors:")
            for contributor in explanation.positive_contributors:
                lines.append(f"  +{contributor.contributor:.2f} {contributor.factor_name} ({contributor.category})")
            lines.append("")
        
        if explanation.negative_contributors:
            lines.append("Negative Contributors:")
            for contributor in explanation.negative_contributors:
                lines.append(f"  -{contributor.contributor:.2f} {contributor.factor_name} ({contributor.category})")
            lines.append("")
        
        lines.append(f"Key Factors: {', '.join(explanation.key_factors)}")
        lines.append("")
        lines.append(f"Summary: {explanation.summary}")
        
        return "\n".join(lines)


def explain_signal(
    factor_scores: Dict[str, float],
    factor_categories: Dict[str, str],
    probability: float,
) -> SignalExplanation:
    """
    Convenience function to explain a signal.
    
    Args:
        factor_scores: Dictionary mapping factor names to scores
        factor_categories: Dictionary mapping factor names to categories
        probability: Signal probability
        
    Returns:
        SignalExplanation
    """
    explainer = SignalExplainer()
    return explainer.explain_signal(factor_scores, factor_categories, probability)
