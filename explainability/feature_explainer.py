"""
Feature Explainer Module

Explains which features contributed most to a prediction.
Shows top contributing features instead of all features.

Example output:
- Relative Strength: +12%
- Volume Spike: +8%
- Delivery %: +6%
- PCR: +4%
- ATR: -2%
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from utils.logger import get_logger

logger = get_logger("feature_explainer")


@dataclass
class FeatureContribution:
    """Represents a single feature's contribution."""
    
    feature_name: str
    contribution: float  # Positive or negative
    importance: float  # 0-1 scale
    direction: str  # "bullish" or "bearish"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature_name,
            "contribution": round(self.contribution, 2),
            "importance": round(self.importance, 3),
            "direction": self.direction,
        }


class FeatureExplainer:
    """
    Explains feature contributions for predictions.
    
    Shows top features that drove the prediction decision.
    """
    
    def __init__(self, top_n: int = 5):
        """
        Initialize the feature explainer.
        
        Args:
            top_n: Number of top features to show
        """
        self.top_n = top_n
        self.logger = logger
    
    def explain_features(
        self,
        features: dict[str, float],
        feature_importance: dict[str, float] | None = None,
    ) -> list[FeatureContribution]:
        """
        Explain feature contributions.
        
        Args:
            features: Dict of feature names to values
            feature_importance: Optional dict of feature importance scores
            
        Returns:
            List of FeatureContribution objects
        """
        self.logger.info("Explaining feature contributions...")
        
        contributions = []
        
        for feature_name, feature_value in features.items():
            # Normalize contribution based on feature value
            if feature_importance:
                importance = feature_importance.get(feature_name, 0)
            else:
                importance = abs(feature_value) / max(abs(v) for v in features.values()) if features else 0
            
            # Determine direction
            direction = "bullish" if feature_value > 0 else "bearish"
            
            # Calculate contribution percentage
            contribution = feature_value * 100  # Scale to percentage
            
            contributions.append(FeatureContribution(
                feature_name=feature_name,
                contribution=contribution,
                importance=importance,
                direction=direction,
            ))
        
        # Sort by importance and take top N
        contributions.sort(key=lambda x: x.importance, reverse=True)
        top_contributions = contributions[:self.top_n]
        
        self.logger.info(f"Explained {len(top_contributions)} top features")
        return top_contributions
    
    def explain_prediction_features(
        self,
        prediction_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Explain features for a specific prediction.
        
        Args:
            prediction_data: Dict with prediction features and metadata
            
        Returns:
            Dict with feature explanation
        """
        features = prediction_data.get('features', {})
        feature_importance = prediction_data.get('feature_importance')
        
        contributions = self.explain_features(features, feature_importance)
        
        return {
            "symbol": prediction_data.get('symbol'),
            "prediction": prediction_data.get('prediction'),
            "top_features": [c.to_dict() for c in contributions],
            "total_features": len(features),
        }
    
    def format_explanation(self, contributions: list[FeatureContribution]) -> str:
        """
        Format feature explanation as human-readable string.
        
        Args:
            contributions: List of FeatureContribution objects
            
        Returns:
            Formatted string
        """
        lines = ["Top Features:"]
        for i, contrib in enumerate(contributions, 1):
            direction_symbol = "↑" if contrib.direction == "bullish" else "↓"
            lines.append(
                f"{i}. {contrib.feature_name}: {direction_symbol} {abs(contrib.contribution):.1f}% "
                f"(importance: {contrib.importance:.2f})"
            )
        return "\n".join(lines)


def explain_features(
    features: dict[str, float],
    feature_importance: dict[str, float] | None = None,
    top_n: int = 5,
) -> dict[str, Any]:
    """
    Convenience function to explain feature contributions.
    
    Args:
        features: Dict of feature names to values
        feature_importance: Optional feature importance scores
        top_n: Number of top features to show
        
    Returns:
        Feature explanation
    """
    explainer = FeatureExplainer(top_n=top_n)
    contributions = explainer.explain_features(features, feature_importance)
    
    return {
        "top_features": [c.to_dict() for c in contributions],
        "formatted": explainer.format_explanation(contributions),
    }
