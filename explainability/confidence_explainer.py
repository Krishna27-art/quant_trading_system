"""
Confidence Explainer Module

Explains the confidence score breakdown.
Shows how each component contributes to the final confidence.

Example output:
- Model Agreement: 25
- Signal Agreement: 20
- Feature Quality: 18
- Market Regime: 15
- Historical Similarity: 9
- Data Quality: 5
--------
Confidence: 92
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from utils.logger import get_logger

logger = get_logger("confidence_explainer")


@dataclass
class ConfidenceComponent:
    """Represents a component of the confidence score."""
    
    component_name: str
    score: float
    weight: float
    contribution: float  # score * weight
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component_name,
            "score": round(self.score, 2),
            "weight": round(self.weight, 3),
            "contribution": round(self.contribution, 2),
        }


class ConfidenceExplainer:
    """
    Explains how the confidence score was composed.
    
    Shows contribution from each component transparently.
    """
    
    def __init__(self):
        """Initialize the confidence explainer."""
        self.logger = logger
    
    def explain_confidence(
        self,
        confidence_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Explain confidence score composition.
        
        Args:
            confidence_data: Dict with confidence score components
            
        Returns:
            Dict with confidence explanation
        """
        self.logger.info("Explaining confidence score composition...")
        
        components = []
        total_contribution = 0
        
        for component_name, component_data in confidence_data.get('components', {}).items():
            score = component_data.get('score', 0)
            weight = component_data.get('weight', 0)
            contribution = score * weight
            
            components.append(ConfidenceComponent(
                component_name=component_name,
                score=score,
                weight=weight,
                contribution=contribution,
            ))
            
            total_contribution += contribution
        
        # Sort by contribution
        components.sort(key=lambda x: x.contribution, reverse=True)
        
        confidence_score = confidence_data.get('confidence_score', total_contribution)
        
        explanation = {
            "confidence_score": round(confidence_score, 2),
            "components": [c.to_dict() for c in components],
            "total_contribution": round(total_contribution, 2),
            "top_component": components[0].component_name if components else None,
        }
        
        self.logger.info(f"Explained confidence score: {confidence_score}")
        return explanation
    
    def format_explanation(self, explanation: dict[str, Any]) -> str:
        """
        Format confidence explanation as human-readable string.
        
        Args:
            explanation: Dict from explain_confidence
            
        Returns:
            Formatted string
        """
        lines = [f"Confidence Score: {explanation['confidence_score']}%"]
        lines.append("Breakdown:")
        
        for component in explanation['components']:
            lines.append(
                f"- {component['component']}: {component['contribution']:.1f} "
                f"(score: {component['score']:.1f}, weight: {component['weight']:.2f})"
            )
        
        lines.append("-" * 40)
        lines.append(f"Total: {explanation['total_contribution']:.1f}%")
        
        return "\n".join(lines)
    
    def get_component_breakdown(
        self,
        explanation: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Get breakdown of confidence components.
        
        Args:
            explanation: Dict from explain_confidence
            
        Returns:
            Dict with component breakdown
        """
        components = explanation['components']
        
        if not components:
            return {}
        
        contributions = [c['contribution'] for c in components]
        total = sum(contributions)
        
        return {
            "total_components": len(components),
            "top_component": explanation['top_component'],
            "top_contribution_pct": max(contributions) / total * 100 if total > 0 else 0,
            "component_distribution": {
                c['component']: c['contribution'] / total * 100 if total > 0 else 0
                for c in components
            },
        }
    
    def assess_confidence_quality(
        self,
        explanation: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Assess the quality of the confidence score.
        
        Args:
            explanation: Dict from explain_confidence
            
        Returns:
            Dict with quality assessment
        """
        components = explanation['components']
        confidence_score = explanation['confidence_score']
        
        if not components:
            return {
                "quality": "unknown",
                "reason": "No components available",
            }
        
        # Check for balanced contributions
        contributions = [c['contribution'] for c in components]
        max_contrib = max(contributions)
        avg_contrib = sum(contributions) / len(contributions)
        
        # Check if any single component dominates (>50%)
        if max_contrib > 50:
            quality = "low"
            reason = f"Single component dominates ({explanation['top_component']}: {max_contrib:.1f}%)"
        # Check if components are well-balanced
        elif max_contrib / avg_contrib < 2:
            quality = "high"
            reason = "Components are well-balanced"
        else:
            quality = "medium"
            reason = "Moderate component balance"
        
        return {
            "quality": quality,
            "reason": reason,
            "max_contribution": round(max_contrib, 2),
            "avg_contribution": round(avg_contrib, 2),
        }


def explain_confidence(
    confidence_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Convenience function to explain confidence score.
    
    Args:
        confidence_data: Dict with confidence score components
        
    Returns:
        Confidence explanation
    """
    explainer = ConfidenceExplainer()
    explanation = explainer.explain_confidence(confidence_data)
    
    return {
        "explanation": explanation,
        "formatted": explainer.format_explanation(explanation),
        "breakdown": explainer.get_component_breakdown(explanation),
        "quality": explainer.assess_confidence_quality(explanation),
    }
