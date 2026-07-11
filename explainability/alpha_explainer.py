"""
Alpha Explainer Module

Explains how the Alpha Score was composed.
Shows contribution from each signal type.

Example output:
- Trend: 26
- Volume: 18
- Options: 15
- Fundamental: 13
- Sector: 10
- Macro: 5
- Sentiment: 4
------------
Alpha: 91
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from utils.logger import get_logger

logger = get_logger("alpha_explainer")


@dataclass
class AlphaComponent:
    """Represents a component of the alpha score."""
    
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


class AlphaExplainer:
    """
    Explains how the Alpha Score was composed.
    
    Shows contribution from each signal type transparently.
    """
    
    def __init__(self):
        """Initialize the alpha explainer."""
        self.logger = logger
    
    def explain_alpha(
        self,
        alpha_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Explain alpha score composition.
        
        Args:
            alpha_data: Dict with alpha score components
            
        Returns:
            Dict with alpha explanation
        """
        self.logger.info("Explaining alpha score composition...")
        
        components = []
        total_contribution = 0
        
        for component_name, component_data in alpha_data.get('components', {}).items():
            score = component_data.get('score', 0)
            weight = component_data.get('weight', 0)
            contribution = score * weight
            
            components.append(AlphaComponent(
                component_name=component_name,
                score=score,
                weight=weight,
                contribution=contribution,
            ))
            
            total_contribution += contribution
        
        # Sort by contribution
        components.sort(key=lambda x: x.contribution, reverse=True)
        
        alpha_score = alpha_data.get('alpha_score', total_contribution)
        
        explanation = {
            "alpha_score": round(alpha_score, 2),
            "components": [c.to_dict() for c in components],
            "total_contribution": round(total_contribution, 2),
            "top_component": components[0].component_name if components else None,
        }
        
        self.logger.info(f"Explained alpha score: {alpha_score}")
        return explanation
    
    def format_explanation(self, explanation: dict[str, Any]) -> str:
        """
        Format alpha explanation as human-readable string.
        
        Args:
            explanation: Dict from explain_alpha
            
        Returns:
            Formatted string
        """
        lines = [f"Alpha Score: {explanation['alpha_score']}"]
        lines.append("Composition:")
        
        for component in explanation['components']:
            lines.append(
                f"- {component['component']}: {component['contribution']:.1f} "
                f"(score: {component['score']:.1f}, weight: {component['weight']:.2f})"
            )
        
        lines.append("-" * 40)
        lines.append(f"Total: {explanation['total_contribution']:.1f}")
        
        return "\n".join(lines)
    
    def get_component_breakdown(
        self,
        explanation: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Get breakdown of alpha components.
        
        Args:
            explanation: Dict from explain_alpha
            
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


def explain_alpha(
    alpha_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Convenience function to explain alpha score.
    
    Args:
        alpha_data: Dict with alpha score components
        
    Returns:
        Alpha explanation
    """
    explainer = AlphaExplainer()
    explanation = explainer.explain_alpha(alpha_data)
    
    return {
        "explanation": explanation,
        "formatted": explainer.format_explanation(explanation),
        "breakdown": explainer.get_component_breakdown(explanation),
    }
