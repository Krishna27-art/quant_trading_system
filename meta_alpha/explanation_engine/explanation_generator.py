"""
Explanation Generator

Generates human-readable explanations for recommendations.
Explains every recommendation in plain language with evidence contributions.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional

from meta_alpha.evidence_engine.evidence import Evidence
from meta_alpha.recommendation_engine.recommendation import Recommendation
from utils.logger import get_logger

logger = get_logger("meta_alpha.explanation_engine")


@dataclass
class EvidenceContribution:
    """Contribution of a single evidence piece."""
    source: str
    factor_name: str
    direction: str
    contribution_score: float
    reason: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "source": self.source,
            "factor_name": self.factor_name,
            "direction": self.direction,
            "contribution_score": round(self.contribution_score, 4),
            "reason": self.reason,
        }


@dataclass
class Explanation:
    """Full explanation for a recommendation."""
    action: str
    summary: str
    evidence_contributions: List[EvidenceContribution]
    key_drivers: List[str]
    risk_factors: List[str]
    confidence_rationale: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "action": self.action,
            "summary": self.summary,
            "evidence_contributions": [c.to_dict() for c in self.evidence_contributions],
            "key_drivers": self.key_drivers,
            "risk_factors": self.risk_factors,
            "confidence_rationale": self.confidence_rationale,
        }


class ExplanationGenerator:
    """
    Generates human-readable explanations for recommendations.
    
    Explains:
    - Why the recommendation was made
    - Evidence contributions
    - Key drivers
    - Risk factors
    - Confidence rationale
    """
    
    def __init__(self):
        """Initialize explanation generator."""
        self._logger = get_logger("meta_alpha.explanation_engine")
    
    def generate_explanation(
        self,
        recommendation: Recommendation,
        evidence_list: List[Evidence],
        weights: Optional[Dict[str, float]] = None,
    ) -> Explanation:
        """
        Generate explanation for recommendation.
        
        Args:
            recommendation: Recommendation to explain
            evidence_list: List of Evidence
            weights: Optional weights for evidence
            
        Returns:
            Explanation
        """
        # Calculate evidence contributions
        contributions = self._calculate_contributions(evidence_list, weights)
        
        # Generate summary
        summary = self._generate_summary(recommendation, contributions)
        
        # Identify key drivers
        key_drivers = self._identify_key_drivers(contributions)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(contributions, recommendation)
        
        # Generate confidence rationale
        confidence_rationale = self._generate_confidence_rationale(
            recommendation,
            contributions,
        )
        
        return Explanation(
            action=recommendation.action,
            summary=summary,
            evidence_contributions=contributions,
            key_drivers=key_drivers,
            risk_factors=risk_factors,
            confidence_rationale=confidence_rationale,
        )
    
    def _calculate_contributions(
        self,
        evidence_list: List[Evidence],
        weights: Optional[Dict[str, float]],
    ) -> List[EvidenceContribution]:
        """
        Calculate evidence contributions.
        
        Args:
            evidence_list: List of Evidence
            weights: Optional weights
            
        Returns:
            List of EvidenceContribution
        """
        contributions = []
        
        for evidence in evidence_list:
            evidence_id = f"{evidence.source}_{evidence.factor_name}"
            weight = weights.get(evidence_id, 1.0) if weights else 1.0
            
            # Calculate contribution score
            contribution_score = evidence.get_bullish_score() * weight
            
            # Generate reason
            reason = self._generate_evidence_reason(evidence)
            
            contributions.append(EvidenceContribution(
                source=evidence.source,
                factor_name=evidence.factor_name,
                direction=evidence.signal_direction,
                contribution_score=contribution_score,
                reason=reason,
            ))
        
        # Sort by contribution score
        contributions.sort(key=lambda x: abs(x.contribution_score), reverse=True)
        
        return contributions
    
    def _generate_evidence_reason(self, evidence: Evidence) -> str:
        """
        Generate reason for a single evidence piece.
        
        Args:
            evidence: Evidence
            
        Returns:
            Reason string
        """
        direction = evidence.signal_direction.upper()
        strength = evidence.strength
        confidence = evidence.confidence
        
        return f"{direction} signal with strength {strength:.2f} and confidence {confidence:.2f}"
    
    def _generate_summary(
        self,
        recommendation: Recommendation,
        contributions: List[EvidenceContribution],
    ) -> str:
        """
        Generate summary explanation.
        
        Args:
            recommendation: Recommendation
            contributions: Evidence contributions
            
        Returns:
            Summary string
        """
        parts = []
        
        parts.append(f"Recommendation: {recommendation.action}")
        parts.append(f"Probability: {recommendation.probability:.1%}")
        parts.append(f"Confidence: {recommendation.confidence}")
        parts.append(f"Expected Return: {recommendation.expected_return:.2%}")
        
        # Add top contributors
        if contributions:
            top_contributors = contributions[:3]
            parts.append("Key Evidence:")
            for contrib in top_contributors:
                sign = "+" if contrib.contribution_score > 0 else ""
                parts.append(f"  {contrib.source}: {sign}{contrib.contribution_score:.3f}")
        
        return " | ".join(parts)
    
    def _identify_key_drivers(
        self,
        contributions: List[EvidenceContribution],
    ) -> List[str]:
        """
        Identify key drivers of the recommendation.
        
        Args:
            contributions: Evidence contributions
            
        Returns:
            List of key driver descriptions
        """
        drivers = []
        
        # Top positive contributors
        positive = [c for c in contributions if c.contribution_score > 0]
        positive.sort(key=lambda x: x.contribution_score, reverse=True)
        
        for contrib in positive[:3]:
            drivers.append(f"{contrib.source} ({contrib.direction})")
        
        return drivers
    
    def _identify_risk_factors(
        self,
        contributions: List[EvidenceContribution],
        recommendation: Recommendation,
    ) -> List[str]:
        """
        Identify risk factors.
        
        Args:
            contributions: Evidence contributions
            recommendation: Recommendation
            
        Returns:
            List of risk factor descriptions
        """
        risks = []
        
        # Negative contributors
        negative = [c for c in contributions if c.contribution_score < 0]
        
        for contrib in negative:
            risks.append(f"{contrib.source} ({contrib.direction})")
        
        # Add risk level
        if recommendation.risk_level == "HIGH":
            risks.append("High market risk")
        elif recommendation.risk_level == "MEDIUM":
            risks.append("Moderate market risk")
        
        return risks
    
    def _generate_confidence_rationale(
        self,
        recommendation: Recommendation,
        contributions: List[EvidenceContribution],
    ) -> str:
        """
        Generate confidence rationale.
        
        Args:
            recommendation: Recommendation
            contributions: Evidence contributions
            
        Returns:
            Confidence rationale string
        """
        parts = []
        
        # Confidence level
        parts.append(f"Confidence is {recommendation.confidence}")
        
        # Evidence agreement
        if contributions:
            positive = sum(1 for c in contributions if c.contribution_score > 0)
            negative = sum(1 for c in contributions if c.contribution_score < 0)
            total = len(contributions)
            
            if positive > negative:
                parts.append(f"based on {positive} positive vs {negative} negative evidence")
            elif negative > positive:
                parts.append(f"based on {negative} negative vs {positive} positive evidence")
            else:
                parts.append("with mixed evidence")
        
        return " ".join(parts)
    
    def generate_plain_text(self, explanation: Explanation) -> str:
        """
        Generate plain text explanation.
        
        Args:
            explanation: Explanation
            
        Returns:
            Plain text string
        """
        lines = []
        
        lines.append(f"RECOMMENDATION: {explanation.action}")
        lines.append("")
        lines.append("SUMMARY")
        lines.append(explanation.summary)
        lines.append("")
        lines.append("EVIDENCE CONTRIBUTIONS")
        
        for contrib in explanation.evidence_contributions:
            sign = "+" if contrib.contribution_score > 0 else ""
            lines.append(f"  {contrib.source}: {sign}{contrib.contribution_score:.3f} - {contrib.reason}")
        
        lines.append("")
        lines.append("KEY DRIVERS")
        for driver in explanation.key_drivers:
            lines.append(f"  - {driver}")
        
        lines.append("")
        lines.append("RISK FACTORS")
        for risk in explanation.risk_factors:
            lines.append(f"  - {risk}")
        
        lines.append("")
        lines.append("CONFIDENCE RATIONALE")
        lines.append(f"  {explanation.confidence_rationale}")
        
        return "\n".join(lines)


def generate_explanation(
    recommendation: Recommendation,
    evidence_list: List[Evidence],
) -> Explanation:
    """
    Convenience function to generate explanation.
    
    Args:
        recommendation: Recommendation
        evidence_list: List of Evidence
        
    Returns:
        Explanation
    """
    generator = ExplanationGenerator()
    return generator.generate_explanation(recommendation, evidence_list)
