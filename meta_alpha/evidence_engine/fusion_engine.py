"""
Fusion Engine

Combines evidence from multiple sources into a unified view.
Does NOT assume independence - accounts for correlation and information content.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

from meta_alpha.evidence_engine.evidence import Evidence, SignalDirection
from utils.logger import get_logger

logger = get_logger("meta_alpha.evidence_engine")


@dataclass
class FusionResult:
    """Result of evidence fusion."""
    bullish_score: float
    bearish_score: float
    neutral_score: float
    missing_evidence: List[str]
    evidence_count: int
    agreement_score: float
    dominant_direction: str
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate fusion result.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check scores sum to approximately 1.0
        total = self.bullish_score + self.bearish_score + self.neutral_score
        if abs(total - 1.0) > 0.01:
            errors.append(f"Scores must sum to 1.0, got {total}")
        
        # Check all scores are between 0 and 1
        if not (0.0 <= self.bullish_score <= 1.0):
            errors.append(f"Bullish score must be between 0 and 1, got {self.bullish_score}")
        if not (0.0 <= self.bearish_score <= 1.0):
            errors.append(f"Bearish score must be between 0 and 1, got {self.bearish_score}")
        if not (0.0 <= self.neutral_score <= 1.0):
            errors.append(f"Neutral score must be between 0 and 1, got {self.neutral_score}")
        
        # Check agreement score
        if not (0.0 <= self.agreement_score <= 1.0):
            errors.append(f"Agreement score must be between 0 and 1, got {self.agreement_score}")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "bullish_score": round(self.bullish_score, 4),
            "bearish_score": round(self.bearish_score, 4),
            "neutral_score": round(self.neutral_score, 4),
            "missing_evidence": self.missing_evidence,
            "evidence_count": self.evidence_count,
            "agreement_score": round(self.agreement_score, 4),
            "dominant_direction": self.dominant_direction,
        }
    
    def get_net_score(self) -> float:
        """
        Get net score (bullish - bearish).
        
        Returns:
            Net score (-1 to 1)
        """
        return self.bullish_score - self.bearish_score


class FusionEngine:
    """
    Combines evidence from multiple sources into a unified view.
    
    Key principles:
    - Does NOT assume independence
    - Accounts for correlation between evidence
    - Weights evidence by quality and confidence
    - Handles missing evidence gracefully
    """
    
    def __init__(
        self,
        correlation_penalty: float = 0.3,
        min_evidence_threshold: int = 3,
    ):
        """
        Initialize fusion engine.
        
        Args:
            correlation_penalty: Penalty for correlated evidence
            min_evidence_threshold: Minimum evidence required for fusion
        """
        self.correlation_penalty = correlation_penalty
        self.min_evidence_threshold = min_evidence_threshold
        self._logger = get_logger("meta_alpha.evidence_engine")
    
    def fuse(
        self,
        evidence_list: List[Evidence],
        weights: Optional[Dict[str, float]] = None,
        correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> FusionResult:
        """
        Fuse evidence into unified view.
        
        Args:
            evidence_list: List of Evidence objects
            weights: Optional weights for each evidence
            correlation_matrix: Optional correlation matrix between evidence
            
        Returns:
            FusionResult
        """
        if not evidence_list:
            return FusionResult(
                bullish_score=0.0,
                bearish_score=0.0,
                neutral_score=1.0,
                missing_evidence=[],
                evidence_count=0,
                agreement_score=0.0,
                dominant_direction=SignalDirection.NEUTRAL.value,
            )
        
        # Check minimum evidence threshold
        if len(evidence_list) < self.min_evidence_threshold:
            self._logger.warning(f"Insufficient evidence: {len(evidence_list)} < {self.min_evidence_threshold}")
        
        # Calculate weighted scores
        bullish_score = 0.0
        bearish_score = 0.0
        neutral_score = 0.0
        
        # Track evidence by direction
        bullish_evidence = []
        bearish_evidence = []
        neutral_evidence = []
        
        for evidence in evidence_list:
            evidence_id = f"{evidence.source}_{evidence.factor_name}"
            weight = weights.get(evidence_id, 1.0 / len(evidence_list)) if weights else 1.0 / len(evidence_list)
            
            # Apply correlation penalty if available
            if correlation_matrix:
                weight = self._apply_correlation_penalty(
                    evidence_id,
                    weight,
                    evidence_list,
                    correlation_matrix,
                )
            
            # Calculate contribution
            contribution = evidence.get_bullish_score() * weight
            
            if evidence.is_bullish():
                bullish_score += contribution
                bullish_evidence.append(evidence_id)
            elif evidence.is_bearish():
                bearish_score += abs(contribution)
                bearish_evidence.append(evidence_id)
            else:
                neutral_score += weight
                neutral_evidence.append(evidence_id)
        
        # Normalize scores
        total = bullish_score + bearish_score + neutral_score
        if total > 0:
            bullish_score /= total
            bearish_score /= total
            neutral_score /= total
        
        # Calculate agreement score
        agreement_score = self._calculate_agreement_score(
            bullish_evidence,
            bearish_evidence,
            neutral_evidence,
        )
        
        # Determine dominant direction
        dominant_direction = self._determine_dominant_direction(
            bullish_score,
            bearish_score,
            neutral_score,
        )
        
        # Identify missing evidence (categories not represented)
        missing_evidence = self._identify_missing_evidence(evidence_list)
        
        return FusionResult(
            bullish_score=bullish_score,
            bearish_score=bearish_score,
            neutral_score=neutral_score,
            missing_evidence=missing_evidence,
            evidence_count=len(evidence_list),
            agreement_score=agreement_score,
            dominant_direction=dominant_direction,
        )
    
    def _apply_correlation_penalty(
        self,
        evidence_id: str,
        weight: float,
        evidence_list: List[Evidence],
        correlation_matrix: Dict[str, Dict[str, float]],
    ) -> float:
        """
        Apply correlation penalty to weight.
        
        Args:
            evidence_id: Evidence identifier
            weight: Original weight
            evidence_list: List of all evidence
            correlation_matrix: Correlation matrix
            
        Returns:
            Adjusted weight
        """
        if evidence_id not in correlation_matrix:
            return weight
        
        # Calculate average correlation with other evidence
        correlations = []
        
        for other_evidence in evidence_list:
            other_id = f"{other_evidence.source}_{other_evidence.factor_name}"
            if other_id == evidence_id:
                continue
            
            correlation = correlation_matrix[evidence_id].get(other_id, 0.0)
            correlations.append(abs(correlation))
        
        if not correlations:
            return weight
        
        avg_correlation = sum(correlations) / len(correlations)
        
        # Apply penalty
        penalty = avg_correlation * self.correlation_penalty
        adjusted_weight = weight * (1.0 - penalty)
        
        return max(0.0, adjusted_weight)
    
    def _calculate_agreement_score(
        self,
        bullish_evidence: List[str],
        bearish_evidence: List[str],
        neutral_evidence: List[str],
    ) -> float:
        """
        Calculate agreement score among evidence.
        
        Args:
            bullish_evidence: List of bullish evidence IDs
            bearish_evidence: List of bearish evidence IDs
            neutral_evidence: List of neutral evidence IDs
            
        Returns:
            Agreement score (0-1)
        """
        total = len(bullish_evidence) + len(bearish_evidence) + len(neutral_evidence)
        
        if total == 0:
            return 0.0
        
        # Find maximum count
        max_count = max(len(bullish_evidence), len(bearish_evidence), len(neutral_evidence))
        
        # Agreement is the proportion of evidence in the dominant direction
        return max_count / total
    
    def _determine_dominant_direction(
        self,
        bullish_score: float,
        bearish_score: float,
        neutral_score: float,
    ) -> str:
        """
        Determine dominant direction.
        
        Args:
            bullish_score: Bullish score
            bearish_score: Bearish score
            neutral_score: Neutral score
            
        Returns:
            Dominant direction
        """
        if bullish_score > bearish_score and bullish_score > neutral_score:
            return SignalDirection.BULLISH.value
        elif bearish_score > bullish_score and bearish_score > neutral_score:
            return SignalDirection.BEARISH.value
        else:
            return SignalDirection.NEUTRAL.value
    
    def _identify_missing_evidence(self, evidence_list: List[Evidence]) -> List[str]:
        """
        Identify missing evidence categories.
        
        Args:
            evidence_list: List of Evidence
            
        Returns:
            List of missing categories
        """
        all_categories = {
            "trend", "momentum", "relative_strength", "liquidity",
            "options", "fundamentals", "macro", "sentiment",
            "market_breadth", "sector_rotation", "conditional_interaction",
        }
        
        present_categories = {e.category for e in evidence_list}
        missing = all_categories - present_categories
        
        return sorted(list(missing))
    
    def fuse_with_confidence(
        self,
        evidence_list: List[Evidence],
        weights: Optional[Dict[str, float]] = None,
        correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> tuple[FusionResult, float]:
        """
        Fuse evidence and return confidence in fusion.
        
        Args:
            evidence_list: List of Evidence objects
            weights: Optional weights for each evidence
            correlation_matrix: Optional correlation matrix
            
        Returns:
            Tuple of (FusionResult, confidence_score)
        """
        fusion_result = self.fuse(evidence_list, weights, correlation_matrix)
        
        # Calculate confidence based on:
        # - Evidence count
        # - Agreement score
        # - Missing evidence
        confidence = self._calculate_fusion_confidence(fusion_result)
        
        return fusion_result, confidence
    
    def _calculate_fusion_confidence(self, fusion_result: FusionResult) -> float:
        """
        Calculate confidence in fusion result.
        
        Args:
            fusion_result: FusionResult
            
        Returns:
            Confidence score (0-1)
        """
        confidence = 0.0
        
        # Evidence count component
        evidence_component = min(fusion_result.evidence_count / 10.0, 1.0)
        confidence += evidence_component * 0.4
        
        # Agreement component
        confidence += fusion_result.agreement_score * 0.4
        
        # Missing evidence penalty
        missing_penalty = len(fusion_result.missing_evidence) / 10.0
        confidence -= missing_penalty * 0.2
        
        return max(0.0, min(1.0, confidence))


def fuse_evidence(
    evidence_list: List[Evidence],
    weights: Optional[Dict[str, float]] = None,
) -> FusionResult:
    """
    Convenience function to fuse evidence.
    
    Args:
        evidence_list: List of Evidence objects
        weights: Optional weights for each evidence
        
    Returns:
        FusionResult
    """
    engine = FusionEngine()
    return engine.fuse(evidence_list, weights)
