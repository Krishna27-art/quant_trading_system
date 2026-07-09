"""
Confidence Engine

Assesses confidence in trading signals.
Confidence depends on data quality, evidence agreement, and historical performance.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger("signal_engine.confidence")


@dataclass
class ConfidenceResult:
    """Confidence assessment for a signal."""
    overall_confidence: float
    data_quality_confidence: float
    evidence_agreement_confidence: float
    regime_similarity_confidence: float
    factor_agreement_confidence: float
    historical_performance_confidence: float
    confidence_level: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "overall_confidence": round(self.overall_confidence, 4),
            "data_quality_confidence": round(self.data_quality_confidence, 4),
            "evidence_agreement_confidence": round(self.evidence_agreement_confidence, 4),
            "regime_similarity_confidence": round(self.regime_similarity_confidence, 4),
            "factor_agreement_confidence": round(self.factor_agreement_confidence, 4),
            "historical_performance_confidence": round(self.historical_performance_confidence, 4),
            "confidence_level": self.confidence_level,
        }


class ConfidenceEngine:
    """
    Assesses confidence in trading signals.
    
    Confidence depends on:
    - Data quality
    - Evidence agreement
    - Regime similarity
    - Factor agreement
    - Historical performance
    """
    
    def __init__(
        self,
        confidence_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize confidence engine.
        
        Args:
            confidence_weights: Optional weights for each confidence component
        """
        self.confidence_weights = confidence_weights or {
            "data_quality": 0.2,
            "evidence_agreement": 0.25,
            "regime_similarity": 0.15,
            "factor_agreement": 0.2,
            "historical_performance": 0.2,
        }
        self._logger = get_logger("signal_engine.confidence")
    
    def calculate_confidence(
        self,
        data_quality: float,
        evidence_agreement: float,
        regime_similarity: float = 0.5,
        factor_agreement: float = 0.5,
        historical_performance: float = 0.5,
    ) -> ConfidenceResult:
        """
        Calculate overall confidence.
        
        Args:
            data_quality: Data quality score (0 to 1)
            evidence_agreement: Evidence agreement score (0 to 1)
            regime_similarity: Regime similarity score (0 to 1)
            factor_agreement: Factor agreement score (0 to 1)
            historical_performance: Historical performance score (0 to 1)
            
        Returns:
            ConfidenceResult
        """
        # Calculate weighted overall confidence
        overall = (
            data_quality * self.confidence_weights["data_quality"] +
            evidence_agreement * self.confidence_weights["evidence_agreement"] +
            regime_similarity * self.confidence_weights["regime_similarity"] +
            factor_agreement * self.confidence_weights["factor_agreement"] +
            historical_performance * self.confidence_weights["historical_performance"]
        )
        
        # Determine confidence level
        confidence_level = self._get_confidence_level(overall)
        
        return ConfidenceResult(
            overall_confidence=overall,
            data_quality_confidence=data_quality,
            evidence_agreement_confidence=evidence_agreement,
            regime_similarity_confidence=regime_similarity,
            factor_agreement_confidence=factor_agreement,
            historical_performance_confidence=historical_performance,
            confidence_level=confidence_level,
        )
    
    def calculate_evidence_agreement(
        self,
        evidence_scores: Dict[str, float],
    ) -> float:
        """
        Calculate agreement among evidence scores.
        
        Args:
            evidence_scores: Dictionary of evidence scores
            
        Returns:
            Agreement score (0 to 1)
        """
        if not evidence_scores:
            return 0.0
        
        values = list(evidence_scores.values())
        
        # Calculate standard deviation
        std_dev = np.std(values)
        
        # Agreement is inverse of dispersion
        agreement = max(0.0, 1.0 - std_dev)
        
        return agreement
    
    def calculate_factor_agreement(
        self,
        category_scores: Dict[str, float],
    ) -> float:
        """
        Calculate agreement across factor categories.
        
        Args:
            category_scores: Dictionary mapping categories to scores
            
        Returns:
            Agreement score (0 to 1)
        """
        if not category_scores:
            return 0.0
        
        values = list(category_scores.values())
        
        # Calculate standard deviation
        std_dev = np.std(values)
        
        # Agreement is inverse of dispersion
        agreement = max(0.0, 1.0 - std_dev)
        
        return agreement
    
    def calculate_regime_similarity(
        self,
        current_regime: str,
        historical_best_regime: str,
    ) -> float:
        """
        Calculate similarity between current and best historical regime.
        
        Args:
            current_regime: Current market regime
            historical_best_regime: Best historical regime for this signal
            
        Returns:
            Similarity score (0 to 1)
        """
        if current_regime == historical_best_regime:
            return 1.0
        
        # Partial similarity for related regimes
        regime_groups = {
            "bull": ["bull", "strong_bull", "weak_bull"],
            "bear": ["bear", "strong_bear", "weak_bear"],
            "sideways": ["sideways", "low_volatility", "high_volatility"],
        }
        
        for group in regime_groups.values():
            if current_regime in group and historical_best_regime in group:
                return 0.5
        
        return 0.0
    
    def _get_confidence_level(self, confidence: float) -> str:
        """
        Get confidence level from score.
        
        Args:
            confidence: Confidence score
            
        Returns:
            Confidence level: "HIGH", "MEDIUM", "LOW"
        """
        if confidence >= 0.8:
            return "HIGH"
        elif confidence >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def is_confidence_sufficient(
        self,
        confidence_result: ConfidenceResult,
        min_confidence: float = 0.6,
    ) -> bool:
        """
        Check if confidence meets minimum threshold.
        
        Args:
            confidence_result: ConfidenceResult
            min_confidence: Minimum confidence threshold
            
        Returns:
            True if confidence is sufficient
        """
        return confidence_result.overall_confidence >= min_confidence


def calculate_confidence(
    data_quality: float,
    evidence_agreement: float,
    regime_similarity: float = 0.5,
    factor_agreement: float = 0.5,
    historical_performance: float = 0.5,
) -> ConfidenceResult:
    """
    Convenience function to calculate confidence.
    
    Args:
        data_quality: Data quality score
        evidence_agreement: Evidence agreement score
        regime_similarity: Regime similarity score
        factor_agreement: Factor agreement score
        historical_performance: Historical performance score
        
    Returns:
        ConfidenceResult
    """
    engine = ConfidenceEngine()
    return engine.calculate_confidence(
        data_quality,
        evidence_agreement,
        regime_similarity,
        factor_agreement,
        historical_performance,
    )
