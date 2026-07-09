"""
Probability Engine

Converts alpha factor evidence into probability estimates.
Combines multiple independent signals into a unified probability score.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger("signal_engine.probability")


@dataclass
class ProbabilityResult:
    """Result of probability estimation."""
    probability: float
    evidence_count: int
    missing_evidence: int
    bullish_evidence: int
    bearish_evidence: int
    neutral_evidence: int
    confidence: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "probability": round(self.probability, 4),
            "evidence_count": self.evidence_count,
            "missing_evidence": self.missing_evidence,
            "bullish_evidence": self.bullish_evidence,
            "bearish_evidence": self.bearish_evidence,
            "neutral_evidence": self.neutral_evidence,
            "confidence": round(self.confidence, 4),
        }


class ProbabilityEngine:
    """
    Converts alpha factor evidence into probability estimates.
    
    Combines multiple independent signals into a unified probability score
    using evidence aggregation techniques.
    """
    
    def __init__(
        self,
        min_evidence: int = 3,
        bullish_threshold: float = 0.6,
        bearish_threshold: float = 0.4,
    ):
        """
        Initialize probability engine.
        
        Args:
            min_evidence: Minimum evidence required for valid probability
            bullish_threshold: Threshold for bullish classification
            bearish_threshold: Threshold for bearish classification
        """
        self.min_evidence = min_evidence
        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold
        self._logger = get_logger("signal_engine.probability")
    
    def estimate_probability(
        self,
        evidence: Dict[str, float],
        weights: Optional[Dict[str, float]] = None,
    ) -> ProbabilityResult:
        """
        Estimate probability from alpha factor evidence.
        
        Args:
            evidence: Dictionary mapping factor names to evidence scores (-1 to 1)
            weights: Optional weights for each factor (default: equal weights)
            
        Returns:
            ProbabilityResult
        """
        # Validate inputs
        if not evidence:
            return ProbabilityResult(
                probability=0.5,
                evidence_count=0,
                missing_evidence=0,
                bullish_evidence=0,
                bearish_evidence=0,
                neutral_evidence=0,
                confidence=0.0,
            )
        
        # Count evidence types
        bullish = sum(1 for v in evidence.values() if v > 0.2)
        bearish = sum(1 for v in evidence.values() if v < -0.2)
        neutral = sum(1 for v in evidence.values() if -0.2 <= v <= 0.2)
        
        # Calculate weighted average
        if weights:
            total_weight = sum(weights.get(k, 1.0) for k in evidence.keys())
            if total_weight == 0:
                weighted_sum = 0
            else:
                weighted_sum = sum(
                    v * weights.get(k, 1.0) for k, v in evidence.items()
                )
            probability = 0.5 + (weighted_sum / total_weight) * 0.5
        else:
            avg_evidence = np.mean(list(evidence.values()))
            probability = 0.5 + avg_evidence * 0.5
        
        # Clip to valid range
        probability = max(0.0, min(1.0, probability))
        
        # Calculate confidence based on evidence count and agreement
        confidence = self._calculate_confidence(evidence)
        
        return ProbabilityResult(
            probability=probability,
            evidence_count=len(evidence),
            missing_evidence=0,
            bullish_evidence=bullish,
            bearish_evidence=bearish,
            neutral_evidence=neutral,
            confidence=confidence,
        )
    
    def estimate_probability_with_missing(
        self,
        evidence: Dict[str, float],
        expected_factors: List[str],
        weights: Optional[Dict[str, float]] = None,
    ) -> ProbabilityResult:
        """
        Estimate probability accounting for missing evidence.
        
        Args:
            evidence: Dictionary mapping factor names to evidence scores
            expected_factors: List of expected factor names
            weights: Optional weights for each factor
            
        Returns:
            ProbabilityResult
        """
        # Calculate missing evidence
        missing = set(expected_factors) - set(evidence.keys())
        missing_count = len(missing)
        
        # Estimate probability with available evidence
        result = self.estimate_probability(evidence, weights)
        result.missing_evidence = missing_count
        
        # Adjust confidence based on missing evidence
        if missing_count > 0:
            evidence_ratio = len(evidence) / len(expected_factors)
            result.confidence *= evidence_ratio
        
        return result
    
    def _calculate_confidence(self, evidence: Dict[str, float]) -> float:
        """
        Calculate confidence in probability estimate.
        
        Args:
            evidence: Dictionary of evidence scores
            
        Returns:
            Confidence score (0 to 1)
        """
        if not evidence:
            return 0.0
        
        # Base confidence from evidence count
        evidence_confidence = min(len(evidence) / 10.0, 1.0)
        
        # Agreement confidence (how much evidence agrees)
        values = list(evidence.values())
        if len(values) > 1:
            std_dev = np.std(values)
            agreement_confidence = max(0.0, 1.0 - std_dev)
        else:
            agreement_confidence = 0.5
        
        # Combined confidence
        confidence = 0.6 * evidence_confidence + 0.4 * agreement_confidence
        
        return confidence
    
    def classify_signal(self, probability: float) -> str:
        """
        Classify signal based on probability.
        
        Args:
            probability: Probability score
            
        Returns:
            Classification: "STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"
        """
        if probability >= 0.8:
            return "STRONG_BUY"
        elif probability >= self.bullish_threshold:
            return "BUY"
        elif probability <= 0.2:
            return "STRONG_SELL"
        elif probability <= self.bearish_threshold:
            return "SELL"
        else:
            return "HOLD"
    
    def batch_estimate(
        self,
        evidence_list: List[Dict[str, float]],
        weights: Optional[Dict[str, float]] = None,
    ) -> List[ProbabilityResult]:
        """
        Estimate probabilities for multiple evidence sets.
        
        Args:
            evidence_list: List of evidence dictionaries
            weights: Optional weights for each factor
            
        Returns:
            List of ProbabilityResult
        """
        results = []
        
        for evidence in evidence_list:
            result = self.estimate_probability(evidence, weights)
            results.append(result)
        
        return results


def estimate_probability(
    evidence: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> ProbabilityResult:
    """
    Convenience function to estimate probability.
    
    Args:
        evidence: Dictionary mapping factor names to evidence scores
        weights: Optional weights for each factor
        
    Returns:
        ProbabilityResult
    """
    engine = ProbabilityEngine()
    return engine.estimate_probability(evidence, weights)
