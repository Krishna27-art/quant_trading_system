"""
Confidence Engine

Assesses confidence in trading signals.
Confidence depends on factor agreement, data quality, regime match, and historical performance.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

from meta_alpha.evidence_engine.evidence import Evidence
from meta_alpha.evidence_engine.fusion_engine import FusionResult
from meta_alpha.evidence_weighting.quality_score import QualityScore
from research.interactions.market_context.market_context import MarketContext
from utils.logger import get_logger

logger = get_logger("meta_alpha.confidence_engine")


@dataclass
class ConfidenceResult:
    """Result of confidence assessment."""
    confidence_level: str  # "HIGH", "MEDIUM", "LOW"
    confidence_score: float  # 0-1
    factor_agreement: float
    data_quality: float
    regime_match: float
    historical_performance: float
    missing_data_penalty: float
    model_stability: float
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate confidence result.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check confidence score is between 0 and 1
        if not (0.0 <= self.confidence_score <= 1.0):
            errors.append(f"Confidence score must be between 0 and 1, got {self.confidence_score}")
        
        # Check all component scores are between 0 and 1
        components = [
            ("factor_agreement", self.factor_agreement),
            ("data_quality", self.data_quality),
            ("regime_match", self.regime_match),
            ("historical_performance", self.historical_performance),
            ("missing_data_penalty", self.missing_data_penalty),
            ("model_stability", self.model_stability),
        ]
        
        for name, value in components:
            if not (0.0 <= value <= 1.0):
                errors.append(f"{name} must be between 0 and 1, got {value}")
        
        # Check confidence level is valid
        valid_levels = ["HIGH", "MEDIUM", "LOW"]
        if self.confidence_level not in valid_levels:
            errors.append(f"Invalid confidence level: {self.confidence_level}")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "confidence_level": self.confidence_level,
            "confidence_score": round(self.confidence_score, 4),
            "factor_agreement": round(self.factor_agreement, 4),
            "data_quality": round(self.data_quality, 4),
            "regime_match": round(self.regime_match, 4),
            "historical_performance": round(self.historical_performance, 4),
            "missing_data_penalty": round(self.missing_data_penalty, 4),
            "model_stability": round(self.model_stability, 4),
        }


class ConfidenceEngine:
    """
    Assesses confidence in trading signals.
    
    Confidence depends on:
    - Factor agreement
    - Data quality
    - Regime match
    - Historical performance
    - Missing data
    - Model stability
    """
    
    def __init__(
        self,
        high_threshold: float = 0.7,
        low_threshold: float = 0.4,
    ):
        """
        Initialize confidence engine.
        
        Args:
            high_threshold: Threshold for HIGH confidence
            low_threshold: Threshold for LOW confidence
        """
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self._logger = get_logger("meta_alpha.confidence_engine")
    
    def assess_confidence(
        self,
        evidence_list: List[Evidence],
        fusion_result: FusionResult,
        quality_scores: Optional[Dict[str, QualityScore]] = None,
        current_context: Optional[MarketContext] = None,
        historical_performance: Optional[Dict[str, float]] = None,
    ) -> ConfidenceResult:
        """
        Assess confidence in signal.
        
        Args:
            evidence_list: List of Evidence objects
            fusion_result: FusionResult from evidence fusion
            quality_scores: Optional quality scores for evidence
            current_context: Current market context
            historical_performance: Optional historical performance data
            
        Returns:
            ConfidenceResult
        """
        # Calculate factor agreement
        factor_agreement = self._calculate_factor_agreement(evidence_list)
        
        # Calculate data quality
        data_quality = self._calculate_data_quality(quality_scores, evidence_list)
        
        # Calculate regime match
        regime_match = self._calculate_regime_match(evidence_list, current_context)
        
        # Calculate historical performance
        historical_performance_score = self._calculate_historical_performance(
            evidence_list,
            historical_performance,
        )
        
        # Calculate missing data penalty
        missing_data_penalty = self._calculate_missing_data_penalty(evidence_list)
        
        # Calculate model stability
        model_stability = self._calculate_model_stability(fusion_result)
        
        # Combine into overall confidence score
        confidence_score = self._calculate_overall_confidence(
            factor_agreement,
            data_quality,
            regime_match,
            historical_performance_score,
            missing_data_penalty,
            model_stability,
        )
        
        # Determine confidence level
        confidence_level = self._determine_confidence_level(confidence_score)
        
        return ConfidenceResult(
            confidence_level=confidence_level,
            confidence_score=confidence_score,
            factor_agreement=factor_agreement,
            data_quality=data_quality,
            regime_match=regime_match,
            historical_performance=historical_performance_score,
            missing_data_penalty=missing_data_penalty,
            model_stability=model_stability,
        )
    
    def _calculate_factor_agreement(self, evidence_list: List[Evidence]) -> float:
        """
        Calculate factor agreement score.
        
        Args:
            evidence_list: List of Evidence
            
        Returns:
            Agreement score (0-1)
        """
        if not evidence_list:
            return 0.0
        
        # Count directions
        bullish = sum(1 for e in evidence_list if e.is_bullish())
        bearish = sum(1 for e in evidence_list if e.is_bearish())
        neutral = sum(1 for e in evidence_list if e.is_neutral())
        
        total = len(evidence_list)
        
        # Agreement is the proportion in the dominant direction
        max_count = max(bullish, bearish, neutral)
        agreement = max_count / total if total > 0 else 0.0
        
        return agreement
    
    def _calculate_data_quality(
        self,
        quality_scores: Optional[Dict[str, QualityScore]],
        evidence_list: List[Evidence],
    ) -> float:
        """
        Calculate data quality score.
        
        Args:
            quality_scores: Quality scores for evidence
            evidence_list: List of Evidence
            
        Returns:
            Data quality score (0-1)
        """
        if not quality_scores:
            return 0.5  # Neutral if no quality scores
        
        quality_values = []
        
        for evidence in evidence_list:
            evidence_id = f"{evidence.source}_{evidence.factor_name}"
            if evidence_id in quality_scores:
                quality_values.append(quality_scores[evidence_id].overall_score / 100.0)
        
        if not quality_values:
            return 0.5
        
        return sum(quality_values) / len(quality_values)
    
    def _calculate_regime_match(
        self,
        evidence_list: List[Evidence],
        current_context: Optional[MarketContext],
    ) -> float:
        """
        Calculate regime match score.
        
        Args:
            evidence_list: List of Evidence
            current_context: Current market context
            
        Returns:
            Regime match score (0-1)
        """
        if current_context is None:
            return 0.5  # Neutral if no context
        
        if not evidence_list:
            return 0.0
        
        match_scores = []
        
        for evidence in evidence_list:
            # Check if evidence matches current regime
            match = 0.5  # Default neutral
            
            if evidence.category == "trend":
                if current_context.trend in ["bull", "bear"]:
                    match = 0.8
                else:
                    match = 0.3
            
            elif evidence.category == "momentum":
                if current_context.volatility == "high":
                    match = 0.7
                elif current_context.volatility == "low":
                    match = 0.4
                else:
                    match = 0.5
            
            elif evidence.category == "options":
                if current_context.options_sentiment == evidence.signal_direction:
                    match = 0.8
                else:
                    match = 0.3
            
            match_scores.append(match)
        
        return sum(match_scores) / len(match_scores)
    
    def _calculate_historical_performance(
        self,
        evidence_list: List[Evidence],
        historical_performance: Optional[Dict[str, float]],
    ) -> float:
        """
        Calculate historical performance score.
        
        Args:
            evidence_list: List of Evidence
            historical_performance: Historical performance data
            
        Returns:
            Historical performance score (0-1)
        """
        if not historical_performance:
            return 0.5  # Neutral if no historical data
        
        performance_values = []
        
        for evidence in evidence_list:
            evidence_id = f"{evidence.source}_{evidence.factor_name}"
            if evidence_id in historical_performance:
                # Normalize performance to 0-1 range
                perf = historical_performance[evidence_id]
                normalized = (perf + 0.2) / 0.4  # Assuming range [-0.2, 0.2]
                performance_values.append(max(0.0, min(1.0, normalized)))
        
        if not performance_values:
            return 0.5
        
        return sum(performance_values) / len(performance_values)
    
    def _calculate_missing_data_penalty(self, evidence_list: List[Evidence]) -> float:
        """
        Calculate missing data penalty.
        
        Args:
            evidence_list: List of Evidence
            
        Returns:
            Missing data penalty score (0-1, higher is better)
        """
        # Check for missing confidence or strength
        missing_count = 0
        
        for evidence in evidence_list:
            if evidence.confidence is None or evidence.strength is None:
                missing_count += 1
        
        if not evidence_list:
            return 0.0
        
        # Penalty is inverse of missing ratio
        missing_ratio = missing_count / len(evidence_list)
        return 1.0 - missing_ratio
    
    def _calculate_model_stability(self, fusion_result: FusionResult) -> float:
        """
        Calculate model stability score.
        
        Args:
            fusion_result: FusionResult
            
        Returns:
            Model stability score (0-1)
        """
        # Stability based on agreement and evidence count
        stability = fusion_result.agreement_score * 0.7
        
        # Boost for sufficient evidence
        if fusion_result.evidence_count >= 5:
            stability += 0.3
        
        return min(1.0, stability)
    
    def _calculate_overall_confidence(
        self,
        factor_agreement: float,
        data_quality: float,
        regime_match: float,
        historical_performance: float,
        missing_data_penalty: float,
        model_stability: float,
    ) -> float:
        """
        Calculate overall confidence score.
        
        Args:
            factor_agreement: Factor agreement score
            data_quality: Data quality score
            regime_match: Regime match score
            historical_performance: Historical performance score
            missing_data_penalty: Missing data penalty score
            model_stability: Model stability score
            
        Returns:
            Overall confidence score (0-1)
        """
        # Weighted average
        weights = {
            "factor_agreement": 0.25,
            "data_quality": 0.20,
            "regime_match": 0.15,
            "historical_performance": 0.20,
            "missing_data_penalty": 0.10,
            "model_stability": 0.10,
        }
        
        confidence = (
            factor_agreement * weights["factor_agreement"] +
            data_quality * weights["data_quality"] +
            regime_match * weights["regime_match"] +
            historical_performance * weights["historical_performance"] +
            missing_data_penalty * weights["missing_data_penalty"] +
            model_stability * weights["model_stability"]
        )
        
        return max(0.0, min(1.0, confidence))
    
    def _determine_confidence_level(self, confidence_score: float) -> str:
        """
        Determine confidence level from score.
        
        Args:
            confidence_score: Confidence score
            
        Returns:
            Confidence level: "HIGH", "MEDIUM", or "LOW"
        """
        if confidence_score >= self.high_threshold:
            return "HIGH"
        elif confidence_score >= self.low_threshold:
            return "MEDIUM"
        else:
            return "LOW"


def assess_confidence(
    evidence_list: List[Evidence],
    fusion_result: FusionResult,
) -> ConfidenceResult:
    """
    Convenience function to assess confidence.
    
    Args:
        evidence_list: List of Evidence
        fusion_result: FusionResult
        
    Returns:
        ConfidenceResult
    """
    engine = ConfidenceEngine()
    return engine.assess_confidence(evidence_list, fusion_result)
