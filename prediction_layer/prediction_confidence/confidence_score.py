"""
Confidence Score Calculator

Combines all confidence factors into a final confidence score.
This is the main entry point for the confidence engine.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from prediction_layer.prediction_confidence.model_agreement import ModelAgreementResult
from prediction_layer.prediction_confidence.signal_confidence import SignalAgreementResult
from prediction_layer.prediction_confidence.feature_confidence import FeatureConfidenceResult
from prediction_layer.prediction_confidence.regime_confidence import RegimeConfidenceResult
from prediction_layer.prediction_confidence.historical_similarity import HistoricalSimilarityResult

from utils.logger import get_logger

logger = get_logger("prediction_layer.prediction_confidence.confidence_score")


@dataclass
class ConfidenceResult:
    """Final confidence result combining all factors."""
    overall_confidence: float
    confidence_level: str
    model_agreement: float
    signal_agreement: float
    feature_confidence: float
    regime_confidence: float
    historical_similarity: float
    data_quality: float
    expected_return: float
    risk_score: float
    reason: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "overall_confidence": round(self.overall_confidence, 4),
            "confidence_level": self.confidence_level,
            "model_agreement": round(self.model_agreement, 4),
            "signal_agreement": round(self.signal_agreement, 4),
            "feature_confidence": round(self.feature_confidence, 4),
            "regime_confidence": round(self.regime_confidence, 4),
            "historical_similarity": round(self.historical_similarity, 4),
            "data_quality": round(self.data_quality, 4),
            "expected_return": round(self.expected_return, 4),
            "risk_score": round(self.risk_score, 4),
            "reason": self.reason,
        }


class ConfidenceScoreCalculator:
    """
    Combines all confidence factors into a final confidence score.
    
    Factors:
    - Model agreement (25%)
    - Signal agreement (20%)
    - Feature confidence (20%)
    - Regime confidence (15%)
    - Historical similarity (10%)
    - Data quality (10%)
    """
    
    def __init__(
        self,
        confidence_weights: Optional[Dict[str, float]] = None,
        high_threshold: float = 0.8,
        low_threshold: float = 0.5,
    ):
        """
        Initialize confidence score calculator.
        
        Args:
            confidence_weights: Optional weights for each confidence factor
            high_threshold: Threshold for HIGH confidence
            low_threshold: Threshold for LOW confidence
        """
        self.confidence_weights = confidence_weights or {
            "model_agreement": 0.25,
            "signal_agreement": 0.20,
            "feature_confidence": 0.20,
            "regime_confidence": 0.15,
            "historical_similarity": 0.10,
            "data_quality": 0.10,
        }
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self._logger = get_logger("prediction_layer.prediction_confidence.confidence_score")
    
    def calculate_confidence(
        self,
        model_agreement: ModelAgreementResult,
        signal_agreement: SignalAgreementResult,
        feature_confidence: FeatureConfidenceResult,
        regime_confidence: RegimeConfidenceResult,
        historical_similarity: HistoricalSimilarityResult,
        data_quality: float,
        expected_return: float,
        risk_score: float,
    ) -> ConfidenceResult:
        """
        Calculate overall confidence from all factors.
        
        Args:
            model_agreement: Model agreement result
            signal_agreement: Signal agreement result
            feature_confidence: Feature confidence result
            regime_confidence: Regime confidence result
            historical_similarity: Historical similarity result
            data_quality: Data quality score (0-1)
            expected_return: Expected return percentage
            risk_score: Risk score (0-1)
            
        Returns:
            ConfidenceResult
        """
        # Extract scores from results
        model_score = model_agreement.agreement_score
        signal_score = signal_agreement.agreement_score
        feature_score = feature_confidence.confidence_score
        regime_score = regime_confidence.confidence_score
        history_score = historical_similarity.confidence_score
        
        # Calculate weighted overall confidence
        overall_confidence = (
            model_score * self.confidence_weights["model_agreement"] +
            signal_score * self.confidence_weights["signal_agreement"] +
            feature_score * self.confidence_weights["feature_confidence"] +
            regime_score * self.confidence_weights["regime_confidence"] +
            history_score * self.confidence_weights["historical_similarity"] +
            data_quality * self.confidence_weights["data_quality"]
        )
        
        # Adjust confidence based on risk-return profile
        risk_adjusted_confidence = self._adjust_for_risk_return(
            overall_confidence,
            expected_return,
            risk_score,
        )
        
        # Determine confidence level
        confidence_level = self._get_confidence_level(risk_adjusted_confidence)
        
        # Generate reason
        reason = self._generate_reason(
            model_agreement,
            signal_agreement,
            feature_confidence,
            regime_confidence,
            historical_similarity,
        )
        
        self._logger.info(
            f"Overall confidence calculated: {confidence_level} "
            f"(score={risk_adjusted_confidence:.4f})"
        )
        
        return ConfidenceResult(
            overall_confidence=risk_adjusted_confidence,
            confidence_level=confidence_level,
            model_agreement=model_score,
            signal_agreement=signal_score,
            feature_confidence=feature_score,
            regime_confidence=regime_score,
            historical_similarity=history_score,
            data_quality=data_quality,
            expected_return=expected_return,
            risk_score=risk_score,
            reason=reason,
        )
    
    def _adjust_for_risk_return(
        self,
        confidence: float,
        expected_return: float,
        risk_score: float,
    ) -> float:
        """
        Adjust confidence based on risk-return profile.
        
        Args:
            confidence: Base confidence score
            expected_return: Expected return percentage
            risk_score: Risk score (0-1, higher is riskier)
            
        Returns:
            Adjusted confidence score
        """
        # Boost confidence for high expected return
        return_boost = max(0.0, min(0.1, expected_return / 100.0))
        
        # Reduce confidence for high risk
        risk_penalty = risk_score * 0.15
        
        adjusted = confidence + return_boost - risk_penalty
        
        return max(0.0, min(1.0, adjusted))
    
    def _get_confidence_level(self, confidence: float) -> str:
        """
        Get confidence level from score.
        
        Args:
            confidence: Confidence score
            
        Returns:
            Confidence level: "HIGH", "MEDIUM", "LOW"
        """
        if confidence >= self.high_threshold:
            return "HIGH"
        elif confidence >= self.low_threshold:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_reason(
        self,
        model_agreement: ModelAgreementResult,
        signal_agreement: SignalAgreementResult,
        feature_confidence: FeatureConfidenceResult,
        regime_confidence: RegimeConfidenceResult,
        historical_similarity: HistoricalSimilarityResult,
    ) -> str:
        """
        Generate human-readable reason for confidence level.
        
        Args:
            model_agreement: Model agreement result
            signal_agreement: Signal agreement result
            feature_confidence: Feature confidence result
            regime_confidence: Regime confidence result
            historical_similarity: Historical similarity result
            
        Returns:
            Reason string
        """
        reasons = []
        
        # Model agreement
        if model_agreement.agreement_level == "HIGH":
            reasons.append(f"Strong model agreement ({model_agreement.dominant_direction.value})")
        elif model_agreement.agreement_level == "MEDIUM":
            reasons.append(f"Moderate model agreement")
        
        # Signal agreement
        if signal_agreement.agreement_level == "HIGH":
            reasons.append(f"Strong signal consensus ({signal_agreement.dominant_direction.value})")
        elif signal_agreement.agreement_level == "MEDIUM":
            reasons.append(f"Moderate signal agreement")
        
        # Feature confidence
        if feature_confidence.confidence_level == "HIGH":
            reasons.append(f"High-quality features ({feature_confidence.top_features_used} top features)")
        elif feature_confidence.confidence_level == "MEDIUM":
            reasons.append(f"Good feature quality")
        
        # Regime confidence
        if regime_confidence.confidence_level == "HIGH":
            reasons.append(f"Favorable market regime ({regime_confidence.current_regime.value})")
        elif regime_confidence.confidence_level == "MEDIUM":
            reasons.append(f"Neutral market regime")
        
        # Historical similarity
        if historical_similarity.confidence_level == "HIGH":
            reasons.append(f"Strong historical pattern match ({historical_similarity.success_rate:.0%} success)")
        elif historical_similarity.confidence_level == "MEDIUM":
            reasons.append(f"Moderate historical similarity")
        
        if not reasons:
            return "Limited confidence factors available"
        
        return ". ".join(reasons) + "."
    
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


def calculate_confidence_score(
    model_agreement: ModelAgreementResult,
    signal_agreement: SignalAgreementResult,
    feature_confidence: FeatureConfidenceResult,
    regime_confidence: RegimeConfidenceResult,
    historical_similarity: HistoricalSimilarityResult,
    data_quality: float,
    expected_return: float,
    risk_score: float,
) -> ConfidenceResult:
    """
    Convenience function to calculate overall confidence score.
    
    Args:
        model_agreement: Model agreement result
        signal_agreement: Signal agreement result
        feature_confidence: Feature confidence result
        regime_confidence: Regime confidence result
        historical_similarity: Historical similarity result
        data_quality: Data quality score (0-1)
        expected_return: Expected return percentage
        risk_score: Risk score (0-1)
        
    Returns:
        ConfidenceResult
    """
    calculator = ConfidenceScoreCalculator()
    return calculator.calculate_confidence(
        model_agreement,
        signal_agreement,
        feature_confidence,
        regime_confidence,
        historical_similarity,
        data_quality,
        expected_return,
        risk_score,
    )
