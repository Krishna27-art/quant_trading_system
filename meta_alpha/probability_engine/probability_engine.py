"""
Probability Engine

Converts evidence fusion results into probability estimates.
Produces calibrated probability of success for trading decisions.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

from meta_alpha.evidence_engine.fusion_engine import FusionResult
from utils.logger import get_logger

logger = get_logger("meta_alpha.probability_engine")


@dataclass
class ProbabilityResult:
    """Result of probability estimation."""
    probability: float
    confidence_interval: tuple[float, float]
    calibration_score: Optional[float]
    sample_size: int
    is_calibrated: bool
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate probability result.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check probability is between 0 and 1
        if not (0.0 <= self.probability <= 1.0):
            errors.append(f"Probability must be between 0 and 1, got {self.probability}")
        
        # Check confidence interval bounds
        if self.confidence_interval[0] < 0.0:
            errors.append(f"Confidence interval lower bound must be >= 0, got {self.confidence_interval[0]}")
        if self.confidence_interval[1] > 1.0:
            errors.append(f"Confidence interval upper bound must be <= 1, got {self.confidence_interval[1]}")
        if self.confidence_interval[0] > self.confidence_interval[1]:
            errors.append(f"Confidence interval lower bound must be <= upper bound")
        
        # Check for NaN or Inf
        if np.isnan(self.probability) or np.isinf(self.probability):
            errors.append("Probability cannot be NaN or Inf")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "probability": round(self.probability, 4),
            "confidence_interval": (
                round(self.confidence_interval[0], 4),
                round(self.confidence_interval[1], 4),
            ),
            "calibration_score": round(self.calibration_score, 4) if self.calibration_score is not None else None,
            "sample_size": self.sample_size,
            "is_calibrated": self.is_calibrated,
        }


class ProbabilityEngine:
    """
    Converts evidence fusion results into probability estimates.
    
    Process:
    - Takes fusion result (bullish/bearish/neutral scores)
    - Converts to probability of success
    - Applies calibration if available
    - Returns confidence interval
    """
    
    def __init__(
        self,
        neutral_threshold: float = 0.5,
        min_probability: float = 0.1,
        max_probability: float = 0.9,
    ):
        """
        Initialize probability engine.
        
        Args:
            neutral_threshold: Threshold for neutral probability
            min_probability: Minimum probability output
            max_probability: Maximum probability output
        """
        self.neutral_threshold = neutral_threshold
        self.min_probability = min_probability
        self.max_probability = max_probability
        self._logger = get_logger("meta_alpha.probability_engine")
    
    def calculate_probability(
        self,
        fusion_result: FusionResult,
        calibration_params: Optional[Dict[str, Any]] = None,
    ) -> ProbabilityResult:
        """
        Calculate probability from fusion result.
        
        Args:
            fusion_result: FusionResult from evidence fusion
            calibration_params: Optional calibration parameters
            
        Returns:
            ProbabilityResult
        """
        # Calculate base probability from net score
        net_score = fusion_result.get_net_score()
        
        # Convert net score (-1 to 1) to probability (0 to 1)
        base_probability = (net_score + 1.0) / 2.0
        
        # Apply calibration if available
        if calibration_params:
            probability = self._apply_calibration(base_probability, calibration_params)
        else:
            probability = base_probability
        
        # Clamp to valid range
        probability = max(self.min_probability, min(self.max_probability, probability))
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(
            probability,
            fusion_result.evidence_count,
            fusion_result.agreement_score,
        )
        
        # Calculate calibration score
        calibration_score = self._calculate_calibration_score(
            probability,
            fusion_result,
        )
        
        return ProbabilityResult(
            probability=probability,
            confidence_interval=confidence_interval,
            calibration_score=calibration_score,
            sample_size=fusion_result.evidence_count,
            is_calibrated=calibration_params is not None,
        )
    
    def _apply_calibration(
        self,
        probability: float,
        calibration_params: Dict[str, Any],
    ) -> float:
        """
        Apply calibration to probability.
        
        Args:
            probability: Base probability
            calibration_params: Calibration parameters
            
        Returns:
            Calibrated probability
        """
        method = calibration_params.get("method", "none")
        
        if method == "platt_scaling":
            # Platt scaling: sigmoid(a * p + b)
            a = calibration_params.get("a", 1.0)
            b = calibration_params.get("b", 0.0)
            calibrated = 1.0 / (1.0 + np.exp(-(a * probability + b)))
        elif method == "isotonic":
            # Isotonic regression (simplified)
            # In practice, would use isotonic regression model
            slope = calibration_params.get("slope", 1.0)
            intercept = calibration_params.get("intercept", 0.0)
            calibrated = slope * probability + intercept
        elif method == "temperature":
            # Temperature scaling
            temperature = calibration_params.get("temperature", 1.0)
            calibrated = probability ** (1.0 / temperature)
        else:
            calibrated = probability
        
        return max(0.0, min(1.0, calibrated))
    
    def _calculate_confidence_interval(
        self,
        probability: float,
        evidence_count: int,
        agreement_score: float,
    ) -> tuple[float, float]:
        """
        Calculate confidence interval for probability.
        
        Args:
            probability: Probability estimate
            evidence_count: Number of evidence items
            agreement_score: Agreement among evidence
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # Standard error based on evidence count
        # More evidence = narrower interval
        if evidence_count > 0:
            se = np.sqrt(probability * (1 - probability) / evidence_count)
        else:
            se = 0.1  # Default standard error
        
        # Adjust by agreement score
        # Higher agreement = narrower interval
        se *= (1.0 - agreement_score * 0.5)
        
        # 95% confidence interval (approximately 2 standard errors)
        lower = max(0.0, probability - 2 * se)
        upper = min(1.0, probability + 2 * se)
        
        return (lower, upper)
    
    def _calculate_calibration_score(
        self,
        probability: float,
        fusion_result: FusionResult,
    ) -> float:
        """
        Calculate calibration quality score.
        
        Args:
            probability: Probability estimate
            fusion_result: FusionResult
            
        Returns:
            Calibration score (0-1)
        """
        score = 0.0
        
        # Evidence count component
        evidence_component = min(fusion_result.evidence_count / 10.0, 1.0)
        score += evidence_component * 0.3
        
        # Agreement component
        score += fusion_result.agreement_score * 0.4
        
        # Missing evidence penalty
        missing_penalty = len(fusion_result.missing_evidence) / 10.0
        score -= missing_penalty * 0.3
        
        return max(0.0, min(1.0, score))
    
    def classify_signal(
        self,
        probability: float,
        buy_threshold: float = 0.6,
        sell_threshold: float = 0.4,
    ) -> str:
        """
        Classify probability into signal.
        
        Args:
            probability: Probability estimate
            buy_threshold: Threshold for buy signal
            sell_threshold: Threshold for sell signal
            
        Returns:
            Signal: "BUY", "SELL", or "HOLD"
        """
        if probability >= buy_threshold:
            return "BUY"
        elif probability <= sell_threshold:
            return "SELL"
        else:
            return "HOLD"
    
    def batch_calculate_probability(
        self,
        fusion_results: list[FusionResult],
        calibration_params: Optional[Dict[str, Any]] = None,
    ) -> list[ProbabilityResult]:
        """
        Calculate probabilities for multiple fusion results.
        
        Args:
            fusion_results: List of FusionResult
            calibration_params: Optional calibration parameters
            
        Returns:
            List of ProbabilityResult
        """
        results = []
        
        for fusion_result in fusion_results:
            try:
                probability_result = self.calculate_probability(
                    fusion_result,
                    calibration_params,
                )
                results.append(probability_result)
            except Exception as e:
                self._logger.error(f"Failed to calculate probability: {e}")
                # Return neutral result on error
                results.append(ProbabilityResult(
                    probability=0.5,
                    confidence_interval=(0.4, 0.6),
                    calibration_score=0.0,
                    sample_size=0,
                    is_calibrated=False,
                ))
        
        return results


def calculate_probability(
    fusion_result: FusionResult,
    calibration_params: Optional[Dict[str, Any]] = None,
) -> ProbabilityResult:
    """
    Convenience function to calculate probability.
    
    Args:
        fusion_result: FusionResult from evidence fusion
        calibration_params: Optional calibration parameters
        
    Returns:
        ProbabilityResult
    """
    engine = ProbabilityEngine()
    return engine.calculate_probability(fusion_result, calibration_params)
