"""
Model Agreement Calculator

Calculates agreement between multiple independently trained ML models.
High agreement between models increases confidence in predictions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

import numpy as np

from utils.logger import get_logger

logger = get_logger("prediction_layer.prediction_confidence.model_agreement")


class PredictionDirection(Enum):
    """Prediction direction enumeration."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class ModelPrediction:
    """Prediction from a single model."""
    model_name: str
    direction: PredictionDirection
    probability: float
    confidence: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "direction": self.direction.value,
            "probability": round(self.probability, 4),
            "confidence": round(self.confidence, 4) if self.confidence else None,
        }


@dataclass
class ModelAgreementResult:
    """Result of model agreement calculation."""
    agreement_score: float
    agreement_level: str
    dominant_direction: PredictionDirection
    direction_counts: Dict[str, int]
    average_probability: float
    probability_std: float
    participating_models: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "agreement_score": round(self.agreement_score, 4),
            "agreement_level": self.agreement_level,
            "dominant_direction": self.dominant_direction.value,
            "direction_counts": self.direction_counts,
            "average_probability": round(self.average_probability, 4),
            "probability_std": round(self.probability_std, 4),
            "participating_models": self.participating_models,
        }


class ModelAgreementCalculator:
    """
    Calculates agreement between multiple model predictions.
    
    Agreement is based on:
    - Direction consensus (how many models agree on direction)
    - Probability consistency (how similar are the probability scores)
    - Weighted by individual model confidence if available
    """
    
    def __init__(
        self,
        high_agreement_threshold: float = 0.8,
        low_agreement_threshold: float = 0.5,
    ):
        """
        Initialize model agreement calculator.
        
        Args:
            high_agreement_threshold: Threshold for HIGH agreement
            low_agreement_threshold: Threshold for LOW agreement
        """
        self.high_agreement_threshold = high_agreement_threshold
        self.low_agreement_threshold = low_agreement_threshold
        self._logger = get_logger("prediction_layer.prediction_confidence.model_agreement")
    
    def calculate_agreement(
        self,
        predictions: List[ModelPrediction],
    ) -> ModelAgreementResult:
        """
        Calculate agreement between model predictions.
        
        Args:
            predictions: List of ModelPrediction objects
            
        Returns:
            ModelAgreementResult
        """
        if not predictions:
            self._logger.warning("No predictions provided for agreement calculation")
            return ModelAgreementResult(
                agreement_score=0.0,
                agreement_level="NONE",
                dominant_direction=PredictionDirection.HOLD,
                direction_counts={},
                average_probability=0.0,
                probability_std=0.0,
                participating_models=[],
            )
        
        # Count directions
        direction_counts = self._count_directions(predictions)
        
        # Calculate dominant direction
        dominant_direction = self._get_dominant_direction(direction_counts)
        
        # Calculate direction agreement score
        direction_agreement = self._calculate_direction_agreement(
            direction_counts,
            len(predictions),
        )
        
        # Calculate probability statistics
        probabilities = [p.probability for p in predictions]
        average_probability = np.mean(probabilities)
        probability_std = np.std(probabilities)
        
        # Calculate probability consistency (inverse of std)
        probability_consistency = max(0.0, 1.0 - probability_std)
        
        # Combine direction agreement and probability consistency
        agreement_score = (
            direction_agreement * 0.7 +
            probability_consistency * 0.3
        )
        
        # Determine agreement level
        agreement_level = self._get_agreement_level(agreement_score)
        
        participating_models = [p.model_name for p in predictions]
        
        self._logger.info(
            f"Model agreement calculated: {agreement_level} "
            f"(score={agreement_score:.4f}, dominant={dominant_direction.value})"
        )
        
        return ModelAgreementResult(
            agreement_score=agreement_score,
            agreement_level=agreement_level,
            dominant_direction=dominant_direction,
            direction_counts=direction_counts,
            average_probability=average_probability,
            probability_std=probability_std,
            participating_models=participating_models,
        )
    
    def _count_directions(
        self,
        predictions: List[ModelPrediction],
    ) -> Dict[str, int]:
        """
        Count predictions by direction.
        
        Args:
            predictions: List of ModelPrediction objects
            
        Returns:
            Dictionary mapping direction to count
        """
        counts = {
            "BUY": 0,
            "SELL": 0,
            "HOLD": 0,
        }
        
        for prediction in predictions:
            counts[prediction.direction.value] += 1
        
        return counts
    
    def _get_dominant_direction(
        self,
        direction_counts: Dict[str, int],
    ) -> PredictionDirection:
        """
        Get the dominant prediction direction.
        
        Args:
            direction_counts: Dictionary mapping direction to count
            
        Returns:
            PredictionDirection
        """
        if not direction_counts:
            return PredictionDirection.HOLD
        
        max_count = max(direction_counts.values())
        
        # Get all directions with max count
        dominant_directions = [
            direction for direction, count in direction_counts.items()
            if count == max_count
        ]
        
        # If tie, prefer HOLD
        if len(dominant_directions) > 1:
            if "HOLD" in dominant_directions:
                return PredictionDirection.HOLD
        
        return PredictionDirection(dominant_directions[0])
    
    def _calculate_direction_agreement(
        self,
        direction_counts: Dict[str, int],
        total_predictions: int,
    ) -> float:
        """
        Calculate direction agreement score.
        
        Args:
            direction_counts: Dictionary mapping direction to count
            total_predictions: Total number of predictions
            
        Returns:
            Agreement score (0-1)
        """
        if total_predictions == 0:
            return 0.0
        
        max_count = max(direction_counts.values())
        agreement = max_count / total_predictions
        
        return agreement
    
    def _get_agreement_level(self, agreement_score: float) -> str:
        """
        Get agreement level from score.
        
        Args:
            agreement_score: Agreement score
            
        Returns:
            Agreement level: "HIGH", "MEDIUM", "LOW", "NONE"
        """
        if agreement_score >= self.high_agreement_threshold:
            return "HIGH"
        elif agreement_score >= self.low_agreement_threshold:
            return "MEDIUM"
        elif agreement_score > 0.0:
            return "LOW"
        else:
            return "NONE"
    
    def calculate_weighted_agreement(
        self,
        predictions: List[ModelPrediction],
        model_weights: Optional[Dict[str, float]] = None,
    ) -> ModelAgreementResult:
        """
        Calculate weighted agreement between model predictions.
        
        Args:
            predictions: List of ModelPrediction objects
            model_weights: Optional weights for each model
            
        Returns:
            ModelAgreementResult
        """
        if not model_weights:
            return self.calculate_agreement(predictions)
        
        # Apply weights to direction counts
        weighted_counts = {
            "BUY": 0.0,
            "SELL": 0.0,
            "HOLD": 0.0,
        }
        
        total_weight = 0.0
        
        for prediction in predictions:
            weight = model_weights.get(prediction.model_name, 1.0)
            weighted_counts[prediction.direction.value] += weight
            total_weight += weight
        
        # Normalize weighted counts
        if total_weight > 0:
            for direction in weighted_counts:
                weighted_counts[direction] /= total_weight
        
        # Calculate dominant direction from weighted counts
        max_weight = max(weighted_counts.values())
        dominant_direction_str = max(
            weighted_counts,
            key=weighted_counts.get,
        )
        dominant_direction = PredictionDirection(dominant_direction_str)
        
        # Calculate weighted agreement
        weighted_agreement = max_weight
        
        # Calculate probability statistics
        probabilities = [p.probability for p in predictions]
        average_probability = np.mean(probabilities)
        probability_std = np.std(probabilities)
        
        # Calculate probability consistency
        probability_consistency = max(0.0, 1.0 - probability_std)
        
        # Combine weighted direction agreement and probability consistency
        agreement_score = (
            weighted_agreement * 0.7 +
            probability_consistency * 0.3
        )
        
        # Determine agreement level
        agreement_level = self._get_agreement_level(agreement_score)
        
        participating_models = [p.model_name for p in predictions]
        
        self._logger.info(
            f"Weighted model agreement calculated: {agreement_level} "
            f"(score={agreement_score:.4f}, dominant={dominant_direction.value})"
        )
        
        return ModelAgreementResult(
            agreement_score=agreement_score,
            agreement_level=agreement_level,
            dominant_direction=dominant_direction,
            direction_counts={k: int(v * total_weight) for k, v in weighted_counts.items()},
            average_probability=average_probability,
            probability_std=probability_std,
            participating_models=participating_models,
        )


def calculate_model_agreement(
    predictions: List[ModelPrediction],
) -> ModelAgreementResult:
    """
    Convenience function to calculate model agreement.
    
    Args:
        predictions: List of ModelPrediction objects
        
    Returns:
        ModelAgreementResult
    """
    calculator = ModelAgreementCalculator()
    return calculator.calculate_agreement(predictions)
