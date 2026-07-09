"""
Ensemble Engine

Combines predictions from multiple independent models.
Aggregates evidence rather than blindly averaging predictions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger("signal_engine.ensemble")


@dataclass
class ModelPrediction:
    """Prediction from a single model."""
    model_name: str
    prediction: float  # 0 to 1 probability
    confidence: float  # 0 to 1 confidence
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "prediction": round(self.prediction, 4),
            "confidence": round(self.confidence, 4),
        }


@dataclass
class EnsembleResult:
    """Result of ensemble combination."""
    combined_prediction: float
    combined_confidence: float
    model_agreement: float
    individual_predictions: List[ModelPrediction]
    dominant_model: Optional[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "combined_prediction": round(self.combined_prediction, 4),
            "combined_confidence": round(self.combined_confidence, 4),
            "model_agreement": round(self.model_agreement, 4),
            "individual_predictions": [p.to_dict() for p in self.individual_predictions],
            "dominant_model": self.dominant_model,
        }


class EnsembleEngine:
    """
    Combines predictions from multiple independent models.
    
    Models:
    - Tree models (XGBoost, LightGBM)
    - Neural networks (if available)
    - Statistical models
    - Factor models
    - Rule engines
    """
    
    def __init__(
        self,
        model_weights: Optional[Dict[str, float]] = None,
        combination_method: str = "weighted_average",
    ):
        """
        Initialize ensemble engine.
        
        Args:
            model_weights: Optional weights for each model
            combination_method: Method for combining predictions
        """
        self.model_weights = model_weights or {}
        self.combination_method = combination_method
        self._logger = get_logger("signal_engine.ensemble")
    
    def combine_predictions(
        self,
        predictions: List[ModelPrediction],
    ) -> EnsembleResult:
        """
        Combine predictions from multiple models.
        
        Args:
            predictions: List of ModelPrediction
            
        Returns:
            EnsembleResult
        """
        if not predictions:
            return EnsembleResult(
                combined_prediction=0.5,
                combined_confidence=0.0,
                model_agreement=0.0,
                individual_predictions=[],
                dominant_model=None,
            )
        
        # Calculate combined prediction based on method
        if self.combination_method == "weighted_average":
            combined_pred = self._weighted_average(predictions)
        elif self.combination_method == "median":
            combined_pred = self._median(predictions)
        elif self.combination_method == "max":
            combined_pred = self._max(predictions)
        elif self.combination_method == "min":
            combined_pred = self._min(predictions)
        else:
            combined_pred = self._weighted_average(predictions)
        
        # Calculate combined confidence
        combined_conf = self._calculate_combined_confidence(predictions)
        
        # Calculate model agreement
        model_agreement = self._calculate_agreement(predictions)
        
        # Determine dominant model
        dominant_model = self._find_dominant_model(predictions)
        
        return EnsembleResult(
            combined_prediction=combined_pred,
            combined_confidence=combined_conf,
            model_agreement=model_agreement,
            individual_predictions=predictions,
            dominant_model=dominant_model,
        )
    
    def _weighted_average(self, predictions: List[ModelPrediction]) -> float:
        """Calculate weighted average of predictions."""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for pred in predictions:
            weight = self.model_weights.get(pred.model_name, 1.0)
            weighted_sum += pred.prediction * weight * pred.confidence
            total_weight += weight * pred.confidence
        
        if total_weight == 0:
            return 0.5
        
        return weighted_sum / total_weight
    
    def _median(self, predictions: List[ModelPrediction]) -> float:
        """Calculate median of predictions."""
        values = [p.prediction for p in predictions]
        return np.median(values)
    
    def _max(self, predictions: List[ModelPrediction]) -> float:
        """Calculate maximum of predictions."""
        return max(p.prediction for p in predictions)
    
    def _min(self, predictions: List[ModelPrediction]) -> float:
        """Calculate minimum of predictions."""
        return min(p.prediction for p in predictions)
    
    def _calculate_combined_confidence(self, predictions: List[ModelPrediction]) -> float:
        """Calculate combined confidence from individual confidences."""
        if not predictions:
            return 0.0
        
        confidences = [p.confidence for p in predictions]
        
        # Average confidence
        avg_confidence = np.mean(confidences)
        
        # Boost confidence if models agree
        agreement = self._calculate_agreement(predictions)
        confidence_boost = agreement * 0.2
        
        combined_conf = min(1.0, avg_confidence + confidence_boost)
        
        return combined_conf
    
    def _calculate_agreement(self, predictions: List[ModelPrediction]) -> float:
        """Calculate agreement among model predictions."""
        if not predictions:
            return 0.0
        
        values = [p.prediction for p in predictions]
        
        # Calculate standard deviation
        std_dev = np.std(values)
        
        # Agreement is inverse of dispersion
        agreement = max(0.0, 1.0 - std_dev)
        
        return agreement
    
    def _find_dominant_model(self, predictions: List[ModelPrediction]) -> Optional[str]:
        """Find the model with highest confidence."""
        if not predictions:
            return None
        
        dominant = max(predictions, key=lambda p: p.confidence)
        return dominant.model_name
    
    def add_model_weight(self, model_name: str, weight: float) -> None:
        """
        Add or update weight for a model.
        
        Args:
            model_name: Name of model
            weight: Weight for model
        """
        self.model_weights[model_name] = weight
        self._logger.info(f"Updated weight for {model_name}: {weight}")


def combine_ensemble_predictions(
    predictions: List[ModelPrediction],
    model_weights: Optional[Dict[str, float]] = None,
    combination_method: str = "weighted_average",
) -> EnsembleResult:
    """
    Convenience function to combine ensemble predictions.
    
    Args:
        predictions: List of ModelPrediction
        model_weights: Optional weights for each model
        combination_method: Method for combining predictions
        
    Returns:
        EnsembleResult
    """
    engine = EnsembleEngine(model_weights=model_weights, combination_method=combination_method)
    return engine.combine_predictions(predictions)
