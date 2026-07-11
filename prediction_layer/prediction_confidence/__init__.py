"""
Prediction Confidence Engine

Assesses confidence in model predictions by combining multiple factors:
- Model agreement between different ML models
- Signal agreement between different signal types
- Feature quality based on feature ranking
- Market regime match
- Historical similarity to past market conditions
- Data quality assessment
"""

from prediction_layer.prediction_confidence.model_agreement import ModelAgreementCalculator
from prediction_layer.prediction_confidence.signal_confidence import SignalConfidenceCalculator
from prediction_layer.prediction_confidence.feature_confidence import FeatureConfidenceCalculator
from prediction_layer.prediction_confidence.regime_confidence import RegimeConfidenceCalculator
from prediction_layer.prediction_confidence.historical_similarity import HistoricalSimilarityCalculator
from prediction_layer.prediction_confidence.confidence_score import ConfidenceScoreCalculator
from prediction_layer.prediction_confidence.confidence_score import ConfidenceResult

__all__ = [
    "ModelAgreementCalculator",
    "SignalConfidenceCalculator",
    "FeatureConfidenceCalculator",
    "RegimeConfidenceCalculator",
    "HistoricalSimilarityCalculator",
    "ConfidenceScoreCalculator",
    "ConfidenceResult",
]
