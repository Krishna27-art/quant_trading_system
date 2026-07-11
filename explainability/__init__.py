"""
Explainability Module

Provides comprehensive explainability for predictions.
Explains why predictions are made using features, signals, alpha, confidence,
and historical similarity.

Main Components:
- FeatureExplainer: Explain top contributing features
- SignalExplainer: Explain which signals fired with ratings
- AlphaExplainer: Explain alpha score composition
- ConfidenceExplainer: Explain confidence score breakdown
- HistoricalSimilarityFinder: Find similar historical setups
- PredictionReportGenerator: Generate human-readable reports
"""

from explainability.feature_explainer import (
    FeatureExplainer,
    FeatureContribution,
    explain_features,
)
from explainability.signal_explainer import (
    SignalExplainer,
    SignalRating,
    explain_signals,
)
from explainability.alpha_explainer import (
    AlphaExplainer,
    AlphaComponent,
    explain_alpha,
)
from explainability.confidence_explainer import (
    ConfidenceExplainer,
    ConfidenceComponent,
    explain_confidence,
)
from explainability.historical_similarity import (
    HistoricalSimilarityFinder,
    SimilarSetup,
    find_historical_similarity,
)
from explainability.prediction_report import (
    PredictionReportGenerator,
    generate_prediction_report,
)

__all__ = [
    # Feature Explanation
    "FeatureExplainer",
    "FeatureContribution",
    "explain_features",
    # Signal Explanation
    "SignalExplainer",
    "SignalRating",
    "explain_signals",
    # Alpha Explanation
    "AlphaExplainer",
    "AlphaComponent",
    "explain_alpha",
    # Confidence Explanation
    "ConfidenceExplainer",
    "ConfidenceComponent",
    "explain_confidence",
    # Historical Similarity
    "HistoricalSimilarityFinder",
    "SimilarSetup",
    "find_historical_similarity",
    # Prediction Reports
    "PredictionReportGenerator",
    "generate_prediction_report",
]
