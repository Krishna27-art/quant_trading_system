"""
Prediction Learning Engine

Continuous learning system that improves predictions over time by:
- Storing prediction metadata and results
- Analyzing failures and successes
- Learning feature performance
- Learning signal performance
- Learning regime-specific performance
- Generating improvement suggestions
"""

from prediction_layer.prediction_learning.prediction_history import PredictionHistory
from prediction_layer.prediction_learning.prediction_result import PredictionResult
from prediction_layer.prediction_learning.failure_analysis import FailureAnalyzer
from prediction_layer.prediction_learning.weekly_report import WeeklyReportGenerator
from prediction_layer.prediction_learning.feature_learning import FeatureLearningEngine
from prediction_layer.prediction_learning.signal_learning import SignalLearningEngine
from prediction_layer.prediction_learning.regime_learning import RegimeLearningEngine
from prediction_layer.prediction_learning.improvement_suggestions import ImprovementSuggestionEngine

__all__ = [
    "PredictionHistory",
    "PredictionResult",
    "FailureAnalyzer",
    "WeeklyReportGenerator",
    "FeatureLearningEngine",
    "SignalLearningEngine",
    "RegimeLearningEngine",
    "ImprovementSuggestionEngine",
]
