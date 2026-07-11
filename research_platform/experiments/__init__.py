"""
Experiment Tracker

The research memory of the entire quant platform.
Tracks every research attempt systematically.
"""

from research_platform.experiments.base import (
    # Enums
    ExperimentStatus,
    ExperimentPriority,
    ExperimentDecision,
    ExperimentType,
    ProjectType,
    # Data classes
    ResearchProject,
    Experiment,
    DatasetSnapshot,
    FeatureSnapshot,
    SignalSnapshot,
    AlphaSnapshot,
    Hyperparameters,
    TrainingMetrics,
    TradingMetrics,
    RegimePerformance,
    SectorPerformance,
    FeatureImportance,
    ResearchNote,
    LLMSummary,
)
from research_platform.experiments.project_manager import ProjectManager
from research_platform.experiments.experiment_runner import ExperimentRunner
from research_platform.experiments.dataset_version import DatasetVersionTracker
from research_platform.experiments.feature_snapshot import FeatureSnapshotManager
from research_platform.experiments.signal_snapshot import SignalSnapshotManager
from research_platform.experiments.alpha_snapshot import AlphaSnapshotManager
from research_platform.experiments.metrics_logger import MetricsLogger
from research_platform.experiments.regime_sector_performance import RegimeSectorPerformanceTracker
from research_platform.experiments.feature_importance import FeatureImportanceTracker
from research_platform.experiments.charts import ChartsGenerator
from research_platform.experiments.research_notes import ResearchNotesManager
from research_platform.experiments.llm_summary import LLMSummaryGenerator
from research_platform.experiments.comparison import ExperimentComparisonEngine
from research_platform.experiments.decision_engine import DecisionEngine
from research_platform.experiments.dashboard import ResearchDashboard

__all__ = [
    # Enums
    "ExperimentStatus",
    "ExperimentPriority",
    "ExperimentDecision",
    "ExperimentType",
    "ProjectType",
    # Data classes
    "ResearchProject",
    "Experiment",
    "DatasetSnapshot",
    "FeatureSnapshot",
    "SignalSnapshot",
    "AlphaSnapshot",
    "Hyperparameters",
    "TrainingMetrics",
    "TradingMetrics",
    "RegimePerformance",
    "SectorPerformance",
    "FeatureImportance",
    "ResearchNote",
    "LLMSummary",
    # Managers
    "ProjectManager",
    "ExperimentRunner",
    "DatasetVersionTracker",
    "FeatureSnapshotManager",
    "SignalSnapshotManager",
    "AlphaSnapshotManager",
    "MetricsLogger",
    "RegimeSectorPerformanceTracker",
    "FeatureImportanceTracker",
    "ChartsGenerator",
    "ResearchNotesManager",
    "LLMSummaryGenerator",
    "ExperimentComparisonEngine",
    "DecisionEngine",
    "ResearchDashboard",
]
