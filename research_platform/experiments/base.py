"""
Experiment Tracker Base Classes

Defines core data structures for experiment tracking.
This is the research memory of the entire platform.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger("experiments.base")


class ExperimentStatus(Enum):
    """Status of an experiment."""
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    ARCHIVED = "ARCHIVED"


class ExperimentPriority(Enum):
    """Priority of an experiment."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ExperimentDecision(Enum):
    """Decision made after experiment completion."""
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    NEEDS_MORE_TESTING = "NEEDS_MORE_TESTING"
    MERGED = "MERGED"
    ARCHIVED = "ARCHIVED"


class ExperimentType(Enum):
    """Type of experiment."""
    MODEL_TRAINING = "MODEL_TRAINING"
    FEATURE_RESEARCH = "FEATURE_RESEARCH"
    SIGNAL_RESEARCH = "SIGNAL_RESEARCH"
    ALPHA_RESEARCH = "ALPHA_RESEARCH"
    BACKTESTING = "BACKTESTING"
    REGIME_RESEARCH = "REGIME_RESEARCH"
    RISK_RESEARCH = "RISK_RESEARCH"
    HYPERPARAMETER_TUNING = "HYPERPARAMETER_TUNING"
    LLM_RESEARCH = "LLM_RESEARCH"


class ProjectType(Enum):
    """Type of research project."""
    INTRADAY = "INTRADAY"
    SWING = "SWING"
    LONGTERM = "LONGTERM"
    FEATURE_RESEARCH = "FEATURE_RESEARCH"
    MODEL_RESEARCH = "MODEL_RESEARCH"
    ALPHA_RESEARCH = "ALPHA_RESEARCH"
    SIGNAL_RESEARCH = "SIGNAL_RESEARCH"
    RISK_RESEARCH = "RISK_RESEARCH"


@dataclass
class ResearchProject:
    """
    Research project container.
    
    Groups related experiments together.
    """
    project_id: str
    name: str
    project_type: ProjectType
    description: str
    created_by: str
    created_at: datetime = field(default_factory=datetime.now)
    status: ExperimentStatus = ExperimentStatus.CREATED
    priority: ExperimentPriority = ExperimentPriority.MEDIUM
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'project_id': self.project_id,
            'name': self.name,
            'project_type': self.project_type.value,
            'description': self.description,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'status': self.status.value,
            'priority': self.priority.value,
            'tags': self.tags,
            'notes': self.notes,
        }


@dataclass
class Experiment:
    """
    Core experiment data structure.
    
    Every research attempt is an experiment.
    """
    experiment_id: str
    project_id: str
    name: str
    experiment_type: ExperimentType
    purpose: str
    research_question: str
    created_by: str
    created_at: datetime = field(default_factory=datetime.now)
    status: ExperimentStatus = ExperimentStatus.CREATED
    priority: ExperimentPriority = ExperimentPriority.MEDIUM
    decision: Optional[ExperimentDecision] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'experiment_id': self.experiment_id,
            'project_id': self.project_id,
            'name': self.name,
            'experiment_type': self.experiment_type.value,
            'purpose': self.purpose,
            'research_question': self.research_question,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'status': self.status.value,
            'priority': self.priority.value,
            'decision': self.decision.value if self.decision else None,
            'tags': self.tags,
            'notes': self.notes,
        }


@dataclass
class DatasetSnapshot:
    """
    Snapshot of dataset used in experiment.
    
    Ensures reproducibility by tracking exact data version.
    """
    snapshot_id: str
    experiment_id: str
    dataset_name: str
    version: str
    rows: int
    features: int
    date_range: str
    symbols: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'snapshot_id': self.snapshot_id,
            'experiment_id': self.experiment_id,
            'dataset_name': self.dataset_name,
            'version': self.version,
            'rows': self.rows,
            'features': self.features,
            'date_range': self.date_range,
            'symbols': self.symbols,
            'created_at': self.created_at.isoformat(),
        }


@dataclass
class FeatureSnapshot:
    """
    Snapshot of features used in experiment.
    
    Tracks exactly which features were used.
    """
    snapshot_id: str
    experiment_id: str
    feature_names: List[str]
    feature_count: int
    feature_types: Dict[str, str]
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'snapshot_id': self.snapshot_id,
            'experiment_id': self.experiment_id,
            'feature_names': self.feature_names,
            'feature_count': self.feature_count,
            'feature_types': self.feature_types,
            'created_at': self.created_at.isoformat(),
        }


@dataclass
class SignalSnapshot:
    """
    Snapshot of signal configuration used in experiment.
    
    Tracks signal versions and weights.
    """
    snapshot_id: str
    experiment_id: str
    signal_versions: Dict[str, str]
    signal_weights: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'snapshot_id': self.snapshot_id,
            'experiment_id': self.experiment_id,
            'signal_versions': self.signal_versions,
            'signal_weights': self.signal_weights,
            'created_at': self.created_at.isoformat(),
        }


@dataclass
class AlphaSnapshot:
    """
    Snapshot of alpha configuration used in experiment.
    
    Tracks alpha weights and rules.
    """
    snapshot_id: str
    experiment_id: str
    alpha_version: str
    alpha_weights: Dict[str, float]
    alpha_rules: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'snapshot_id': self.snapshot_id,
            'experiment_id': self.experiment_id,
            'alpha_version': self.alpha_version,
            'alpha_weights': self.alpha_weights,
            'alpha_rules': self.alpha_rules,
            'created_at': self.created_at.isoformat(),
        }


@dataclass
class Hyperparameters:
    """
    Hyperparameters used in experiment.
    
    Stores all model and training parameters.
    """
    hyperparameters_id: str
    experiment_id: str
    parameters: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'hyperparameters_id': self.hyperparameters_id,
            'experiment_id': self.experiment_id,
            'parameters': self.parameters,
            'created_at': self.created_at.isoformat(),
        }


@dataclass
class TrainingMetrics:
    """
    Training metrics from model training.
    
    Standard ML metrics for model evaluation.
    """
    metrics_id: str
    experiment_id: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    log_loss: float
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'metrics_id': self.metrics_id,
            'experiment_id': self.experiment_id,
            'accuracy': round(self.accuracy, 4),
            'precision': round(self.precision, 4),
            'recall': round(self.recall, 4),
            'f1_score': round(self.f1_score, 4),
            'roc_auc': round(self.roc_auc, 4),
            'log_loss': round(self.log_loss, 4),
            'created_at': self.created_at.isoformat(),
        }


@dataclass
class TradingMetrics:
    """
    Trading metrics from backtesting.
    
    These matter more for trading than ML metrics.
    """
    metrics_id: str
    experiment_id: str
    win_rate: float
    average_return: float
    average_loss: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    expectancy: float
    calmar_ratio: float
    total_trades: int
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'metrics_id': self.metrics_id,
            'experiment_id': self.experiment_id,
            'win_rate': round(self.win_rate, 4),
            'average_return': round(self.average_return, 4),
            'average_loss': round(self.average_loss, 4),
            'profit_factor': round(self.profit_factor, 4),
            'sharpe_ratio': round(self.sharpe_ratio, 4),
            'sortino_ratio': round(self.sortino_ratio, 4),
            'max_drawdown': round(self.max_drawdown, 4),
            'expectancy': round(self.expectancy, 4),
            'calmar_ratio': round(self.calmar_ratio, 4),
            'total_trades': self.total_trades,
            'created_at': self.created_at.isoformat(),
        }


@dataclass
class RegimePerformance:
    """
    Performance by market regime.
    
    Shows where the model works best.
    """
    performance_id: str
    experiment_id: str
    regime: str
    win_rate: float
    sharpe_ratio: float
    total_trades: int
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'performance_id': self.performance_id,
            'experiment_id': self.experiment_id,
            'regime': self.regime,
            'win_rate': round(self.win_rate, 4),
            'sharpe_ratio': round(self.sharpe_ratio, 4),
            'total_trades': self.total_trades,
            'created_at': self.created_at.isoformat(),
        }


@dataclass
class SectorPerformance:
    """
    Performance by sector.
    
    Shows sector-specific performance.
    """
    performance_id: str
    experiment_id: str
    sector: str
    win_rate: float
    sharpe_ratio: float
    total_trades: int
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'performance_id': self.performance_id,
            'experiment_id': self.experiment_id,
            'sector': self.sector,
            'win_rate': round(self.win_rate, 4),
            'sharpe_ratio': round(self.sharpe_ratio, 4),
            'total_trades': self.total_trades,
            'created_at': self.created_at.isoformat(),
        }


@dataclass
class FeatureImportance:
    """
    Feature importance from experiment.
    
    Tracks top and worst features.
    """
    importance_id: str
    experiment_id: str
    top_features: Dict[str, float]
    worst_features: Dict[str, float]
    method: str  # 'shap', 'permutation', 'gain'
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'importance_id': self.importance_id,
            'experiment_id': self.experiment_id,
            'top_features': {k: round(v, 4) for k, v in self.top_features.items()},
            'worst_features': {k: round(v, 4) for k, v in self.worst_features.items()},
            'method': self.method,
            'created_at': self.created_at.isoformat(),
        }


@dataclass
class ResearchNote:
    """
    Research notes for experiment.
    
    Captures learnings and conclusions.
    """
    note_id: str
    experiment_id: str
    author: str
    content: str
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'note_id': self.note_id,
            'experiment_id': self.experiment_id,
            'author': self.author,
            'content': self.content,
            'created_at': self.created_at.isoformat(),
        }


@dataclass
class LLMSummary:
    """
    LLM-generated summary of experiment results.
    
    Provides AI-powered insights.
    """
    summary_id: str
    experiment_id: str
    best_model: str
    best_alpha: str
    weakest_feature: str
    recommendation: str
    insights: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'summary_id': self.summary_id,
            'experiment_id': self.experiment_id,
            'best_model': self.best_model,
            'best_alpha': self.best_alpha,
            'weakest_feature': self.weakest_feature,
            'recommendation': self.recommendation,
            'insights': self.insights,
            'created_at': self.created_at.isoformat(),
        }
