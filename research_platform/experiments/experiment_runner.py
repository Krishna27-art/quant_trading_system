"""
Experiment Runner with MLflow Integration

Runs experiments and tracks all metadata with MLflow.
"""

import os
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

from research_platform.experiments.base import (
    Experiment,
    ExperimentType,
    ExperimentStatus,
    ExperimentPriority,
    ExperimentDecision,
)
from utils.logger import get_logger

logger = get_logger("experiments.runner")


class ExperimentRunner:
    """
    Experiment Runner with MLflow integration.
    
    Manages:
    - Running experiments with MLflow tracking
    - Logging parameters, metrics, artifacts
    - Git commit tracking
    - Reproducibility
    """
    
    def __init__(
        self,
        mlflow_tracking_uri: Optional[str] = None,
        experiment_name: str = "quant_research",
    ):
        """
        Initialize experiment runner.
        
        Args:
            mlflow_tracking_uri: MLflow tracking URI
            experiment_name: Default MLflow experiment name
        """
        self.mlflow_tracking_uri = mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "file:///./mlruns")
        self.experiment_name = experiment_name
        self.experiments: Dict[str, Experiment] = {}
        
        # Configure MLflow
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        self._logger = get_logger("experiments.runner")
        self._logger.info(f"MLflow tracking URI: {self.mlflow_tracking_uri}")
    
    @contextmanager
    def run_experiment(
        self,
        experiment: Experiment,
        description: Optional[str] = None,
    ):
        """
        Context manager for running an experiment.
        
        Args:
            experiment: Experiment object
            description: Optional description for MLflow run
            
        Yields:
            MLflow active run
        """
        # Update experiment status
        experiment.status = ExperimentStatus.RUNNING
        self.experiments[experiment.experiment_id] = experiment
        
        # Start MLflow run
        with mlflow.start_run(
            run_name=f"{experiment.name}_{experiment.experiment_id}",
            description=description or experiment.purpose,
        ) as run:
            # Log experiment metadata
            mlflow.log_params({
                'experiment_id': experiment.experiment_id,
                'project_id': experiment.project_id,
                'experiment_type': experiment.experiment_type.value,
                'research_question': experiment.research_question,
                'created_by': experiment.created_by,
                'priority': experiment.priority.value,
                'tags': ','.join(experiment.tags),
            })
            
            self._logger.info(
                f"Started experiment {experiment.experiment_id}: {experiment.name}"
            )
            
            try:
                yield run
                
                # Mark as completed if no exception
                experiment.status = ExperimentStatus.COMPLETED
                self._logger.info(
                    f"Completed experiment {experiment.experiment_id}"
                )
                
            except Exception as e:
                # Mark as failed on exception
                experiment.status = ExperimentStatus.FAILED
                mlflow.log_param('error', str(e))
                self._logger.error(
                    f"Failed experiment {experiment.experiment_id}: {e}"
                )
                raise
    
    def log_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Log parameters to MLflow.
        
        Args:
            parameters: Dictionary of parameters
        """
        mlflow.log_params(parameters)
        self._logger.info(f"Logged {len(parameters)} parameters")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        mlflow.log_metrics(metrics, step=step)
        self._logger.info(f"Logged {len(metrics)} metrics")
    
    def log_training_metrics(
        self,
        accuracy: float,
        precision: float,
        recall: float,
        f1_score: float,
        roc_auc: float,
        log_loss: float,
        step: Optional[int] = None,
    ) -> None:
        """
        Log training metrics.
        
        Args:
            accuracy: Accuracy score
            precision: Precision score
            recall: Recall score
            f1_score: F1 score
            roc_auc: ROC AUC score
            log_loss: Log loss
            step: Optional step number
        """
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'roc_auc': roc_auc,
            'log_loss': log_loss,
        }
        self.log_metrics(metrics, step=step)
    
    def log_trading_metrics(
        self,
        win_rate: float,
        average_return: float,
        average_loss: float,
        profit_factor: float,
        sharpe_ratio: float,
        sortino_ratio: float,
        max_drawdown: float,
        expectancy: float,
        calmar_ratio: float,
        total_trades: int,
        step: Optional[int] = None,
    ) -> None:
        """
        Log trading metrics.
        
        Args:
            win_rate: Win rate
            average_return: Average return
            average_loss: Average loss
            profit_factor: Profit factor
            sharpe_ratio: Sharpe ratio
            sortino_ratio: Sortino ratio
            max_drawdown: Maximum drawdown
            expectancy: Expectancy
            calmar_ratio: Calmar ratio
            total_trades: Total trades
            step: Optional step number
        """
        metrics = {
            'win_rate': win_rate,
            'average_return': average_return,
            'average_loss': average_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'expectancy': expectancy,
            'calmar_ratio': calmar_ratio,
            'total_trades': total_trades,
        }
        self.log_metrics(metrics, step=step)
    
    def log_model(
        self,
        model: Any,
        model_type: str = "sklearn",
        artifact_path: str = "model",
    ) -> None:
        """
        Log model to MLflow.
        
        Args:
            model: Trained model
            model_type: Type of model (sklearn, xgboost, lightgbm)
            artifact_path: Artifact path for model
        """
        if model_type == "sklearn":
            mlflow.sklearn.log_model(model, artifact_path)
        elif model_type == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path)
        elif model_type == "lightgbm":
            mlflow.lightgbm.log_model(model, artifact_path)
        else:
            self._logger.warning(f"Unknown model type: {model_type}")
        
        self._logger.info(f"Logged {model_type} model")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log artifact to MLflow.
        
        Args:
            local_path: Local path to artifact
            artifact_path: Optional artifact path in MLflow
        """
        mlflow.log_artifact(local_path, artifact_path)
        self._logger.info(f"Logged artifact: {local_path}")
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        """
        Log directory of artifacts to MLflow.
        
        Args:
            local_dir: Local directory path
            artifact_path: Optional artifact path in MLflow
        """
        mlflow.log_artifacts(local_dir, artifact_path)
        self._logger.info(f"Logged artifacts from: {local_dir}")
    
    def log_dataset_info(
        self,
        dataset_name: str,
        version: str,
        rows: int,
        features: int,
        date_range: str,
        symbols: List[str],
    ) -> None:
        """
        Log dataset information.
        
        Args:
            dataset_name: Name of dataset
            version: Dataset version
            rows: Number of rows
            features: Number of features
            date_range: Date range
            symbols: List of symbols
        """
        params = {
            'dataset_name': dataset_name,
            'dataset_version': version,
            'dataset_rows': rows,
            'dataset_features': features,
            'dataset_date_range': date_range,
            'dataset_symbols': ','.join(symbols),
        }
        mlflow.log_params(params)
        self._logger.info(f"Logged dataset info: {dataset_name} v{version}")
    
    def log_feature_importance(
        self,
        feature_importance: Dict[str, float],
        method: str = "gain",
    ) -> None:
        """
        Log feature importance.
        
        Args:
            feature_importance: Dictionary of feature names to importance scores
            method: Method used (gain, shap, permutation)
        """
        # Log as parameters for top features

        top_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])
        
        for i, (feature, importance) in enumerate(top_features.items()):
            mlflow.log_param(f'feature_{i+1}', f"{feature}:{importance:.4f}")
        
        self._logger.info(f"Logged feature importance ({method})")
    
    def log_git_commit(self) -> None:
        """Log current git commit if available."""
        try:
            import subprocess
            
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
            )
            
            if result.returncode == 0:
                commit_hash = result.stdout.strip()
                mlflow.log_param('git_commit', commit_hash)
                self._logger.info(f"Logged git commit: {commit_hash}")
        except Exception as e:
            self._logger.warning(f"Could not log git commit: {e}")
    
    def log_environment_info(self) -> None:
        """Log environment information."""
        import sys
        import platform
        
        env_info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'mlflow_version': mlflow.__version__,
        }
        
        mlflow.log_params(env_info)
        self._logger.info("Logged environment info")
    
    def set_experiment_tags(self, tags: Dict[str, str]) -> None:
        """
        Set tags for the current run.
        
        Args:
            tags: Dictionary of tags
        """
        mlflow.set_tags(tags)
        self._logger.info(f"Set {len(tags)} tags")
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        return self.experiments.get(experiment_id)
    
    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
        experiment_type: Optional[ExperimentType] = None,
    ) -> List[Experiment]:
        """
        List experiments with optional filters.
        
        Args:
            status: Optional status filter
            experiment_type: Optional experiment type filter
            
        Returns:
            List of matching experiments
        """
        experiments = list(self.experiments.values())
        
        if status:
            experiments = [e for e in experiments if e.status == status]
        
        if experiment_type:
            experiments = [e for e in experiments if e.experiment_type == experiment_type]
        
        return experiments
    
    def update_experiment_decision(
        self,
        experiment_id: str,
        decision: ExperimentDecision,
    ) -> bool:
        """
        Update experiment decision.
        
        Args:
            experiment_id: Experiment ID
            decision: Decision made
            
        Returns:
            True if updated successfully
        """
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            self._logger.error(f"Experiment not found: {experiment_id}")
            return False
        
        experiment.decision = decision
        mlflow.log_param('decision', decision.value)
        
        self._logger.info(
            f"Updated experiment {experiment_id} decision to {decision.value}"
        )
        return True
