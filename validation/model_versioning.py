"""
Model Versioning System

Implements timestamped model directories instead of overwriting models/latest.pkl.
Tracks git commit, dataset hash, feature hash, and full lineage metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import hashlib
import subprocess
from pathlib import Path
import shutil

from utils.logger import get_logger

logger = get_logger("validation.model_versioning")


@dataclass
class ModelMetadata:
    """Metadata for a trained model."""
    model_version: str
    model_type: str  # logistic, random_forest, xgboost, etc.
    timeframe: str  # INTRADAY, SWING, LONGTERM
    direction: str  # long, short, or both
    training_date: datetime
    dataset_hash: str
    feature_hash: str
    git_commit: Optional[str]
    git_branch: Optional[str]
    python_version: str
    feature_schema_version: str
    hyperparameters: Dict[str, Any]
    training_metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]]
    model_path: str
    metadata_path: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "model_version": self.model_version,
            "model_type": self.model_type,
            "timeframe": self.timeframe,
            "direction": self.direction,
            "training_date": self.training_date.isoformat(),
            "dataset_hash": self.dataset_hash,
            "feature_hash": self.feature_hash,
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "python_version": self.python_version,
            "feature_schema_version": self.feature_schema_version,
            "hyperparameters": self.hyperparameters,
            "training_metrics": self.training_metrics,
            "feature_importance": self.feature_importance,
            "model_path": self.model_path,
            "metadata_path": self.metadata_path,
        }


class ModelVersionManager:
    """
    Manages model versioning with timestamped directories.
    
    Structure:
    models/
        2026-07-09_14-30-00/
            model.pkl
            metadata.json
            metrics.json
            feature_hash.json
            dataset_hash.json
            config.json
        2026-07-09_15-45-30/
            ...
    """
    
    def __init__(self, base_path: str = "models/saved"):
        """
        Initialize model version manager.
        
        Args:
            base_path: Base directory for model storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def get_version_directory(self, version: Optional[str] = None) -> Path:
        """
        Get directory for a model version.
        
        Args:
            version: Version string (if None, generates timestamp)
            
        Returns:
            Path to version directory
        """
        if version:
            version_dir = self.base_path / version
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            version_dir = self.base_path / timestamp
        
        version_dir.mkdir(parents=True, exist_ok=True)
        return version_dir
    
    def save_model(
        self,
        model: Any,
        metadata: ModelMetadata,
        version: Optional[str] = None,
    ) -> str:
        """
        Save a model with full metadata.
        
        Args:
            model: Trained model object
            metadata: Model metadata
            version: Version string (if None, uses timestamp)
            
        Returns:
            Version string used
        """
        version_dir = self.get_version_directory(version)
        version = version_dir.name
        
        # Save model
        model_path = version_dir / "model.pkl"
        import joblib
        joblib.dump(model, model_path)
        
        # Update metadata paths
        metadata.model_path = str(model_path)
        metadata.metadata_path = str(version_dir / "metadata.json")
        metadata.model_version = version
        
        # Save metadata
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Save metrics separately
        metrics_path = version_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metadata.training_metrics, f, indent=2)
        
        # Save hyperparameters
        config_path = version_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(metadata.hyperparameters, f, indent=2)
        
        # Save feature hash
        feature_hash_path = version_dir / "feature_hash.json"
        with open(feature_hash_path, "w") as f:
            json.dump({"feature_hash": metadata.feature_hash}, f, indent=2)
        
        # Save dataset hash
        dataset_hash_path = version_dir / "dataset_hash.json"
        with open(dataset_hash_path, "w") as f:
            json.dump({"dataset_hash": metadata.dataset_hash}, f, indent=2)
        
        logger.info(f"Saved model version {version} to {version_dir}")
        
        # Update latest symlink
        self._update_latest_symlink(version_dir)
        
        return version
    
    def load_model(self, version: str) -> tuple[Any, ModelMetadata]:
        """
        Load a model by version.
        
        Args:
            version: Version string
            
        Returns:
            Tuple of (model, metadata)
        """
        version_dir = self.base_path / version
        
        if not version_dir.exists():
            raise FileNotFoundError(f"Model version {version} not found")
        
        # Load model
        model_path = version_dir / "model.pkl"
        import joblib
        model = joblib.load(model_path)
        
        # Load metadata
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata_dict = json.load(f)
        
        metadata = ModelMetadata(
            model_version=metadata_dict["model_version"],
            model_type=metadata_dict["model_type"],
            timeframe=metadata_dict["timeframe"],
            direction=metadata_dict["direction"],
            training_date=datetime.fromisoformat(metadata_dict["training_date"]),
            dataset_hash=metadata_dict["dataset_hash"],
            feature_hash=metadata_dict["feature_hash"],
            git_commit=metadata_dict.get("git_commit"),
            git_branch=metadata_dict.get("git_branch"),
            python_version=metadata_dict["python_version"],
            feature_schema_version=metadata_dict["feature_schema_version"],
            hyperparameters=metadata_dict["hyperparameters"],
            training_metrics=metadata_dict["training_metrics"],
            feature_importance=metadata_dict.get("feature_importance"),
            model_path=metadata_dict["model_path"],
            metadata_path=metadata_dict["metadata_path"],
        )
        
        logger.info(f"Loaded model version {version}")
        
        return model, metadata
    
    def load_latest(self) -> tuple[Any, ModelMetadata]:
        """
        Load the latest model version.
        
        Returns:
            Tuple of (model, metadata)
        """
        latest_path = self.base_path / "latest"
        
        if not latest_path.exists() or not latest_path.is_symlink():
            # Find the most recent version directory
            versions = [d for d in self.base_path.iterdir() if d.is_dir() and d.name != "latest"]
            if not versions:
                raise FileNotFoundError("No model versions found")
            latest_dir = max(versions, key=lambda x: x.stat().st_mtime)
        else:
            latest_dir = latest_path.resolve()
        
        version = latest_dir.name
        return self.load_model(version)
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """
        List all available model versions.
        
        Returns:
            List of version info dictionaries
        """
        versions = []
        
        for version_dir in self.base_path.iterdir():
            if not version_dir.is_dir() or version_dir.name == "latest":
                continue
            
            metadata_path = version_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                
                versions.append({
                    "version": version_dir.name,
                    "model_type": metadata.get("model_type"),
                    "timeframe": metadata.get("timeframe"),
                    "direction": metadata.get("direction"),
                    "training_date": metadata.get("training_date"),
                    "dataset_hash": metadata.get("dataset_hash"),
                    "feature_hash": metadata.get("feature_hash"),
                })
        
        # Sort by training date descending
        versions.sort(key=lambda x: x["training_date"], reverse=True)
        
        return versions
    
    def _update_latest_symlink(self, target_dir: Path) -> None:
        """Update the 'latest' symlink to point to the target directory."""
        latest_path = self.base_path / "latest"
        
        # Remove existing symlink
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        
        # Create new symlink
        latest_path.symlink_to(target_dir)
        logger.debug(f"Updated latest symlink to {target_dir}")


def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        logger.warning(f"Failed to get git commit: {e}")
    return None


def get_git_branch() -> Optional[str]:
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        logger.warning(f"Failed to get git branch: {e}")
    return None


def get_python_version() -> str:
    """Get Python version."""
    import sys
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def create_model_metadata(
    model_type: str,
    timeframe: str,
    direction: str,
    dataset_hash: str,
    feature_hash: str,
    hyperparameters: Dict[str, Any],
    training_metrics: Dict[str, float],
    feature_importance: Optional[Dict[str, float]] = None,
    feature_schema_version: str = "v1.0",
) -> ModelMetadata:
    """
    Create model metadata with git information.
    
    Args:
        model_type: Type of model
        timeframe: Timeframe (INTRADAY, SWING, LONGTERM)
        direction: Direction (long, short, both)
        dataset_hash: Hash of training dataset
        feature_hash: Hash of feature list
        hyperparameters: Model hyperparameters
        training_metrics: Training metrics
        feature_importance: Feature importance scores
        feature_schema_version: Feature schema version
        
    Returns:
        ModelMetadata
    """
    return ModelMetadata(
        model_version="",  # Will be set when saved
        model_type=model_type,
        timeframe=timeframe,
        direction=direction,
        training_date=datetime.now(),
        dataset_hash=dataset_hash,
        feature_hash=feature_hash,
        git_commit=get_git_commit(),
        git_branch=get_git_branch(),
        python_version=get_python_version(),
        feature_schema_version=feature_schema_version,
        hyperparameters=hyperparameters,
        training_metrics=training_metrics,
        feature_importance=feature_importance,
        model_path="",
        metadata_path="",
    )
