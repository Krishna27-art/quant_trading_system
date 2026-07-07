"""
Model Registry

Central registry for all prediction models.
Manages model lifecycle: registration, loading, versioning, and retrieval.
"""

from typing import Dict, Type
import json
from pathlib import Path
from .base_model import BaseModel, ModelMetadata


class ModelRegistry:
    """
    Central registry for managing prediction models.
    
    Features:
    - Register models by name and version
    - Load models from disk
    - Track model metadata
    - Version control
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._registered_models: Dict[str, Type[BaseModel]] = {}
        self._loaded_models: Dict[str, BaseModel] = {}
    
    def register_model_class(self, name: str, model_class: Type[BaseModel]) -> None:
        """
        Register a model class for instantiation.
        
        Args:
            name: Model name (e.g., "lightgbm", "xgboost")
            model_class: Model class implementing BaseModel interface
        """
        self._registered_models[name] = model_class
    
    def load_model(self, name: str, version: str = "latest") -> BaseModel:
        """
        Load a model from disk.
        
        Args:
            name: Model name
            version: Model version (default: latest)
            
        Returns:
            Loaded model instance
        """
        cache_key = f"{name}_{version}"
        
        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]
        
        if version == "latest":
            version = self._get_latest_version(name)
        
        model_path = self.models_dir / name / version / "model.pkl"
        metadata_path = self.models_dir / name / version / "metadata.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        metadata = ModelMetadata(**metadata_dict)
        
        # Instantiate and load model
        model_class = self._registered_models.get(name)
        if model_class is None:
            raise ValueError(f"Model class not registered: {name}")
        
        model = model_class(metadata=metadata)
        model.load(str(model_path))
        
        self._loaded_models[cache_key] = model
        return model
    
    def save_model(self, model: BaseModel, name: str, version: str) -> None:
        """
        Save a model to disk with metadata.
        
        Args:
            model: Model instance to save
            name: Model name
            version: Model version
        """
        model_dir = self.models_dir / name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / "model.pkl"
        metadata_path = model_dir / "metadata.json"
        
        # Save model
        model.save(str(model_path))
        
        # Save metadata
        if model.metadata:
            with open(metadata_path, 'w') as f:
                json.dump(model.metadata.__dict__, f, indent=2, default=str)
    
    def list_models(self) -> Dict[str, list]:
        """
        List all available models and their versions.
        
        Returns:
            Dictionary mapping model names to list of versions
        """
        models = {}
        for name_dir in self.models_dir.iterdir():
            if name_dir.is_dir():
                versions = [v.name for v in name_dir.iterdir() if v.is_dir()]
                models[name_dir.name] = sorted(versions, reverse=True)
        return models
    
    def _get_latest_version(self, name: str) -> str:
        """Get the latest version for a model."""
        model_dir = self.models_dir / name
        if not model_dir.exists():
            raise ValueError(f"Model not found: {name}")
        
        versions = [v.name for v in model_dir.iterdir() if v.is_dir()]
        if not versions:
            raise ValueError(f"No versions found for model: {name}")
        
        return sorted(versions, reverse=True)[0]
    
    def create_version(self, name: str) -> str:
        """
        Create a new version string for a model.
        
        Args:
            name: Model name
            
        Returns:
            New version string (e.g., "v1.0.0")
        """
        model_dir = self.models_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        versions = [v.name for v in model_dir.iterdir() if v.is_dir()]
        if not versions:
            return "v1.0.0"
        
        # Parse latest version and increment
        latest = sorted(versions, reverse=True)[0]
        major, minor, patch = map(int, latest.replace('v', '').split('.'))
        return f"v{major}.{minor}.{patch + 1}"


# Global registry instance
registry = ModelRegistry()
