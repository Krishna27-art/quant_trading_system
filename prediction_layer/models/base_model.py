"""
Unified Model Interface

All prediction models must implement this interface.
This ensures consistency across LightGBM, XGBoost, CatBoost, LSTM, Transformer, etc.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class ModelMetadata:
    """Metadata for model versioning and tracking."""
    model_name: str
    model_version: str
    feature_version: str
    dataset_version: str
    training_date: str
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]


class BaseModel(ABC):
    """
    Abstract base class for all prediction models.
    
    Every model must implement:
    - fit(X, y): Train the model
    - predict(X): Return class predictions (BUY/SELL/HOLD)
    - predict_proba(X): Return probability estimates
    - save(path): Persist model to disk
    - load(path): Load model from disk
    """
    
    def __init__(self, metadata: Optional[ModelMetadata] = None):
        self.metadata = metadata
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model on features X and labels y.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target labels (0=SELL, 1=HOLD, 2=BUY or similar encoding)
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return class predictions.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predictions array of shape (n_samples,)
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return probability estimates for each class.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Probability array of shape (n_samples, n_classes)
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Persist model to disk.
        
        Args:
            path: File path to save the model
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load model from disk.
        
        Args:
            path: File path to load the model from
        """
        pass
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Return feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        return {}
    
    def get_metadata(self) -> ModelMetadata:
        """Return model metadata."""
        return self.metadata
