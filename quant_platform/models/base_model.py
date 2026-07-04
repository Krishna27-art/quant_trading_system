"""
Abstract Base Class for Institutional Alpha Predictive Models

Enforces consistent interface across LightGBM, XGBoost, CatBoost, and Ensembles.
Requires probability outputs and feature importance extraction.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseAlphaModel(ABC):
    def __init__(self, model_id: str, name: str, version: str):
        self.model_id = model_id
        self.name = name
        self.version = version
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, X: List[Dict[str, Any]], y: List[int], feature_names: List[str]) -> None:
        """
        Train the predictive model on point-in-time features.
        y values should be binary (1 for alpha up, 0 for down/neutral) or multi-class.
        """
        pass

    @abstractmethod
    def predict_proba(self, X: List[Dict[str, Any]]) -> List[float]:
        """
        Returns calibrated probability P(y=1 | X) for each sample.
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Returns normalized feature importance (summing to 1.0 or raw gain).
        """
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        pass
