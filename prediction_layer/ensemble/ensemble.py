"""
ML Ensemble

Combines multiple models into a meta-learner.
Implements stacking ensemble for improved prediction accuracy.
"""

from typing import Dict, List, Optional
import numpy as np
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from .models.base_model import BaseModel


class EnsembleModel(BaseModel):
    """
    Ensemble model that combines multiple base models.
    
    Supports:
    - Stacking ensemble (meta-learner)
    - Voting ensemble (soft/hard voting)
    """
    
    def __init__(
        self,
        base_models: Dict[str, BaseModel],
        method: str = "stacking",
        metadata: Optional = None
    ):
        """
        Initialize ensemble.
        
        Args:
            base_models: Dictionary of model_name -> BaseModel instance
            method: Ensemble method ('stacking' or 'voting')
            metadata: Model metadata
        """
        super().__init__(metadata)
        self.base_models = base_models
        self.method = method
        self.ensemble = None
        self.meta_learner = LogisticRegression(max_iter=1000)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the ensemble on training data.
        
        Args:
            X: Feature matrix
            y: Target labels
        """
        # Fit all base models
        for name, model in self.base_models.items():
            model.fit(X, y)
        
        # Create sklearn estimators from base models
        estimators = [
            (name, self._wrap_model(model))
            for name, model in self.base_models.items()
        ]
        
        # Create ensemble
        if self.method == "stacking":
            self.ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=self.meta_learner,
                cv=5
            )
        else:  # voting
            self.ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft'
            )
        
        # Fit ensemble
        self.ensemble.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        return self.ensemble.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        return self.ensemble.predict_proba(X)
    
    def save(self, path: str) -> None:
        """Save ensemble to disk."""
        import joblib
        joblib.dump({
            'ensemble': self.ensemble,
            'base_models': self.base_models,
            'method': self.method,
            'metadata': self.metadata
        }, path)
    
    def load(self, path: str) -> None:
        """Load ensemble from disk."""
        import joblib
        data = joblib.load(path)
        self.ensemble = data['ensemble']
        self.base_models = data['base_models']
        self.method = data['method']
        self.metadata = data['metadata']
        self.is_fitted = True
    
    def _wrap_model(self, model: BaseModel):
        """Wrap BaseModel to sklearn interface."""
        class ModelWrapper:
            def __init__(self, model):
                self.model = model
            
            def fit(self, X, y):
                self.model.fit(X, y)
                return self
            
            def predict(self, X):
                return self.model.predict(X)
            
            def predict_proba(self, X):
                return self.model.predict_proba(X)
        
        return ModelWrapper(model)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get aggregated feature importance from base models."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted first")
        
        # Aggregate importance from all base models
        all_importance = {}
        for name, model in self.base_models.items():
            try:
                imp = model.get_feature_importance()
                for feat, val in imp.items():
                    if feat not in all_importance:
                        all_importance[feat] = []
                    all_importance[feat].append(val)
            except Exception:
                continue
        
        # Average across models
        avg_importance = {
            feat: np.mean(vals)
            for feat, vals in all_importance.items()
        }
        
        return avg_importance
