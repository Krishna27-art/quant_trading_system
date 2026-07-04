"""
SHAP Explainability

Provides model explainability using SHAP values.
Shows which features contributed most to each prediction.
"""

from typing import Dict, List, Optional
import numpy as np
import shap
from pathlib import Path
import joblib


class SHAPExplainer:
    """
    SHAP-based model explainability.
    
    Provides:
    - Feature importance for individual predictions
    - Global feature importance
    - Summary plots
    - Force plots
    """
    
    def __init__(self, model, feature_names: List[str], background_data: Optional[np.ndarray] = None):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model (must implement predict or predict_proba)
            feature_names: List of feature names
            background_data: Background dataset for TreeExplainer (optional)
        """
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self.explainer = None
        
        # Initialize explainer based on model type
        self._init_explainer()
    
    def _init_explainer(self):
        """Initialize appropriate SHAP explainer."""
        try:
            # Try TreeExplainer first (works for LightGBM, XGBoost, CatBoost)
            if self.background_data is not None:
                self.explainer = shap.TreeExplainer(self.model, data=self.background_data)
            else:
                self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            try:
                # Fall back to KernelExplainer (model-agnostic)
                if self.background_data is not None:
                    self.explainer = shap.KernelExplainer(self.model.predict, self.background_data[:100])
                else:
                    raise ValueError("Background data required for KernelExplainer")
            except Exception as e:
                raise ValueError(f"Could not initialize SHAP explainer: {e}")
    
    def explain_instance(self, X: np.ndarray, index: int = 0) -> Dict[str, float]:
        """
        Explain a single prediction.
        
        Args:
            X: Feature matrix
            index: Index of instance to explain
            
        Returns:
            Dictionary mapping feature names to SHAP values
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized")
        
        shap_values = self.explainer.shap_values(X[index:index+1])
        
        if isinstance(shap_values, list):
            # Multi-class output - use first class
            shap_values = shap_values[0]
        
        shap_values = shap_values.flatten()
        
        return dict(zip(self.feature_names, shap_values))
    
    def explain_batch(self, X: np.ndarray) -> List[Dict[str, float]]:
        """
        Explain multiple predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            List of feature importance dictionaries
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized")
        
        shap_values = self.explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        explanations = []
        for i in range(X.shape[0]):
            exp = dict(zip(self.feature_names, shap_values[i]))
            explanations.append(exp)
        
        return explanations
    
    def get_global_importance(self, X: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Get global feature importance.
        
        Args:
            X: Feature matrix (optional, uses background if not provided)
            
        Returns:
            Dictionary mapping feature names to mean absolute SHAP values
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized")
        
        if X is None:
            if self.background_data is None:
                raise ValueError("Either X or background_data must be provided")
            X = self.background_data
        
        shap_values = self.explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        return dict(zip(self.feature_names, mean_abs_shap))
    
    def get_top_features(
        self,
        shap_values: Dict[str, float],
        top_n: int = 5
    ) -> List[tuple]:
        """
        Get top N features by absolute SHAP value.
        
        Args:
            shap_values: Dictionary of feature -> SHAP value
            top_n: Number of top features to return
            
        Returns:
            List of (feature, shap_value) tuples sorted by absolute value
        """
        sorted_features = sorted(
            shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_features[:top_n]
    
    def format_explanation(
        self,
        shap_values: Dict[str, float],
        direction: str
    ) -> List[str]:
        """
        Format SHAP values into human-readable explanations.
        
        Args:
            shap_values: Dictionary of feature -> SHAP value
            direction: Prediction direction (BUY/SELL)
            
        Returns:
            List of human-readable reason strings
        """
        top_features = self.get_top_features(shap_values, top_n=5)
        reasons = []
        
        for feature, value in top_features:
            if abs(value) < 0.01:
                continue
            
            feature_lower = feature.lower()
            impact = "increased" if value > 0 else "decreased"
            
            if 'rsi' in feature_lower:
                reasons.append(f"RSI {impact} prediction probability")
            elif 'macd' in feature_lower:
                reasons.append(f"MACD {impact} prediction probability")
            elif 'ema' in feature_lower or 'sma' in feature_lower:
                reasons.append(f"Moving average {impact} prediction probability")
            elif 'volume' in feature_lower:
                reasons.append(f"Volume {impact} prediction probability")
            elif 'oi' in feature_lower or 'option' in feature_lower:
                reasons.append(f"Options data {impact} prediction probability")
            elif 'news' in feature_lower or 'sentiment' in feature_lower:
                reasons.append(f"News sentiment {impact} prediction probability")
            elif 'sector' in feature_lower:
                reasons.append(f"Sector performance {impact} prediction probability")
            else:
                reasons.append(f"{feature} {impact} prediction probability")
        
        if not reasons:
            reasons.append("Prediction based on multiple technical and fundamental factors")
        
        return reasons
