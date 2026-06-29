from typing import Any

import numpy as np
import pandas as pd
import shap
import torch


class ExplainabilityEngine:
    """
    Wraps SHAP (SHapley Additive exPlanations) to provide local and global explainability
    for tree-based and deep learning models.
    """

    def __init__(self, model: Any, model_type: str = "tree"):
        self.model = model
        self.model_type = model_type
        self.explainer = None

    def fit(self, background_data: pd.DataFrame):
        """Fits the SHAP explainer on a background dataset."""
        if self.model_type == "tree":
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == "deep":
            # DeepExplainer requires PyTorch tensors as background
            background_tensor = torch.tensor(background_data.values, dtype=torch.float32)
            self.explainer = shap.DeepExplainer(self.model, background_tensor)
        else:
            self.explainer = shap.KernelExplainer(self.model.predict, background_data)

    def get_local_explanation(self, observation: pd.DataFrame) -> dict[str, float]:
        """
        Calculates SHAP values for a single prediction (e.g. 'Why did we buy today?').
        Returns a dictionary of Feature -> Impact.
        """
        if self.explainer is None:
            raise RuntimeError("Explainer must be fitted first.")

        shap_values = self.explainer.shap_values(observation)

        # Format into a readable dict
        if isinstance(shap_values, list):  # For multi-class
            shap_values = shap_values[1]  # Assuming class 1 is the target

        feature_names = observation.columns
        # Handle 1D or 2D array output from SHAP
        vals = shap_values[0] if len(np.shape(shap_values)) > 1 else shap_values

        return dict(zip(feature_names, vals, strict=False))

    def get_global_explanation(self, X: pd.DataFrame) -> dict[str, float]:
        """
        Returns mean absolute SHAP values per feature, providing a global
        measure of each feature's importance across the dataset.

        Args:
            X: DataFrame of observations to explain.

        Returns:
            Dictionary mapping feature name to its mean |SHAP value|.
        """
        if self.explainer is None:
            raise RuntimeError("Explainer must be fitted first.")

        shap_values = self.explainer.shap_values(X)

        # For multi-class output, use class 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_array = np.array(shap_values)
        mean_abs = np.mean(np.abs(shap_array), axis=0)

        return dict(zip(X.columns, mean_abs, strict=False))
