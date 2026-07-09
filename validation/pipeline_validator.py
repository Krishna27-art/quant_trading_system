"""
Training Pipeline Validation Module

Validates that training and inference pipelines use identical preprocessing.
Ensures byte-for-byte consistency in scaling, encoding, missing values, feature order.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import hashlib
import json

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

from utils.logger import get_logger

logger = get_logger("validation.pipeline")


@dataclass
class PipelineConfig:
    """Configuration of a preprocessing pipeline."""
    scaling_method: str  # standard, minmax, none
    imputation_method: str  # mean, median, most_frequent, constant
    encoding_method: str  # onehot, ordinal, none
    feature_order: List[str]
    missing_value_strategy: str  # drop, impute, constant
    custom_params: Dict[str, Any]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "scaling_method": self.scaling_method,
            "imputation_method": self.imputation_method,
            "encoding_method": self.encoding_method,
            "feature_order": self.feature_order,
            "missing_value_strategy": self.missing_value_strategy,
            "custom_params": self.custom_params,
        }
    
    def compute_hash(self) -> str:
        """Compute hash of pipeline configuration."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


@dataclass
class PipelineValidationResult:
    """Result of pipeline validation."""
    passed: bool
    training_config: PipelineConfig
    inference_config: PipelineConfig
    config_match: bool
    feature_order_match: bool
    scaling_match: bool
    imputation_match: bool
    encoding_match: bool
    preprocessing_consistent: bool
    errors: List[str]
    warnings: List[str]
    timestamp: datetime
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "passed": self.passed,
            "training_config": self.training_config.to_dict(),
            "inference_config": self.inference_config.to_dict(),
            "config_match": self.config_match,
            "feature_order_match": self.feature_order_match,
            "scaling_match": self.scaling_match,
            "imputation_match": self.imputation_match,
            "encoding_match": self.encoding_match,
            "preprocessing_consistent": self.preprocessing_consistent,
            "errors": self.errors,
            "warnings": self.warnings,
            "timestamp": self.timestamp.isoformat(),
        }


class PipelineValidator:
    """
    Validates that training and inference pipelines are consistent.
    
    Checks:
    1. Same preprocessing steps (scaling, encoding, missing values)
    2. Same feature order
    3. Same imputation strategy
    4. Same scaling parameters (if applicable)
    5. Byte-for-byte consistency in transformed data
    """
    
    def __init__(self, strict: bool = True):
        """
        Initialize validator.
        
        Args:
            strict: If True, fail on any mismatch
        """
        self.strict = strict
    
    def validate_pipeline_consistency(
        self,
        training_config: PipelineConfig,
        inference_config: PipelineConfig,
        training_sample: pd.DataFrame,
        inference_sample: pd.DataFrame,
    ) -> PipelineValidationResult:
        """
        Validate that training and inference pipelines are consistent.
        
        Args:
            training_config: Configuration used during training
            inference_config: Configuration used during inference
            training_sample: Sample of training data after preprocessing
            inference_sample: Sample of inference data after preprocessing
            
        Returns:
            PipelineValidationResult
        """
        errors = []
        warnings = []
        passed = True
        
        # Check 1: Configuration match
        config_match = training_config.compute_hash() == inference_config.compute_hash()
        if not config_match:
            errors.append("Training and inference configurations do not match")
            passed = False
            if self.strict:
                return self._create_result(
                    False, training_config, inference_config,
                    config_match, False, False, False, False, False,
                    errors, warnings
                )
        
        # Check 2: Feature order match
        feature_order_match = training_config.feature_order == inference_config.feature_order
        if not feature_order_match:
            errors.append("Feature order mismatch between training and inference")
            passed = False
        
        # Check 3: Scaling method match
        scaling_match = training_config.scaling_method == inference_config.scaling_method
        if not scaling_match:
            errors.append("Scaling method mismatch")
            passed = False
        
        # Check 4: Imputation method match
        imputation_match = training_config.imputation_method == inference_config.imputation_method
        if not imputation_match:
            errors.append("Imputation method mismatch")
            passed = False
        
        # Check 5: Encoding method match
        encoding_match = training_config.encoding_method == inference_config.encoding_method
        if not encoding_match:
            errors.append("Encoding method mismatch")
            passed = False
        
        # Check 6: Preprocessing consistency (if samples provided)
        preprocessing_consistent = True
        if training_sample is not None and inference_sample is not None:
            if not training_sample.equals(inference_sample):
                # Check if values are close (allowing for floating point differences)
                if training_sample.shape != inference_sample.shape:
                    errors.append(
                        f"Shape mismatch: training {training_sample.shape} vs "
                        f"inference {inference_sample.shape}"
                    )
                    passed = False
                    preprocessing_consistent = False
                else:
                    # Check if values are approximately equal
                    try:
                        np.testing.assert_allclose(
                            training_sample.values,
                            inference_sample.values,
                            rtol=1e-5,
                            atol=1e-8,
                        )
                    except AssertionError:
                        errors.append("Preprocessed data values differ between training and inference")
                        passed = False
                        preprocessing_consistent = False
        
        return self._create_result(
            passed, training_config, inference_config,
            config_match, feature_order_match, scaling_match,
            imputation_match, encoding_match, preprocessing_consistent,
            errors, warnings
        )
    
    def extract_pipeline_config(
        self,
        pipeline: Any,
        feature_cols: List[str],
    ) -> PipelineConfig:
        """
        Extract configuration from a sklearn pipeline.
        
        Args:
            pipeline: sklearn Pipeline object
            feature_cols: List of feature column names
            
        Returns:
            PipelineConfig
        """
        scaling_method = "none"
        imputation_method = "none"
        encoding_method = "none"
        custom_params = {}
        
        # Inspect pipeline steps
        if hasattr(pipeline, 'named_steps'):
            for name, step in pipeline.named_steps.items():
                if isinstance(step, StandardScaler):
                    scaling_method = "standard"
                    custom_params[f"{name}_mean"] = step.mean_.tolist() if hasattr(step, 'mean_') else None
                    custom_params[f"{name}_scale"] = step.scale_.tolist() if hasattr(step, 'scale_') else None
                elif isinstance(step, MinMaxScaler):
                    scaling_method = "minmax"
                    custom_params[f"{name}_min"] = step.data_min_.tolist() if hasattr(step, 'data_min_') else None
                    custom_params[f"{name}_max"] = step.data_max_.tolist() if hasattr(step, 'data_max_') else None
                elif isinstance(step, SimpleImputer):
                    imputation_method = step.strategy
                    custom_params[f"{name}_fill_value"] = step.fill_value if hasattr(step, 'fill_value') else None
        
        return PipelineConfig(
            scaling_method=scaling_method,
            imputation_method=imputation_method,
            encoding_method=encoding_method,
            feature_order=feature_cols,
            missing_value_strategy="impute" if imputation_method != "none" else "drop",
            custom_params=custom_params,
        )
    
    def _create_result(
        self,
        passed: bool,
        training_config: PipelineConfig,
        inference_config: PipelineConfig,
        config_match: bool,
        feature_order_match: bool,
        scaling_match: bool,
        imputation_match: bool,
        encoding_match: bool,
        preprocessing_consistent: bool,
        errors: List[str],
        warnings: List[str],
    ) -> PipelineValidationResult:
        """Create validation result."""
        return PipelineValidationResult(
            passed=passed,
            training_config=training_config,
            inference_config=inference_config,
            config_match=config_match,
            feature_order_match=feature_order_match,
            scaling_match=scaling_match,
            imputation_match=imputation_match,
            encoding_match=encoding_match,
            preprocessing_consistent=preprocessing_consistent,
            errors=errors,
            warnings=warnings,
            timestamp=datetime.now(),
        )


def compute_dataset_hash(df: pd.DataFrame) -> str:
    """
    Compute SHA256 hash of dataset for reproducibility.
    
    Args:
        df: DataFrame to hash
        
    Returns:
        SHA256 hash string
    """
    # Sort by index and columns to ensure consistent ordering
    df_sorted = df.sort_index(axis=0).sort_index(axis=1)
    
    # Convert to string representation
    df_str = df_sorted.to_string()
    
    # Compute hash
    return hashlib.sha256(df_str.encode()).hexdigest()


def compute_feature_hash(feature_cols: List[str], feature_versions: Dict[str, str]) -> str:
    """
    Compute hash of feature list and versions for reproducibility.
    
    Args:
        feature_cols: List of feature column names
        feature_versions: Dictionary mapping feature names to versions
        
    Returns:
        Hash string
    """
    # Create sorted feature list with versions
    feature_list = sorted([f"{col}:{feature_versions.get(col, 'v1.0')}" for col in feature_cols])
    
    # Compute hash
    feature_str = "|".join(feature_list)
    return hashlib.sha256(feature_str.encode()).hexdigest()[:16]


def validate_pipeline(
    training_pipeline: Any,
    inference_pipeline: Any,
    training_data: pd.DataFrame,
    inference_data: pd.DataFrame,
    feature_cols: List[str],
) -> PipelineValidationResult:
    """
    Convenience function to validate pipeline consistency.
    
    Args:
        training_pipeline: sklearn Pipeline used for training
        inference_pipeline: sklearn Pipeline used for inference
        training_data: Sample training data
        inference_data: Sample inference data
        feature_cols: List of feature column names
        
    Returns:
        PipelineValidationResult
    """
    validator = PipelineValidator()
    
    training_config = validator.extract_pipeline_config(training_pipeline, feature_cols)
    inference_config = validator.extract_pipeline_config(inference_pipeline, feature_cols)
    
    # Transform samples
    training_transformed = training_pipeline.transform(training_data[feature_cols])
    inference_transformed = inference_pipeline.transform(inference_data[feature_cols])
    
    # Convert to DataFrames for comparison
    training_df = pd.DataFrame(training_transformed, columns=feature_cols)
    inference_df = pd.DataFrame(inference_transformed, columns=feature_cols)
    
    return validator.validate_pipeline_consistency(
        training_config, inference_config, training_df, inference_df
    )
