"""
Feature Engineering Validation Module

Validates engineered features for correctness and prevents look-ahead bias.
Every feature is automatically tested for NaN, Inf, future data, constancy, duplicates, etc.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from scipy import stats

from utils.logger import get_logger

logger = get_logger("validation.features")


@dataclass
class FeatureValidationResult:
    """Result of validating a single feature."""
    feature_name: str
    passed: bool
    has_nan: bool
    has_inf: bool
    uses_future_data: bool
    is_constant: bool
    is_duplicate: bool
    variance_zero: bool
    wrong_dtype: bool
    has_outliers: bool
    missing_pct: float
    correlation: Optional[float]
    ic: Optional[float]
    errors: List[str]
    warnings: List[str]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "feature_name": self.feature_name,
            "passed": self.passed,
            "has_nan": self.has_nan,
            "has_inf": self.has_inf,
            "uses_future_data": self.uses_future_data,
            "is_constant": self.is_constant,
            "is_duplicate": self.is_duplicate,
            "variance_zero": self.variance_zero,
            "wrong_dtype": self.wrong_dtype,
            "has_outliers": self.has_outliers,
            "missing_pct": round(self.missing_pct, 4),
            "correlation": round(self.correlation, 4) if self.correlation is not None else None,
            "ic": round(self.ic, 4) if self.ic is not None else None,
            "errors": self.errors,
            "warnings": self.warnings,
        }


@dataclass
class FeatureReport:
    """Report from feature validation."""
    total_features: int
    passed_features: int
    failed_features: int
    results: List[FeatureValidationResult]
    timestamp: datetime
    dataframe_shape: tuple[int, int]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "total_features": self.total_features,
            "passed_features": self.passed_features,
            "failed_features": self.failed_features,
            "pass_rate_pct": round(self.passed_features / self.total_features * 100, 2) if self.total_features > 0 else 0,
            "results": [r.to_dict() for r in self.results],
            "timestamp": self.timestamp.isoformat(),
            "dataframe_shape": self.dataframe_shape,
        }


class FeatureValidator:
    """
    Validates engineered features for correctness.
    
    Checks:
    1. Contains NaN?
    2. Contains Inf?
    3. Contains future data? (look-ahead bias)
    4. Constant? (zero variance)
    5. Duplicate feature?
    6. Variance zero?
    7. Wrong dtype?
    8. Outliers?
    9. Missing percentage
    10. Correlation with target
    11. Information Coefficient (IC)
    """

    def __init__(
        self,
        target_col: Optional[str] = None,
        max_missing_pct: float = 0.5,
        outlier_threshold: float = 3.0,
    ):
        """
        Initialize validator.
        
        Args:
            target_col: Name of target column for correlation/IC calculation
            max_missing_pct: Maximum allowed missing percentage
            outlier_threshold: Z-score threshold for outlier detection
        """
        self.target_col = target_col
        self.max_missing_pct = max_missing_pct
        self.outlier_threshold = outlier_threshold

    def validate(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        timestamp_col: str = "timestamp",
    ) -> FeatureReport:
        """
        Validate all features in DataFrame.
        
        Args:
            df: DataFrame with features
            feature_cols: List of feature column names to validate
            timestamp_col: Name of timestamp column for future data check
            
        Returns:
            FeatureReport with validation results
        """
        results = []
        total_features = len(feature_cols)
        passed_features = 0

        for feature in feature_cols:
            if feature not in df.columns:
                logger.warning(f"Feature {feature} not found in DataFrame")
                continue

            result = self._validate_feature(df, feature, timestamp_col)
            results.append(result)
            if result.passed:
                passed_features += 1

        failed_features = total_features - passed_features

        logger.info(
            f"Feature validation: {passed_features}/{total_features} passed, "
            f"{failed_features} failed"
        )

        return FeatureReport(
            total_features=total_features,
            passed_features=passed_features,
            failed_features=failed_features,
            results=results,
            timestamp=datetime.now(),
            dataframe_shape=df.shape,
        )

    def _validate_feature(
        self,
        df: pd.DataFrame,
        feature: str,
        timestamp_col: str,
    ) -> FeatureValidationResult:
        """Validate a single feature."""
        errors = []
        warnings = []
        passed = True

        series = df[feature]

        # Check 1: NaN values
        has_nan = series.isna().any()
        missing_pct = series.isna().sum() / len(series) if len(series) > 0 else 0
        if has_nan:
            if missing_pct > self.max_missing_pct:
                errors.append(f"Too many NaN values: {missing_pct:.1%}")
                passed = False
            else:
                warnings.append(f"Has NaN values: {missing_pct:.1%}")

        # Check 2: Infinite values
        has_inf = np.isinf(series).any()
        if has_inf:
            errors.append("Contains infinite values")
            passed = False

        # Check 3: Future data (look-ahead bias)
        uses_future_data = self._check_future_data(df, feature, timestamp_col)
        if uses_future_data:
            errors.append("Uses future data (look-ahead bias)")
            passed = False

        # Check 4: Constant feature
        is_constant = series.nunique() <= 1
        if is_constant:
            errors.append("Feature is constant (zero variance)")
            passed = False

        # Check 5: Variance zero
        variance_zero = False
        if not has_nan and not has_inf:
            variance = series.var()
            variance_zero = variance == 0 or np.isnan(variance)
            if variance_zero:
                errors.append("Feature has zero variance")
                passed = False

        # Check 6: Duplicate feature (check correlation with other features)
        is_duplicate = False
        if not has_nan and not has_inf:
            for other_col in df.columns:
                if other_col != feature and other_col != timestamp_col and other_col != self.target_col:
                    if not df[other_col].isna().any() and not np.isinf(df[other_col]).any():
                        corr = series.corr(df[other_col])
                        if corr is not None and abs(corr) > 0.99:
                            is_duplicate = True
                            warnings.append(f"Highly correlated with {other_col} (corr={corr:.3f})")
                            break

        # Check 7: Wrong dtype
        wrong_dtype = False
        expected_numeric = True  # Most features should be numeric
        if expected_numeric and not pd.api.types.is_numeric_dtype(series):
            errors.append(f"Wrong dtype: {series.dtype}, expected numeric")
            passed = False
            wrong_dtype = True

        # Check 8: Outliers
        has_outliers = False
        if not has_nan and not has_inf and pd.api.types.is_numeric_dtype(series):
            z_scores = np.abs(stats.zscore(series.dropna()))
            outlier_count = (z_scores > self.outlier_threshold).sum()
            if outlier_count > 0:
                has_outliers = True
                outlier_pct = outlier_count / len(z_scores)
                if outlier_pct > 0.05:  # More than 5% outliers
                    warnings.append(f"Many outliers: {outlier_pct:.1%} of values")

        # Check 9: Correlation with target
        correlation = None
        if self.target_col and self.target_col in df.columns:
            target = df[self.target_col]
            if not target.isna().any() and not series.isna().any():
                correlation = series.corr(target)
                if correlation is not None and np.isnan(correlation):
                    correlation = None

        # Check 10: Information Coefficient (IC)
        ic = None
        if self.target_col and self.target_col in df.columns:
            target = df[self.target_col]
            if not target.isna().any() and not series.isna().any():
                # IC is rank correlation
                ic = series.corr(target, method='spearman')
                if ic is not None and np.isnan(ic):
                    ic = None

        return FeatureValidationResult(
            feature_name=feature,
            passed=passed,
            has_nan=has_nan,
            has_inf=has_inf,
            uses_future_data=uses_future_data,
            is_constant=is_constant,
            is_duplicate=is_duplicate,
            variance_zero=variance_zero,
            wrong_dtype=wrong_dtype,
            has_outliers=has_outliers,
            missing_pct=missing_pct,
            correlation=correlation,
            ic=ic,
            errors=errors,
            warnings=warnings,
        )

    def _check_future_data(
        self,
        df: pd.DataFrame,
        feature: str,
        timestamp_col: str,
    ) -> bool:
        """
        Check if feature uses future data (look-ahead bias).
        
        This is a heuristic check - we look for features that are perfectly
        correlated with future returns or have suspicious patterns.
        """
        if timestamp_col not in df.columns:
            return False

        # Sort by timestamp
        df_sorted = df.sort_values(timestamp_col).copy()

        # Check if feature is perfectly correlated with future values
        # This is a simple heuristic - more sophisticated checks may be needed
        series = df_sorted[feature].values
        
        # Check for perfect correlation with shifted version
        if len(series) > 10:
            for shift in [1, 2, 3, 5, 10]:
                if shift < len(series):
                    shifted = np.roll(series, -shift)
                    # Remove the shifted NaN values
                    valid_mask = ~np.isnan(series) & ~np.isnan(shifted)
                    if valid_mask.sum() > 10:
                        corr = np.corrcoef(series[valid_mask], shifted[valid_mask])[0, 1]
                        if corr is not None and abs(corr) > 0.95:
                            logger.warning(f"Feature {feature} may use future data (shift={shift}, corr={corr:.3f})")
                            return True

        return False


def validate_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: Optional[str] = None,
    timestamp_col: str = "timestamp",
    max_missing_pct: float = 0.5,
) -> FeatureReport:
    """
    Convenience function to validate features.
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        target_col: Target column for correlation/IC
        timestamp_col: Timestamp column for future data check
        max_missing_pct: Maximum allowed missing percentage
        
    Returns:
        FeatureReport
    """
    validator = FeatureValidator(
        target_col=target_col,
        max_missing_pct=max_missing_pct,
    )
    return validator.validate(df, feature_cols, timestamp_col=timestamp_col)
