"""
Base Factor Abstract Class

All alpha factors must inherit from BaseFactor.
This ensures consistent interface, metadata, and validation across the factor library.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from utils.logger import get_logger

logger = get_logger("research.base_factor")


@dataclass
class FactorMetadata:
    """Metadata for a factor."""
    name: str
    category: str  # momentum, volatility, volume, liquidity, options, macro, sentiment, fundamental, etc.
    version: str
    author: str
    timeframe: str  # INTRADAY, SWING, LONGTERM
    lookback: int  # Lookback period in bars/days
    prediction_horizon: int  # Prediction horizon in bars/days
    required_columns: List[str]
    uses_future_data: bool
    description: str
    created_at: datetime
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "category": self.category,
            "version": self.version,
            "author": self.author,
            "timeframe": self.timeframe,
            "lookback": self.lookback,
            "prediction_horizon": self.prediction_horizon,
            "required_columns": self.required_columns,
            "uses_future_data": self.uses_future_data,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class FactorValidationResult:
    """Result of factor validation."""
    passed: bool
    has_nan: bool
    has_inf: bool
    is_constant: bool
    has_variance: bool
    correct_length: bool
    correct_timestamps: bool
    data_quality_score: float
    errors: List[str]
    warnings: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "passed": self.passed,
            "has_nan": self.has_nan,
            "has_inf": self.has_inf,
            "is_constant": self.is_constant,
            "has_variance": self.has_variance,
            "correct_length": self.correct_length,
            "correct_timestamps": self.correct_timestamps,
            "data_quality_score": round(self.data_quality_score, 4),
            "errors": self.errors,
            "warnings": self.warnings,
        }


class BaseFactor(ABC):
    """
    Abstract base class for all alpha factors.
    
    Every factor must:
    1. Inherit from BaseFactor
    2. Implement compute(), validate(), and metadata() methods
    3. Define required properties
    4. Pass automatic validation
    """
    
    def __init__(self):
        self._logger = get_logger(f"research.factor.{self.__class__.__name__}")
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Factor name."""
        pass
    
    @property
    @abstractmethod
    def category(self) -> str:
        """Factor category (momentum, volatility, etc.)."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Factor version."""
        pass
    
    @property
    @abstractmethod
    def author(self) -> str:
        """Factor author."""
        pass
    
    @property
    @abstractmethod
    def timeframe(self) -> str:
        """Timeframe (INTRADAY, SWING, LONGTERM)."""
        pass
    
    @property
    @abstractmethod
    def lookback(self) -> int:
        """Lookback period in bars/days."""
        pass
    
    @property
    @abstractmethod
    def prediction_horizon(self) -> int:
        """Prediction horizon in bars/days."""
        pass
    
    @property
    @abstractmethod
    def required_columns(self) -> List[str]:
        """Required columns in input DataFrame."""
        pass
    
    @property
    @abstractmethod
    def uses_future_data(self) -> bool:
        """Whether factor uses future data (should be False)."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description."""
        pass
    
    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute factor values.
        
        Args:
            df: DataFrame with required columns
            
        Returns:
            Series with factor values
        """
        pass
    
    def validate(self, df: pd.DataFrame) -> FactorValidationResult:
        """
        Validate factor output.
        
        Args:
            df: DataFrame with factor values
            
        Returns:
            FactorValidationResult
        """
        errors = []
        warnings = []
        passed = True
        
        # Check 1: NaN values
        has_nan = df.isna().any().any()
        if has_nan:
            nan_pct = df.isna().sum().sum() / (df.shape[0] * df.shape[1])
            if nan_pct > 0.1:  # More than 10% NaN
                errors.append(f"Too many NaN values: {nan_pct:.1%}")
                passed = False
            else:
                warnings.append(f"Has NaN values: {nan_pct:.1%}")
        
        # Check 2: Infinite values
        has_inf = np.isinf(df.select_dtypes(include=[np.number])).any().any()
        if has_inf:
            errors.append("Contains infinite values")
            passed = False
        
        # Check 3: Constant values
        is_constant = False
        if df.select_dtypes(include=[np.number]).shape[1] > 0:
            for col in df.select_dtypes(include=[np.number]).columns:
                if df[col].nunique() <= 1:
                    errors.append(f"Column {col} is constant (zero variance)")
                    passed = False
                    is_constant = True
        
        # Check 4: Variance
        has_variance = True
        if df.select_dtypes(include=[np.number]).shape[1] > 0:
            for col in df.select_dtypes(include=[np.number]).columns:
                if df[col].var() == 0 or np.isnan(df[col].var()):
                    has_variance = False
        
        # Check 5: Length
        correct_length = len(df) >= self.lookback
        if not correct_length:
            errors.append(f"Insufficient data: {len(df)} rows, need at least {self.lookback}")
            passed = False
        
        # Check 6: Timestamps
        correct_timestamps = True
        if isinstance(df.index, pd.DatetimeIndex):
            if not df.index.is_monotonic_increasing:
                errors.append("Timestamps are not monotonically increasing")
                passed = False
                correct_timestamps = False
        
        # Data quality score
        quality_score = 1.0
        if has_nan:
            quality_score -= 0.2
        if has_inf:
            quality_score -= 0.3
        if is_constant:
            quality_score -= 0.3
        if not has_variance:
            quality_score -= 0.2
        quality_score = max(0.0, quality_score)
        
        return FactorValidationResult(
            passed=passed,
            has_nan=has_nan,
            has_inf=has_inf,
            is_constant=is_constant,
            has_variance=has_variance,
            correct_length=correct_length,
            correct_timestamps=correct_timestamps,
            data_quality_score=quality_score,
            errors=errors,
            warnings=warnings,
        )
    
    def metadata(self) -> FactorMetadata:
        """
        Get factor metadata.
        
        Returns:
            FactorMetadata
        """
        return FactorMetadata(
            name=self.name,
            category=self.category,
            version=self.version,
            author=self.author,
            timeframe=self.timeframe,
            lookback=self.lookback,
            prediction_horizon=self.prediction_horizon,
            required_columns=self.required_columns,
            uses_future_data=self.uses_future_data,
            description=self.description,
            created_at=datetime.now(),
            last_updated=datetime.now(),
        )
    
    def compute_and_validate(self, df: pd.DataFrame) -> tuple[pd.Series, FactorValidationResult]:
        """
        Compute factor and validate output.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (factor values, validation result)
        """
        # Check required columns
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Compute factor
        factor_values = self.compute(df)
        
        # Validate
        factor_df = pd.DataFrame({"factor": factor_values})
        validation_result = self.validate(factor_df)
        
        if not validation_result.passed:
            self._logger.error(f"Factor {self.name} validation failed: {validation_result.errors}")
        
        return factor_values, validation_result
