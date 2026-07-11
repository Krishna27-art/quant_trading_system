"""
Base Feature Class

Standardized interface for all features in the Feature Laboratory.
Every feature must inherit from this class and implement the required methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from enum import Enum


class FeatureCategory(Enum):
    """Feature categories."""
    TECHNICAL = "technical"
    VOLUME = "volume"
    OPTIONS = "options"
    FUNDAMENTALS = "fundamentals"
    MACRO = "macro"
    SENTIMENT = "sentiment"
    SECTOR = "sector"
    MARKET = "market"


class Timeframe(Enum):
    """Supported timeframes."""
    TICK = "tick"
    MIN_1 = "1m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    MIN_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1D"
    WEEK_1 = "1W"
    MONTH_1 = "1M"


@dataclass
class FeatureMetadata:
    """
    Standardized metadata for every feature.
    
    This ensures every feature has complete documentation and
    can be tracked, versioned, and analyzed consistently.
    """
    feature_name: str
    description: str
    category: FeatureCategory
    timeframe: Timeframe
    required_columns: List[str]
    output_range: Optional[str] = None
    version: str = "1.0"
    author: str = "system"
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    computation_method: str = ""
    assumptions: str = ""
    limitations: str = ""
    references: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "feature_name": self.feature_name,
            "description": self.description,
            "category": self.category.value,
            "timeframe": self.timeframe.value,
            "required_columns": self.required_columns,
            "output_range": self.output_range,
            "version": self.version,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "computation_method": self.computation_method,
            "assumptions": self.assumptions,
            "limitations": self.limitations,
            "references": self.references,
        }


@dataclass
class FeatureResult:
    """Result of feature computation."""
    feature_name: str
    values: pd.Series
    metadata: FeatureMetadata
    computation_time_ms: float
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "feature_name": self.feature_name,
            "values": self.values.to_dict(),
            "metadata": self.metadata.to_dict(),
            "computation_time_ms": self.computation_time_ms,
            "warnings": self.warnings,
        }


class BaseFeature(ABC):
    """
    Abstract base class for all features.
    
    Every feature must:
    1. Define metadata
    2. Implement compute() method
    3. Validate input data
    4. Handle edge cases
    5. Log computation details
    """
    
    def __init__(self):
        self.metadata: FeatureMetadata = self._define_metadata()
        self._validate_metadata()
    
    @abstractmethod
    def _define_metadata(self) -> FeatureMetadata:
        """
        Define feature metadata.
        
        Returns:
            FeatureMetadata object with complete documentation
        """
        pass
    
    @abstractmethod
    def compute(self, data: pd.DataFrame) -> FeatureResult:
        """
        Compute the feature.
        
        Args:
            data: DataFrame with required columns
            
        Returns:
            FeatureResult with computed values and metadata
        """
        pass
    
    def _validate_metadata(self) -> None:
        """Validate that metadata is complete and correct."""
        if not self.metadata.feature_name:
            raise ValueError("Feature name cannot be empty")
        
        if not self.metadata.description:
            raise ValueError("Feature description cannot be empty")
        
        if not self.metadata.required_columns:
            raise ValueError("Required columns cannot be empty")
        
        if not self.metadata.version:
            raise ValueError("Version cannot be empty")
    
    def validate_input(self, data: pd.DataFrame) -> None:
        """
        Validate input data before computation.
        
        Args:
            data: Input DataFrame
            
        Raises:
            ValueError: If validation fails
        """
        missing_cols = set(self.metadata.required_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns for {self.metadata.feature_name}: {missing_cols}"
            )
        
        if data.empty:
            raise ValueError(f"Input DataFrame is empty for {self.metadata.feature_name}")
        
        # Check for NaN values in required columns
        for col in self.metadata.required_columns:
            if data[col].isna().all():
                raise ValueError(f"Column {col} is all NaN for {self.metadata.feature_name}")
    
    def handle_missing_values(
        self,
        series: pd.Series,
        method: str = "ffill"
    ) -> pd.Series:
        """
        Handle missing values in computed feature.
        
        Args:
            series: Computed feature series
            method: Method to handle missing values ('ffill', 'bfill', 'drop', 'zero')
            
        Returns:
            Series with handled missing values
        """
        if method == "ffill":
            return series.ffill()
        elif method == "bfill":
            return series.bfill()
        elif method == "drop":
            return series.dropna()
        elif method == "zero":
            return series.fillna(0)
        else:
            return series
    
    def compute_with_validation(self, data: pd.DataFrame) -> FeatureResult:
        """
        Compute feature with full validation pipeline.
        
        Args:
            data: Input DataFrame
            
        Returns:
            FeatureResult with computed values
        """
        import time
        import logging
        
        logger = logging.getLogger(__name__)
        
        start_time = time.time()
        warnings = []
        
        try:
            # Validate input
            self.validate_input(data)
            
            # Compute feature
            result = self.compute(data)
            
            # Add computation time
            computation_time = (time.time() - start_time) * 1000
            result.computation_time_ms = computation_time
            
            # Log computation
            logger.info(
                f"Computed {self.metadata.feature_name} "
                f"in {computation_time:.2f}ms "
                f"for {len(data)} rows"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error computing {self.metadata.feature_name}: {e}")
            raise
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get complete feature information.
        
        Returns:
            Dictionary with feature metadata and info
        """
        return {
            "metadata": self.metadata.to_dict(),
            "class_name": self.__class__.__name__,
            "module": self.__class__.__module__,
        }
