"""
Base Feature Class

Institutional-grade feature engineering foundation.
All features inherit from this base class for consistency.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("feature_base")


class Feature(ABC):
    """
    Base class for all features in the feature store.

    Every feature must implement:
    - name: Unique identifier for the feature
    - owner: Person/team responsible for the feature
    - version: Feature version
    - dependencies: List of features or raw data this feature depends on
    - lookback: Number of days of historical data required
    - frequency: Feature update frequency
    - data_source: Source of data for this feature
    - compute(): Method to calculate the feature
    """

    def __init__(self):
        """Initialize the feature."""
        self.logger = logger

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for the feature.

        Returns:
            Feature name string
        """
        pass

    @property
    def owner(self) -> str:
        """
        Person/team responsible for the feature.

        Returns:
            Owner name or team
        """
        return "quant_team"

    @property
    def version(self) -> str:
        """
        Feature version.

        Returns:
            Version string
        """
        return "1.0.0"

    @property
    @abstractmethod
    def dependencies(self) -> list[str]:
        """
        List of features or raw data this feature depends on.

        Returns:
            List of dependency names
        """
        pass

    @property
    @abstractmethod
    def lookback(self) -> int:
        """
        Number of days of historical data required.

        Returns:
            Lookback period in days
        """
        pass

    @property
    def frequency(self) -> str:
        """
        Feature update frequency.

        Returns:
            Frequency string (daily, hourly, etc.)
        """
        return "daily"

    @property
    def data_source(self) -> str:
        """
        Source of data for this feature.

        Returns:
            Data source string
        """
        return "market_data"

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute the feature from input data.

        Args:
            data: DataFrame with required columns

        Returns:
            Series with computed feature values
        """
        pass

    def get_metadata(self) -> dict[str, Any]:
        """
        Get feature metadata.

        Returns:
            Dictionary with all feature metadata
        """
        return {
            "name": self.name,
            "owner": self.owner,
            "version": self.version,
            "dependencies": self.dependencies,
            "lookback": self.lookback,
            "frequency": self.frequency,
            "data_source": self.data_source,
        }

    def validate_input(self, data: pd.DataFrame) -> bool:
        """
        Validate input data has required columns.

        Args:
            data: Input DataFrame

        Returns:
            True if valid, False otherwise
        """
        required_cols = self.dependencies
        missing_cols = [col for col in required_cols if col not in data.columns]

        if missing_cols:
            self.logger.warning(f"Missing columns for {self.name}: {missing_cols}")
            return False

        return True

    def compute_with_validation(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute feature with input validation.

        Args:
            data: Input DataFrame

        Returns:
            Series with computed feature values
        """
        if not self.validate_input(data):
            raise ValueError(f"Invalid input data for feature {self.name}")

        return self.compute(data)


class FeatureStore:
    """
    Central feature store for managing and computing features.

    Supports feature registration, dependency resolution, and batch computation.
    """

    def __init__(self):
        """Initialize the feature store."""
        self.features: dict[str, Feature] = {}
        self.logger = logger

    def register(self, feature: Feature) -> None:
        """
        Register a feature in the store.

        Args:
            feature: Feature instance to register
        """
        self.features[feature.name] = feature
        self.logger.info(f"Registered feature: {feature.name}")

    def get_feature(self, name: str) -> Feature | None:
        """
        Get a feature by name.

        Args:
            name: Feature name

        Returns:
            Feature instance or None if not found
        """
        return self.features.get(name)

    def compute_feature(self, name: str, data: pd.DataFrame) -> pd.Series:
        """
        Compute a single feature.

        Args:
            name: Feature name
            data: Input DataFrame

        Returns:
            Series with computed feature values
        """
        feature = self.get_feature(name)
        if not feature:
            raise ValueError(f"Feature {name} not found in store")

        return feature.compute_with_validation(data)

    def compute_features(self, feature_names: list[str], data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute multiple features.

        Args:
            feature_names: List of feature names to compute
            data: Input DataFrame

        Returns:
            DataFrame with computed features
        """
        result = pd.DataFrame(index=data.index)

        for name in feature_names:
            try:
                feature = self.get_feature(name)
                if feature:
                    result[name] = feature.compute_with_validation(data)
                else:
                    self.logger.warning(f"Feature {name} not found")
            except Exception as e:
                self.logger.error(f"Failed to compute feature {name}: {str(e)}")
                result[name] = np.nan

        return result

    def pit_join(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        on: str = "date",
        by: str | None = "symbol",
        tolerance: pd.Timedelta | None = None,
    ) -> pd.DataFrame:
        """
        Enforce Point-in-Time (PIT) joins using merge_asof to prevent look-ahead bias.

        Args:
            left_df: The primary/query DataFrame (e.g., market prices).
            right_df: The secondary/historical record DataFrame (e.g., fundamentals/macro).
            on: The time/timestamp column to perform the asof join on. Must be sorted in both DataFrames.
            by: Grouping column (e.g., symbol) to match exactly.
            tolerance: Optional maximum time gap allowed for matching.

        Returns:
            The joined DataFrame.
        """
        if on not in left_df.columns:
            raise ValueError(f"Time column '{on}' missing from left DataFrame")
        if on not in right_df.columns:
            raise ValueError(f"Time column '{on}' missing from right DataFrame")

        left_df = left_df.copy()
        right_df = right_df.copy()
        left_df[on] = pd.to_datetime(left_df[on])
        right_df[on] = pd.to_datetime(right_df[on])

        left_sorted = left_df.sort_values(on)
        right_sorted = right_df.sort_values(on)

        if by:
            if by not in left_sorted.columns or by not in right_sorted.columns:
                raise ValueError(f"Grouping column '{by}' missing from one of the DataFrames")

            joined = pd.merge_asof(
                left_sorted, right_sorted, on=on, by=by, direction="backward", tolerance=tolerance
            )
        else:
            joined = pd.merge_asof(
                left_sorted, right_sorted, on=on, direction="backward", tolerance=tolerance
            )

        return joined

    def resolve_dependencies(self, feature_name: str) -> list[str]:
        """
        Resolve all dependencies for a feature (recursive).

        Args:
            feature_name: Feature name

        Returns:
            List of all required features in dependency order
        """
        visited = set()
        result = []

        def _resolve(name: str):
            if name in visited:
                return

            visited.add(name)
            feature = self.get_feature(name)

            if feature:
                for dep in feature.dependencies:
                    if dep in self.features:
                        _resolve(dep)

            result.append(name)

        _resolve(feature_name)
        return result

    def get_max_lookback(self, feature_names: list[str]) -> int:
        """
        Get maximum lookback period for a set of features.

        Args:
            feature_names: List of feature names

        Returns:
            Maximum lookback period in days
        """
        max_lookback = 0

        for name in feature_names:
            feature = self.get_feature(name)
            if feature and feature.lookback > max_lookback:
                max_lookback = feature.lookback

        return max_lookback

    def list_features(self) -> list[str]:
        """
        List all registered features.

        Returns:
            List of feature names
        """
        return list(self.features.keys())

    def get_feature_info(self, name: str) -> dict[str, Any] | None:
        """
        Get information about a feature.

        Args:
            name: Feature name

        Returns:
            Dictionary with feature information
        """
        feature = self.get_feature(name)
        if not feature:
            return None

        return {
            "name": feature.name,
            "dependencies": feature.dependencies,
            "lookback": feature.lookback,
        }
