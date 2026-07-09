"""
Feature Store

Storage and retrieval system for computed factors.
Persists factor values with metadata for reproducibility and lineage.
"""

import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.logger import get_logger

logger = get_logger("research.feature_store")


@dataclass
class FactorData:
    """Container for factor data with metadata."""
    factor_name: str
    values: pd.Series
    metadata: Dict[str, Any]
    computed_at: datetime
    data_hash: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "factor_name": self.factor_name,
            "metadata": self.metadata,
            "computed_at": self.computed_at.isoformat(),
            "data_hash": self.data_hash,
            "length": len(self.values),
            "start_date": str(self.values.index[0]) if len(self.values) > 0 else None,
            "end_date": str(self.values.index[-1]) if len(self.values) > 0 else None,
        }


class FeatureStore:
    """
    Storage and retrieval system for computed factors.
    
    Persists factor values with metadata for:
    - Reproducibility
    - Lineage tracking
    - Fast retrieval
    - Version management
    """
    
    def __init__(self, base_path: str = "research/feature_store/data"):
        """
        Initialize feature store.
        
        Args:
            base_path: Base path for storing factor data
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.factors_path = self.base_path / "factors"
        self.factors_path.mkdir(exist_ok=True)
        
        self.metadata_path = self.base_path / "metadata"
        self.metadata_path.mkdir(exist_ok=True)
        
        self._logger = get_logger("research.feature_store")
    
    def _compute_hash(self, data: pd.Series) -> str:
        """Compute hash of data for integrity checking."""
        return hash(tuple(data.values.tobytes()))
    
    def store_factor(
        self,
        factor_name: str,
        values: pd.Series,
        metadata: Dict[str, Any],
        version: str = "latest",
    ) -> str:
        """
        Store factor data with metadata.
        
        Args:
            factor_name: Name of factor
            values: Series with factor values
            metadata: Factor metadata
            version: Version identifier
            
        Returns:
            Path to stored factor
        """
        # Create factor directory
        factor_dir = self.factors_path / factor_name
        factor_dir.mkdir(exist_ok=True)
        
        # Compute data hash
        data_hash = self._compute_hash(values)
        
        # Create factor data container
        factor_data = FactorData(
            factor_name=factor_name,
            values=values,
            metadata=metadata,
            computed_at=datetime.now(),
            data_hash=data_hash,
        )
        
        # Store values as pickle
        values_path = factor_dir / f"{version}.pkl"
        with open(values_path, "wb") as f:
            pickle.dump(values, f)
        
        # Store metadata as JSON
        metadata_path = self.metadata_path / f"{factor_name}_{version}.json"
        with open(metadata_path, "w") as f:
            json.dump(factor_data.to_dict(), f, indent=2)
        
        self._logger.info(f"Stored factor {factor_name} version {version} to {values_path}")
        return str(values_path)
    
    def load_factor(
        self,
        factor_name: str,
        version: str = "latest",
    ) -> Optional[pd.Series]:
        """
        Load factor data.
        
        Args:
            factor_name: Name of factor
            version: Version identifier
            
        Returns:
            Series with factor values or None if not found
        """
        values_path = self.factors_path / factor_name / f"{version}.pkl"
        
        if not values_path.exists():
            self._logger.warning(f"Factor {factor_name} version {version} not found")
            return None
        
        try:
            with open(values_path, "rb") as f:
                values = pickle.load(f)
            
            self._logger.info(f"Loaded factor {factor_name} version {version}")
            return values
        except Exception as e:
            self._logger.error(f"Failed to load factor {factor_name} version {version}: {e}")
            return None
    
    def load_metadata(
        self,
        factor_name: str,
        version: str = "latest",
    ) -> Optional[Dict[str, Any]]:
        """
        Load factor metadata.
        
        Args:
            factor_name: Name of factor
            version: Version identifier
            
        Returns:
            Metadata dictionary or None if not found
        """
        metadata_path = self.metadata_path / f"{factor_name}_{version}.json"
        
        if not metadata_path.exists():
            self._logger.warning(f"Metadata for {factor_name} version {version} not found")
            return None
        
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            return metadata
        except Exception as e:
            self._logger.error(f"Failed to load metadata for {factor_name} version {version}: {e}")
            return None
    
    def list_factors(self) -> List[str]:
        """
        List all available factors.
        
        Returns:
            List of factor names
        """
        factors = []
        for factor_dir in self.factors_path.iterdir():
            if factor_dir.is_dir():
                factors.append(factor_dir.name)
        
        return sorted(factors)
    
    def list_versions(self, factor_name: str) -> List[str]:
        """
        List available versions for a factor.
        
        Args:
            factor_name: Name of factor
            
        Returns:
            List of version identifiers
        """
        factor_dir = self.factors_path / factor_name
        
        if not factor_dir.exists():
            return []
        
        versions = []
        for version_file in factor_dir.glob("*.pkl"):
            versions.append(version_file.stem)
        
        return sorted(versions)
    
    def delete_factor(
        self,
        factor_name: str,
        version: Optional[str] = None,
    ) -> bool:
        """
        Delete factor data.
        
        Args:
            factor_name: Name of factor
            version: Optional version identifier (if None, delete all versions)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if version:
                # Delete specific version
                values_path = self.factors_path / factor_name / f"{version}.pkl"
                metadata_path = self.metadata_path / f"{factor_name}_{version}.json"
                
                if values_path.exists():
                    values_path.unlink()
                if metadata_path.exists():
                    metadata_path.unlink()
                
                self._logger.info(f"Deleted factor {factor_name} version {version}")
            else:
                # Delete all versions
                factor_dir = self.factors_path / factor_name
                if factor_dir.exists():
                    for file in factor_dir.glob("*"):
                        file.unlink()
                    factor_dir.rmdir()
                
                # Delete metadata files
                for metadata_file in self.metadata_path.glob(f"{factor_name}_*"):
                    metadata_file.unlink()
                
                self._logger.info(f"Deleted all versions of factor {factor_name}")
            
            return True
        except Exception as e:
            self._logger.error(f"Failed to delete factor {factor_name}: {e}")
            return False
    
    def store_batch(
        self,
        factor_data: Dict[str, pd.Series],
        metadata: Dict[str, Dict[str, Any]],
        version: str = "latest",
    ) -> Dict[str, str]:
        """
        Store multiple factors in batch.
        
        Args:
            factor_data: Dictionary mapping factor names to values
            metadata: Dictionary mapping factor names to metadata
            version: Version identifier
            
        Returns:
            Dictionary mapping factor names to storage paths
        """
        paths = {}
        
        for factor_name, values in factor_data.items():
            if factor_name in metadata:
                path = self.store_factor(
                    factor_name=factor_name,
                    values=values,
                    metadata=metadata[factor_name],
                    version=version,
                )
                paths[factor_name] = path
        
        return paths
    
    def load_batch(
        self,
        factor_names: List[str],
        version: str = "latest",
    ) -> Dict[str, pd.Series]:
        """
        Load multiple factors in batch.
        
        Args:
            factor_names: List of factor names
            version: Version identifier
            
        Returns:
            Dictionary mapping factor names to values
        """
        factors = {}
        
        for factor_name in factor_names:
            values = self.load_factor(factor_name, version=version)
            if values is not None:
                factors[factor_name] = values
        
        return factors
    
    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get information about feature store storage.
        
        Returns:
            Dictionary with storage information
        """
        factors = self.list_factors()
        
        total_size = 0
        factor_info = {}
        
        for factor_name in factors:
            factor_dir = self.factors_path / factor_name
            if factor_dir.exists():
                factor_size = sum(f.stat().st_size for f in factor_dir.glob("*") if f.is_file())
                total_size += factor_size
                
                versions = self.list_versions(factor_name)
                factor_info[factor_name] = {
                    "versions": versions,
                    "size_bytes": factor_size,
                    "size_mb": factor_size / (1024 * 1024),
                }
        
        return {
            "total_factors": len(factors),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "factors": factor_info,
        }
