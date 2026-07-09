"""
Feature Metadata Tracking System

Tracks metadata for all engineered features to ensure reproducibility
and enable data-driven feature selection decisions.

Stores:
- Feature Name
- Version
- Window/parameters
- Uses Future Data flag
- Missing %
- Correlation with target
- Information Coefficient (IC)
- Creation date
- Last validation date
- Validation status
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import hashlib
from pathlib import Path

import pandas as pd

from utils.logger import get_logger

logger = get_logger("validation.feature_metadata")


@dataclass
class FeatureMetadata:
    """Metadata for a single feature."""
    feature_name: str
    version: str
    description: str
    parameters: Dict[str, Any]  # e.g., window=14, period=20
    uses_future_data: bool
    missing_pct: float
    correlation: Optional[float]
    ic: Optional[float]
    dtype: str
    created_at: datetime
    last_validated: datetime
    validation_status: str  # PASS, FAIL, WARNING
    validation_errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "feature_name": self.feature_name,
            "version": self.version,
            "description": self.description,
            "parameters": self.parameters,
            "uses_future_data": self.uses_future_data,
            "missing_pct": round(self.missing_pct, 4),
            "correlation": round(self.correlation, 4) if self.correlation is not None else None,
            "ic": round(self.ic, 4) if self.ic is not None else None,
            "dtype": self.dtype,
            "created_at": self.created_at.isoformat(),
            "last_validated": self.last_validated.isoformat(),
            "validation_status": self.validation_status,
            "validation_errors": self.validation_errors,
        }
    
    def compute_hash(self) -> str:
        """Compute hash of feature metadata for versioning."""
        hash_str = f"{self.feature_name}_{self.version}_{json.dumps(self.parameters, sort_keys=True)}"
        return hashlib.sha256(hash_str.encode()).hexdigest()[:16]


class FeatureMetadataRegistry:
    """
    Registry for tracking feature metadata.
    
    Maintains a persistent record of all features with their validation status,
    performance metrics, and lineage information.
    """

    def __init__(self, storage_path: str = "data/feature_metadata.json"):
        """
        Initialize registry.
        
        Args:
            storage_path: Path to JSON file for persistent storage
        """
        self.storage_path = Path(storage_path)
        self.metadata: Dict[str, FeatureMetadata] = {}
        self._load()

    def register(
        self,
        feature_name: str,
        version: str,
        description: str,
        parameters: Dict[str, Any],
        uses_future_data: bool,
        dtype: str,
    ) -> FeatureMetadata:
        """
        Register a new feature.
        
        Args:
            feature_name: Name of the feature
            version: Feature version
            description: Human-readable description
            parameters: Feature parameters (window, period, etc)
            uses_future_data: Whether feature uses future data
            dtype: Data type of the feature
            
        Returns:
            FeatureMetadata object
        """
        now = datetime.now()
        
        metadata = FeatureMetadata(
            feature_name=feature_name,
            version=version,
            description=description,
            parameters=parameters,
            uses_future_data=uses_future_data,
            missing_pct=0.0,  # Will be updated on validation
            correlation=None,  # Will be updated on validation
            ic=None,  # Will be updated on validation
            dtype=dtype,
            created_at=now,
            last_validated=now,
            validation_status="PENDING",
            validation_errors=[],
        )
        
        key = f"{feature_name}_{version}"
        self.metadata[key] = metadata
        self._save()
        
        logger.info(f"Registered feature: {feature_name} v{version}")
        return metadata

    def update_validation(
        self,
        feature_name: str,
        version: str,
        validation_status: str,
        missing_pct: float,
        correlation: Optional[float],
        ic: Optional[float],
        validation_errors: List[str],
    ) -> None:
        """
        Update feature with validation results.
        
        Args:
            feature_name: Name of the feature
            version: Feature version
            validation_status: PASS, FAIL, or WARNING
            missing_pct: Percentage of missing values
            correlation: Correlation with target
            ic: Information coefficient
            validation_errors: List of validation errors
        """
        key = f"{feature_name}_{version}"
        if key not in self.metadata:
            logger.warning(f"Feature {feature_name} v{version} not found in registry")
            return
        
        self.metadata[key].validation_status = validation_status
        self.metadata[key].missing_pct = missing_pct
        self.metadata[key].correlation = correlation
        self.metadata[key].ic = ic
        self.metadata[key].validation_errors = validation_errors
        self.metadata[key].last_validated = datetime.now()
        
        self._save()
        logger.info(f"Updated validation for {feature_name} v{version}: {validation_status}")

    def get_metadata(self, feature_name: str, version: Optional[str] = None) -> Optional[FeatureMetadata]:
        """
        Get metadata for a feature.
        
        Args:
            feature_name: Name of the feature
            version: Feature version (if None, returns latest)
            
        Returns:
            FeatureMetadata or None
        """
        if version:
            key = f"{feature_name}_{version}"
            return self.metadata.get(key)
        else:
            # Return latest version
            matching = {k: v for k, v in self.metadata.items() if k.startswith(f"{feature_name}_")}
            if not matching:
                return None
            # Return the most recently created
            latest = max(matching.values(), key=lambda x: x.created_at)
            return latest

    def list_features(self, status: Optional[str] = None) -> List[FeatureMetadata]:
        """
        List all features, optionally filtered by status.
        
        Args:
            status: Filter by validation status (PASS, FAIL, WARNING)
            
        Returns:
            List of FeatureMetadata
        """
        all_metadata = list(self.metadata.values())
        if status:
            return [m for m in all_metadata if m.validation_status == status]
        return all_metadata

    def get_feature_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive feature report.
        
        Returns:
            Dictionary with feature statistics and details
        """
        all_features = list(self.metadata.values())
        
        total = len(all_features)
        passed = sum(1 for f in all_features if f.validation_status == "PASS")
        failed = sum(1 for f in all_features if f.validation_status == "FAIL")
        warning = sum(1 for f in all_features if f.validation_status == "WARNING")
        pending = sum(1 for f in all_features if f.validation_status == "PENDING")
        
        # Features with future data (should be rejected)
        future_data_features = [f for f in all_features if f.uses_future_data]
        
        # Features with high missing percentage
        high_missing = [f for f in all_features if f.missing_pct > 0.3]
        
        # Features with good IC
        good_ic = [f for f in all_features if f.ic is not None and abs(f.ic) > 0.05]
        
        return {
            "summary": {
                "total_features": total,
                "passed": passed,
                "failed": failed,
                "warning": warning,
                "pending": pending,
                "pass_rate_pct": round(passed / total * 100, 2) if total > 0 else 0,
            },
            "issues": {
                "uses_future_data": len(future_data_features),
                "high_missing_pct": len(high_missing),
                "good_ic_features": len(good_ic),
            },
            "features": [f.to_dict() for f in all_features],
            "generated_at": datetime.now().isoformat(),
        }

    def _load(self) -> None:
        """Load metadata from storage."""
        if not self.storage_path.exists():
            logger.info(f"No existing metadata file at {self.storage_path}, starting fresh")
            return
        
        try:
            with open(self.storage_path) as f:
                data = json.load(f)
            
            for key, meta_dict in data.items():
                self.metadata[key] = FeatureMetadata(
                    feature_name=meta_dict["feature_name"],
                    version=meta_dict["version"],
                    description=meta_dict["description"],
                    parameters=meta_dict["parameters"],
                    uses_future_data=meta_dict["uses_future_data"],
                    missing_pct=meta_dict["missing_pct"],
                    correlation=meta_dict["correlation"],
                    ic=meta_dict["ic"],
                    dtype=meta_dict["dtype"],
                    created_at=datetime.fromisoformat(meta_dict["created_at"]),
                    last_validated=datetime.fromisoformat(meta_dict["last_validated"]),
                    validation_status=meta_dict["validation_status"],
                    validation_errors=meta_dict["validation_errors"],
                )
            
            logger.info(f"Loaded {len(self.metadata)} feature metadata entries")
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")

    def _save(self) -> None:
        """Save metadata to storage."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {key: meta.to_dict() for key, meta in self.metadata.items()}
        
        try:
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(self.metadata)} feature metadata entries")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")


# Global registry instance
_registry: Optional[FeatureMetadataRegistry] = None


def get_feature_registry() -> FeatureMetadataRegistry:
    """Get the global feature metadata registry."""
    global _registry
    if _registry is None:
        _registry = FeatureMetadataRegistry()
    return _registry


def register_feature_from_validation(
    feature_name: str,
    version: str,
    description: str,
    parameters: Dict[str, Any],
    dtype: str,
    validation_result: Any,
) -> FeatureMetadata:
    """
    Register a feature from validation results.
    
    Args:
        feature_name: Name of the feature
        version: Feature version
        description: Human-readable description
        parameters: Feature parameters
        dtype: Data type
        validation_result: FeatureValidationResult from validate_features
        
    Returns:
        FeatureMetadata object
    """
    registry = get_feature_registry()
    
    # Register the feature
    metadata = registry.register(
        feature_name=feature_name,
        version=version,
        description=description,
        parameters=parameters,
        uses_future_data=validation_result.uses_future_data,
        dtype=dtype,
    )
    
    # Update with validation results
    registry.update_validation(
        feature_name=feature_name,
        version=version,
        validation_status="PASS" if validation_result.passed else "FAIL",
        missing_pct=validation_result.missing_pct,
        correlation=validation_result.correlation,
        ic=validation_result.ic,
        validation_errors=validation_result.errors,
    )
    
    return metadata
