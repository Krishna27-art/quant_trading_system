"""
Feature Versioning System

Manages feature versions to ensure reproducibility and enable A/B testing.
Never overwrites features - always creates new versions.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import and_

from database.models import Feature, FeatureMetadata as DBFeatureMetadata


logger = logging.getLogger(__name__)


class FeatureVersionManager:
    """
    Feature Version Manager.
    
    Responsibilities:
    1. Create new feature versions
    2. Track version history
    3. Enable/disable specific versions
    4. Compare versions
    5. Rollback to previous versions
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize Feature Version Manager.
        
        Args:
            db_session: SQLAlchemy database session
        """
        self.db_session = db_session
    
    def create_new_version(
        self,
        feature_name: str,
        changes_description: str,
        author: str = "system"
    ) -> str:
        """
        Create a new version of a feature.
        
        Args:
            feature_name: Name of the feature
            changes_description: Description of changes
            author: Person making the changes
            
        Returns:
            New version string (e.g., "1.1", "2.0")
        """
        # Get current version
        current_metadata = self.db_session.query(DBFeatureMetadata).filter(
            DBFeatureMetadata.feature_name == feature_name
        ).first()
        
        if not current_metadata:
            raise ValueError(f"Feature {feature_name} not found")
        
        current_version = current_metadata.version
        
        # Parse version and increment
        new_version = self._increment_version(current_version)
        
        # Update metadata with new version
        current_metadata.version = new_version
        current_metadata.last_updated = datetime.now()
        
        self.db_session.commit()
        
        logger.info(
            f"Created new version {new_version} for feature {feature_name}. "
            f"Changes: {changes_description}"
        )
        
        return new_version
    
    def _increment_version(self, version: str) -> str:
        """
        Increment version string.
        
        Args:
            version: Current version string (e.g., "1.0", "2.1")
            
        Returns:
            Incremented version string
        """
        try:
            parts = version.split('.')
            if len(parts) == 1:
                major = int(parts[0])
                return f"{major + 1}.0"
            elif len(parts) == 2:
                major = int(parts[0])
                minor = int(parts[1])
                return f"{major}.{minor + 1}"
            else:
                # Default to 1.0 if format is unexpected
                return "1.0"
        except (ValueError, IndexError):
            return "1.0"
    
    def get_version_history(
        self,
        feature_name: str
    ) -> List[Dict[str, Any]]:
        """
        Get version history for a feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            List of version history records
        """
        # This is a simplified implementation
        # In production, you'd have a separate feature_versions table
        # to track full history
        
        metadata = self.db_session.query(DBFeatureMetadata).filter(
            DBFeatureMetadata.feature_name == feature_name
        ).first()
        
        if not metadata:
            return []
        
        return [{
            'feature_name': metadata.feature_name,
            'version': metadata.version,
            'author': metadata.author,
            'created_at': metadata.created_at,
            'last_updated': metadata.last_updated,
        }]
    
    def compare_versions(
        self,
        feature_name: str,
        version_1: str,
        version_2: str
    ) -> Dict[str, Any]:
        """
        Compare two versions of a feature.
        
        Args:
            feature_name: Name of the feature
            version_1: First version
            version_2: Second version
            
        Returns:
            Dictionary with comparison results
        """
        # Get data for both versions
        query_1 = self.db_session.query(Feature).filter(
            and_(
                Feature.feature_name == feature_name,
                Feature.feature_version == version_1
            )
        )
        
        query_2 = self.db_session.query(Feature).filter(
            and_(
                Feature.feature_name == feature_name,
                Feature.feature_version == version_2
            )
        )
        
        count_1 = query_1.count()
        count_2 = query_2.count()
        
        # Get sample values for comparison
        sample_1 = query_1.limit(100).all()
        sample_2 = query_2.limit(100).all()
        
        values_1 = [f.feature_value for f in sample_1]
        values_2 = [f.feature_value for f in sample_2]
        
        return {
            'feature_name': feature_name,
            'version_1': version_1,
            'version_2': version_2,
            'count_1': count_1,
            'count_2': count_2,
            'mean_1': np.mean(values_1) if values_1 else None,
            'mean_2': np.mean(values_2) if values_2 else None,
            'std_1': np.std(values_1) if values_1 else None,
            'std_2': np.std(values_2) if values_2 else None,
        }
    
    def rollback_version(
        self,
        feature_name: str,
        target_version: str
    ) -> None:
        """
        Rollback a feature to a previous version.
        
        Note: This doesn't delete data, just changes the active version
        in the metadata.
        
        Args:
            feature_name: Name of the feature
            target_version: Version to rollback to
        """
        metadata = self.db_session.query(DBFeatureMetadata).filter(
            DBFeatureMetadata.feature_name == feature_name
        ).first()
        
        if not metadata:
            raise ValueError(f"Feature {feature_name} not found")
        
        # Check if target version exists in data
        version_exists = self.db_session.query(Feature).filter(
            and_(
                Feature.feature_name == feature_name,
                Feature.feature_version == target_version
            )
        ).first()
        
        if not version_exists:
            raise ValueError(f"Version {target_version} not found for feature {feature_name}")
        
        # Update metadata
        metadata.version = target_version
        metadata.last_updated = datetime.now()
        
        self.db_session.commit()
        
        logger.info(f"Rolled back {feature_name} to version {target_version}")
    
    def get_active_version(self, feature_name: str) -> str:
        """
        Get the active version for a feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Active version string
        """
        metadata = self.db_session.query(DBFeatureMetadata).filter(
            DBFeatureMetadata.feature_name == feature_name
        ).first()
        
        if not metadata:
            raise ValueError(f"Feature {feature_name} not found")
        
        return metadata.version
    
    def list_all_versions(
        self,
        feature_name: str
    ) -> List[str]:
        """
        List all available versions for a feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            List of version strings
        """
        query = self.db_session.query(Feature.feature_version).filter(
            Feature.feature_name == feature_name
        ).distinct()
        
        versions = [v[0] for v in query.all()]
        versions.sort()
        
        return versions
    
    def promote_version(
        self,
        feature_name: str,
        version: str
    ) -> None:
        """
        Promote a version to be the active version.
        
        Args:
            feature_name: Name of the feature
            version: Version to promote
        """
        self.rollback_version(feature_name, version)
        logger.info(f"Promoted {feature_name} version {version} to active")
