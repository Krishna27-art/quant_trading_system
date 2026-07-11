"""
Feature Generator Engine

Core engine that discovers, calculates, and stores all features.
This is the heart of the Feature Laboratory.
"""

import logging
import time
from datetime import datetime, date
from typing import Dict, List, Optional, Set, Type, Any
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import and_

from database.models import Feature, FeatureMetadata as DBFeatureMetadata
from feature_layer.base_feature import BaseFeature, FeatureCategory, FeatureResult


logger = logging.getLogger(__name__)


class FeatureGenerator:
    """
    Feature Generator Engine.
    
    Responsibilities:
    1. Discover all available feature classes
    2. Filter enabled features
    3. Calculate features for given data
    4. Store features in database
    5. Handle feature versioning
    6. Track computation metrics
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize Feature Generator.
        
        Args:
            db_session: SQLAlchemy database session
        """
        self.db_session = db_session
        self._feature_registry: Dict[str, Type[BaseFeature]] = {}
        self._enabled_features: Set[str] = set()
        self._load_feature_registry()
        self._load_enabled_features()
    
    def _load_feature_registry(self) -> None:
        """
        Discover and register all available feature classes.
        
        This scans the feature_layer directory for all feature classes
        and registers them in the internal registry.
        """
        import importlib
        import inspect
        from pathlib import Path
        
        feature_layer_path = Path(__file__).parent
        
        # Scan each category directory
        for category_dir in feature_layer_path.iterdir():
            if category_dir.is_dir() and not category_dir.name.startswith('_'):
                category_name = category_dir.name
                
                # Import all Python files in the category directory
                for py_file in category_dir.glob('*.py'):
                    if py_file.name.startswith('_'):
                        continue
                    
                    module_name = f"feature_layer.{category_name}.{py_file.stem}"
                    try:
                        module = importlib.import_module(module_name)
                        
                        # Find all BaseFeature subclasses in the module
                        for name, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) and 
                                issubclass(obj, BaseFeature) and 
                                obj != BaseFeature):
                                
                                feature_instance = obj()
                                feature_name = feature_instance.metadata.feature_name
                                
                                self._feature_registry[feature_name] = obj
                                logger.info(f"Registered feature: {feature_name}")
                                
                    except Exception as e:
                        logger.error(f"Failed to load module {module_name}: {e}")
        
        logger.info(f"Feature registry loaded with {len(self._feature_registry)} features")
    
    def _load_enabled_features(self) -> None:
        """Load enabled features from database metadata."""
        try:
            enabled_metadata = self.db_session.query(DBFeatureMetadata).filter(
                DBFeatureMetadata.is_enabled == True
            ).all()
            
            self._enabled_features = {meta.feature_name for meta in enabled_metadata}
            logger.info(f"Loaded {len(self._enabled_features)} enabled features")
            
        except Exception as e:
            logger.error(f"Failed to load enabled features: {e}")
            # If database query fails, enable all registered features
            self._enabled_features = set(self._feature_registry.keys())
    
    def get_available_features(self) -> List[str]:
        """Get list of all available feature names."""
        return list(self._feature_registry.keys())
    
    def get_enabled_features(self) -> List[str]:
        """Get list of enabled feature names."""
        return list(self._enabled_features)
    
    def enable_feature(self, feature_name: str) -> None:
        """
        Enable a feature.
        
        Args:
            feature_name: Name of the feature to enable
        """
        if feature_name not in self._feature_registry:
            raise ValueError(f"Feature {feature_name} not found in registry")
        
        self._enabled_features.add(feature_name)
        
        # Update database
        metadata = self.db_session.query(DBFeatureMetadata).filter(
            DBFeatureMetadata.feature_name == feature_name
        ).first()
        
        if metadata:
            metadata.is_enabled = True
            metadata.last_updated = datetime.now()
            self.db_session.commit()
        
        logger.info(f"Enabled feature: {feature_name}")
    
    def disable_feature(self, feature_name: str) -> None:
        """
        Disable a feature.
        
        Args:
            feature_name: Name of the feature to disable
        """
        if feature_name in self._enabled_features:
            self._enabled_features.remove(feature_name)
        
        # Update database
        metadata = self.db_session.query(DBFeatureMetadata).filter(
            DBFeatureMetadata.feature_name == feature_name
        ).first()
        
        if metadata:
            metadata.is_enabled = False
            metadata.last_updated = datetime.now()
            self.db_session.commit()
        
        logger.info(f"Disabled feature: {feature_name}")
    
    def register_feature_metadata(self, feature: BaseFeature) -> None:
        """
        Register or update feature metadata in database.
        
        Args:
            feature: Feature instance
        """
        metadata = feature.metadata
        
        # Check if metadata already exists
        existing = self.db_session.query(DBFeatureMetadata).filter(
            DBFeatureMetadata.feature_name == metadata.feature_name
        ).first()
        
        import json
        
        if existing:
            # Update existing metadata
            existing.description = metadata.description
            existing.feature_category = metadata.category.value
            existing.timeframe = metadata.timeframe.value
            existing.required_columns = json.dumps(metadata.required_columns)
            existing.output_range = metadata.output_range
            existing.version = metadata.version
            existing.computation_method = metadata.computation_method
            existing.assumptions = metadata.assumptions
            existing.limitations = metadata.limitations
            existing.references = metadata.references
            existing.last_updated = datetime.now()
        else:
            # Create new metadata
            db_metadata = DBFeatureMetadata(
                feature_name=metadata.feature_name,
                description=metadata.description,
                feature_category=metadata.category.value,
                timeframe=metadata.timeframe.value,
                required_columns=json.dumps(metadata.required_columns),
                output_range=metadata.output_range,
                version=metadata.version,
                author=metadata.author,
                computation_method=metadata.computation_method,
                assumptions=metadata.assumptions,
                limitations=metadata.limitations,
                references=metadata.references,
                is_enabled=True,
            )
            self.db_session.add(db_metadata)
        
        self.db_session.commit()
        logger.info(f"Registered metadata for feature: {metadata.feature_name}")
    
    def generate_features(
        self,
        symbol: str,
        data: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        store: bool = True
    ) -> Dict[str, FeatureResult]:
        """
        Generate features for a symbol.
        
        Args:
            symbol: Stock symbol
            data: DataFrame with OHLCV data
            feature_names: Optional list of specific features to compute.
                          If None, computes all enabled features.
            store: Whether to store features in database
            
        Returns:
            Dictionary of feature_name -> FeatureResult
        """
        if feature_names is None:
            feature_names = list(self._enabled_features)
        
        results = {}
        total_start = time.time()
        
        for feature_name in feature_names:
            if feature_name not in self._feature_registry:
                logger.warning(f"Feature {feature_name} not in registry, skipping")
                continue
            
            if feature_name not in self._enabled_features:
                logger.warning(f"Feature {feature_name} is disabled, skipping")
                continue
            
            try:
                # Get feature class and instantiate
                feature_class = self._feature_registry[feature_name]
                feature = feature_class()
                
                # Register metadata if not already registered
                self.register_feature_metadata(feature)
                
                # Compute feature
                result = feature.compute_with_validation(data)
                results[feature_name] = result
                
                logger.info(
                    f"Computed {feature_name} for {symbol} "
                    f"in {result.computation_time_ms:.2f}ms"
                )
                
            except Exception as e:
                logger.error(f"Failed to compute {feature_name} for {symbol}: {e}")
                continue
        
        total_time = (time.time() - total_start) * 1000
        logger.info(
            f"Generated {len(results)} features for {symbol} "
            f"in {total_time:.2f}ms"
        )
        
        # Store features in database if requested
        if store and results:
            self._store_features(symbol, data.index, results)
        
        return results
    
    def _store_features(
        self,
        symbol: str,
        index: pd.DatetimeIndex,
        results: Dict[str, FeatureResult]
    ) -> None:
        """
        Store computed features in database.
        
        Args:
            symbol: Stock symbol
            index: DatetimeIndex of the data
            results: Dictionary of feature results
        """
        try:
            bulk_records = []
            
            for feature_name, result in results.items():
                feature_version = result.metadata.version
                feature_category = result.metadata.category.value
                
                # Create a record for each timestamp
                for timestamp, value in result.values.items():
                    if pd.isna(value):
                        continue
                    
                    record = Feature(
                        symbol=symbol,
                        date=timestamp.date() if isinstance(timestamp, pd.Timestamp) else timestamp,
                        feature_name=feature_name,
                        feature_category=feature_category,
                        feature_version=feature_version,
                        feature_value=float(value),
                        created_at=datetime.now()
                    )
                    bulk_records.append(record)
            
            # Bulk insert
            if bulk_records:
                self.db_session.bulk_save_objects(bulk_records)
                self.db_session.commit()
                logger.info(f"Stored {len(bulk_records)} feature records for {symbol}")
            
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Failed to store features for {symbol}: {e}")
    
    def get_features(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Retrieve features from database.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            feature_names: Optional list of feature names to retrieve
            
        Returns:
            DataFrame with features in wide format
        """
        query = self.db_session.query(Feature).filter(
            and_(
                Feature.symbol == symbol,
                Feature.date >= start_date,
                Feature.date <= end_date
            )
        )
        
        if feature_names:
            query = query.filter(Feature.feature_name.in_(feature_names))
        
        records = query.all()
        
        if not records:
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = []
        for record in records:
            data.append({
                'date': record.date,
                'feature_name': record.feature_name,
                'feature_value': record.feature_value,
                'feature_version': record.feature_version,
            })
        
        df = pd.DataFrame(data)
        
        # Pivot to wide format
        df = df.pivot(
            index='date',
            columns='feature_name',
            values='feature_value'
        )
        
        df.index = pd.to_datetime(df.index)
        return df
    
    def get_latest_features(
        self,
        symbol: str,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Get the latest feature values for a symbol.
        
        Args:
            symbol: Stock symbol
            feature_names: Optional list of feature names to retrieve
            
        Returns:
            Dictionary of feature_name -> value
        """
        subquery = self.db_session.query(
            Feature.symbol,
            func.max(Feature.date).label('max_date')
        ).filter(Feature.symbol == symbol).group_by(Feature.symbol).subquery()
        
        query = self.db_session.query(Feature).join(
            subquery,
            and_(
                Feature.symbol == subquery.c.symbol,
                Feature.date == subquery.c.max_date
            )
        )
        
        if feature_names:
            query = query.filter(Feature.feature_name.in_(feature_names))
        
        records = query.all()
        
        return {record.feature_name: record.feature_value for record in records}
    
    def get_feature_statistics(
        self,
        feature_name: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Get statistics for a specific feature.
        
        Args:
            feature_name: Feature name
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            Dictionary with statistics
        """
        query = self.db_session.query(Feature).filter(
            Feature.feature_name == feature_name
        )
        
        if start_date:
            query = query.filter(Feature.date >= start_date)
        if end_date:
            query = query.filter(Feature.date <= end_date)
        
        records = query.all()
        
        if not records:
            return {}
        
        values = [record.feature_value for record in records]
        
        return {
            'feature_name': feature_name,
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'null_count': sum(1 for v in values if pd.isna(v)),
        }
