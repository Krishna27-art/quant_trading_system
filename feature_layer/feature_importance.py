"""
Feature Importance

Tracks and analyzes feature importance from ML models.
Helps identify which features are most predictive.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import and_

from database.models import FeatureImportance
from sqlalchemy import func


logger = logging.getLogger(__name__)


class FeatureImportanceTracker:
    """
    Feature Importance Tracker.
    
    Responsibilities:
    1. Store feature importance from ML models
    2. Track importance over time
    3. Compare importance across models
    4. Identify top features
    5. Detect feature degradation
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize Feature Importance Tracker.
        
        Args:
            db_session: SQLAlchemy database session
        """
        self.db_session = db_session
    
    def store_feature_importance(
        self,
        model_name: str,
        model_version: str,
        feature_importance: Dict[str, float],
        overwrite: bool = True
    ) -> None:
        """
        Store feature importance scores from a model.
        
        Args:
            model_name: Name of the ML model
            model_version: Version of the model
            feature_importance: Dictionary of feature_name -> importance_score
            overwrite: Whether to overwrite existing records
        """
        logger.info(
            f"Storing feature importance for {model_name} v{model_version} "
            f"with {len(feature_importance)} features"
        )
        
        # Calculate ranks
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for rank, (feature_name, importance_score) in enumerate(sorted_features, start=1):
            # Check if record already exists
            existing = self.db_session.query(FeatureImportance).filter(
                and_(
                    FeatureImportance.model_name == model_name,
                    FeatureImportance.model_version == model_version,
                    FeatureImportance.feature_name == feature_name
                )
            ).first()
            
            if existing and overwrite:
                existing.importance_score = importance_score
                existing.rank = rank
                existing.computed_at = datetime.now()
            elif not existing:
                importance_record = FeatureImportance(
                    model_name=model_name,
                    model_version=model_version,
                    feature_name=feature_name,
                    importance_score=importance_score,
                    rank=rank,
                    computed_at=datetime.now()
                )
                self.db_session.add(importance_record)
        
        self.db_session.commit()
        logger.info(f"Stored feature importance for {model_name} v{model_version}")
    
    def get_feature_importance(
        self,
        model_name: str,
        model_version: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get feature importance for a model.
        
        Args:
            model_name: Name of the model
            model_version: Optional version (if None, gets latest version)
            
        Returns:
            DataFrame with feature importance
        """
        query = self.db_session.query(FeatureImportance).filter(
            FeatureImportance.model_name == model_name
        )
        
        if model_version:
            query = query.filter(FeatureImportance.model_version == model_version)
        else:
            # Get latest version
            subquery = self.db_session.query(
                FeatureImportance.model_name,
                func.max(FeatureImportance.model_version).label('max_version')
            ).filter(
                FeatureImportance.model_name == model_name
            ).group_by(FeatureImportance.model_name).subquery()
            
            query = query.join(
                subquery,
                and_(
                    FeatureImportance.model_name == subquery.c.model_name,
                    FeatureImportance.model_version == subquery.c.max_version
                )
            )
        
        query = query.order_by(FeatureImportance.rank.asc())
        
        records = query.all()
        
        if not records:
            return pd.DataFrame()
        
        data = []
        for record in records:
            data.append({
                'feature_name': record.feature_name,
                'importance_score': record.importance_score,
                'rank': record.rank,
                'model_version': record.model_version,
                'computed_at': record.computed_at,
            })
        
        return pd.DataFrame(data)
    
    def get_top_features(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        n: int = 10
    ) -> pd.DataFrame:
        """
        Get top N features by importance for a model.
        
        Args:
            model_name: Name of the model
            model_version: Optional version
            n: Number of top features to return
            
        Returns:
            DataFrame with top features
        """
        df = self.get_feature_importance(model_name, model_version)
        
        if df.empty:
            return df
        
        return df.head(n)
    
    def compare_feature_importance_across_models(
        self,
        model_names: List[str],
        n: int = 20
    ) -> pd.DataFrame:
        """
        Compare feature importance across multiple models.
        
        Args:
            model_names: List of model names to compare
            n: Number of top features to include
            
        Returns:
            DataFrame with comparison
        """
        comparison_data = []
        
        for model_name in model_names:
            df = self.get_top_features(model_name, n=n)
            
            if not df.empty:
                df['model_name'] = model_name
                comparison_data.append(df)
        
        if not comparison_data:
            return pd.DataFrame()
        
        return pd.concat(comparison_data, ignore_index=True)
    
    def get_feature_importance_history(
        self,
        feature_name: str,
        model_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get historical importance scores for a feature.
        
        Args:
            feature_name: Name of the feature
            model_name: Optional model name filter
            
        Returns:
            DataFrame with historical importance
        """
        query = self.db_session.query(FeatureImportance).filter(
            FeatureImportance.feature_name == feature_name
        )
        
        if model_name:
            query = query.filter(FeatureImportance.model_name == model_name)
        
        query = query.order_by(FeatureImportance.computed_at.asc())
        
        records = query.all()
        
        if not records:
            return pd.DataFrame()
        
        data = []
        for record in records:
            data.append({
                'model_name': record.model_name,
                'model_version': record.model_version,
                'importance_score': record.importance_score,
                'rank': record.rank,
                'computed_at': record.computed_at,
            })
        
        return pd.DataFrame(data)
    
    def calculate_aggregate_importance(
        self,
        feature_names: List[str],
        model_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate aggregate importance across all models.
        
        Args:
            feature_names: List of feature names to analyze
            model_names: Optional list of models to include
            
        Returns:
            DataFrame with aggregate importance
        """
        query = self.db_session.query(FeatureImportance).filter(
            FeatureImportance.feature_name.in_(feature_names)
        )
        
        if model_names:
            query = query.filter(FeatureImportance.model_name.in_(model_names))
        
        records = query.all()
        
        if not records:
            return pd.DataFrame()
        
        # Aggregate by feature
        aggregation = {}
        for record in records:
            if record.feature_name not in aggregation:
                aggregation[record.feature_name] = {
                    'total_importance': 0.0,
                    'count': 0,
                    'avg_rank': 0.0,
                    'models': set(),
                }
            
            aggregation[record.feature_name]['total_importance'] += record.importance_score
            aggregation[record.feature_name]['count'] += 1
            aggregation[record.feature_name]['avg_rank'] += record.rank
            aggregation[record.feature_name]['models'].add(record.model_name)
        
        data = []
        for feature_name, stats in aggregation.items():
            data.append({
                'feature_name': feature_name,
                'avg_importance': stats['total_importance'] / stats['count'],
                'avg_rank': stats['avg_rank'] / stats['count'],
                'model_count': stats['count'],
                'models': list(stats['models']),
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('avg_importance', ascending=False)
        
        return df
    
    def detect_feature_degradation(
        self,
        feature_name: str,
        model_name: str,
        threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Detect if a feature's importance has degraded over time.
        
        Args:
            feature_name: Name of the feature
            model_name: Name of the model
            threshold: Percentage drop considered as degradation
            
        Returns:
            Dictionary with degradation analysis
        """
        history = self.get_feature_importance_history(feature_name, model_name)
        
        if len(history) < 2:
            return {
                'feature_name': feature_name,
                'degraded': False,
                'reason': 'Insufficient history'
            }
        
        latest = history.iloc[-1]
        earliest = history.iloc[0]
        
        importance_drop = (
            (earliest['importance_score'] - latest['importance_score']) /
            earliest['importance_score']
        ) if earliest['importance_score'] > 0 else 0
        
        degraded = importance_drop > threshold
        
        return {
            'feature_name': feature_name,
            'degraded': degraded,
            'importance_drop_pct': round(importance_drop * 100, 2),
            'latest_importance': latest['importance_score'],
            'earliest_importance': earliest['importance_score'],
            'latest_rank': latest['rank'],
            'earliest_rank': earliest['rank'],
            'reason': 'Importance dropped significantly' if degraded else 'No significant degradation'
        }
