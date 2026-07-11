"""
Feature Correlation

Analyzes correlations between features to identify redundancy.
Helps reduce feature dimensionality by removing highly correlated features.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set, Any
import pandas as pd
import numpy as np
from scipy import stats
from sqlalchemy.orm import Session
from sqlalchemy import and_

from database.models import FeatureCorrelation
from sqlalchemy import func


logger = logging.getLogger(__name__)


class FeatureCorrelationAnalyzer:
    """
    Feature Correlation Analyzer.
    
    Responsibilities:
    1. Calculate pairwise correlations between features
    2. Identify highly correlated feature pairs
    3. Suggest features to remove (redundancy reduction)
    4. Track correlation changes over time
    5. Store correlation matrix in database
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize Feature Correlation Analyzer.
        
        Args:
            db_session: SQLAlchemy database session
        """
        self.db_session = db_session
    
    def calculate_correlation_matrix(
        self,
        feature_data: pd.DataFrame,
        method: str = 'pearson'
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for features.
        
        Args:
            feature_data: DataFrame with features as columns
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Correlation matrix as DataFrame
        """
        logger.info(f"Calculating {method} correlation matrix for {len(feature_data.columns)} features")
        
        # Calculate correlation matrix
        corr_matrix = feature_data.corr(method=method)
        
        return corr_matrix
    
    def calculate_correlation_with_pvalues(
        self,
        feature_data: pd.DataFrame,
        method: str = 'pearson'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate correlation matrix with p-values.
        
        Args:
            feature_data: DataFrame with features as columns
            method: Correlation method
            
        Returns:
            Tuple of (correlation_matrix, pvalue_matrix)
        """
        logger.info(f"Calculating {method} correlation with p-values")
        
        features = feature_data.columns
        n = len(features)
        
        corr_matrix = np.zeros((n, n))
        pvalue_matrix = np.zeros((n, n))
        
        for i, feat1 in enumerate(features):
            for j, feat2 in enumerate(features):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    pvalue_matrix[i, j] = 0.0
                else:
                    x = feature_data[feat1].dropna()
                    y = feature_data[feat2].dropna()
                    
                    # Align indices
                    common_idx = x.index.intersection(y.index)
                    x = x.loc[common_idx]
                    y = y.loc[common_idx]
                    
                    if len(x) < 3:
                        corr_matrix[i, j] = np.nan
                        pvalue_matrix[i, j] = np.nan
                        continue
                    
                    if method == 'pearson':
                        corr, pvalue = stats.pearsonr(x, y)
                    elif method == 'spearman':
                        corr, pvalue = stats.spearmanr(x, y)
                    elif method == 'kendall':
                        corr, pvalue = stats.kendalltau(x, y)
                    else:
                        raise ValueError(f"Unknown correlation method: {method}")
                    
                    corr_matrix[i, j] = corr
                    pvalue_matrix[i, j] = pvalue
        
        corr_df = pd.DataFrame(corr_matrix, index=features, columns=features)
        pvalue_df = pd.DataFrame(pvalue_matrix, index=features, columns=features)
        
        return corr_df, pvalue_df
    
    def identify_highly_correlated_features(
        self,
        correlation_matrix: pd.DataFrame,
        threshold: float = 0.9,
        pvalue_threshold: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        Identify pairs of highly correlated features.
        
        Args:
            correlation_matrix: Correlation matrix
            threshold: Correlation threshold (default: 0.9)
            pvalue_threshold: P-value threshold for significance
            
        Returns:
            List of dictionaries with correlated feature pairs
        """
        logger.info(f"Identifying features with correlation > {threshold}")
        
        correlated_pairs = []
        features = correlation_matrix.columns
        
        for i, feat1 in enumerate(features):
            for j, feat2 in enumerate(features):
                if i >= j:  # Avoid duplicates and self-correlation
                    continue
                
                corr = correlation_matrix.iloc[i, j]
                
                if abs(corr) >= threshold:
                    correlated_pairs.append({
                        'feature_1': feat1,
                        'feature_2': feat2,
                        'correlation': corr,
                        'abs_correlation': abs(corr),
                    })
        
        # Sort by absolute correlation
        correlated_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        logger.info(f"Found {len(correlated_pairs)} highly correlated feature pairs")
        return correlated_pairs
    
    def suggest_features_to_remove(
        self,
        correlation_matrix: pd.DataFrame,
        threshold: float = 0.9,
        importance_scores: Optional[Dict[str, float]] = None
    ) -> List[str]:
        """
        Suggest features to remove based on correlation.
        
        Args:
            correlation_matrix: Correlation matrix
            threshold: Correlation threshold
            importance_scores: Optional feature importance scores
            
        Returns:
            List of feature names to remove
        """
        correlated_pairs = self.identify_highly_correlated_features(
            correlation_matrix,
            threshold
        )
        
        if not correlated_pairs:
            return []
        
        # Build graph of correlated features
        feature_groups = {}
        for pair in correlated_pairs:
            feat1, feat2 = pair['feature_1'], pair['feature_2']
            
            if feat1 not in feature_groups:
                feature_groups[feat1] = set()
            if feat2 not in feature_groups:
                feature_groups[feat2] = set()
            
            feature_groups[feat1].add(feat2)
            feature_groups[feat2].add(feat1)
        
        # For each group, keep the most important feature
        features_to_remove = []
        visited = set()
        
        for feature in feature_groups:
            if feature in visited:
                continue
            
            # Get connected component
            group = self._get_connected_component(feature, feature_groups, visited)
            
            if len(group) > 1:
                # Sort by importance if available, otherwise keep first
                if importance_scores:
                    sorted_group = sorted(
                        group,
                        key=lambda f: importance_scores.get(f, 0),
                        reverse=True
                    )
                else:
                    sorted_group = list(group)
                
                # Keep the first, remove the rest
                features_to_remove.extend(sorted_group[1:])
        
        return features_to_remove
    
    def _get_connected_component(
        self,
        feature: str,
        feature_groups: Dict[str, Set[str]],
        visited: Set[str]
    ) -> Set[str]:
        """Get connected component using DFS."""
        component = set()
        stack = [feature]
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            
            visited.add(current)
            component.add(current)
            
            if current in feature_groups:
                for neighbor in feature_groups[current]:
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        return component
    
    def store_correlation_matrix(
        self,
        correlation_matrix: pd.DataFrame,
        pvalue_matrix: Optional[pd.DataFrame] = None,
        sample_size: Optional[int] = None
    ) -> None:
        """
        Store correlation matrix in database.
        
        Args:
            correlation_matrix: Correlation matrix
            pvalue_matrix: Optional p-value matrix
            sample_size: Sample size used for calculation
        """
        logger.info("Storing correlation matrix in database")
        
        features = correlation_matrix.columns
        n = len(features)
        
        for i, feat1 in enumerate(features):
            for j, feat2 in enumerate(features):
                if i >= j:  # Store each pair only once
                    continue
                
                corr = correlation_matrix.iloc[i, j]
                pvalue = pvalue_matrix.iloc[i, j] if pvalue_matrix is not None else None
                
                # Check if record already exists
                existing = self.db_session.query(FeatureCorrelation).filter(
                    and_(
                        FeatureCorrelation.feature_1 == feat1,
                        FeatureCorrelation.feature_2 == feat2
                    )
                ).first()
                
                if existing:
                    existing.correlation_coefficient = corr
                    existing.p_value = pvalue
                    existing.sample_size = sample_size
                    existing.computed_at = datetime.now()
                else:
                    correlation_record = FeatureCorrelation(
                        feature_1=feat1,
                        feature_2=feat2,
                        correlation_coefficient=corr,
                        p_value=pvalue,
                        sample_size=sample_size,
                        computed_at=datetime.now()
                    )
                    self.db_session.add(correlation_record)
        
        self.db_session.commit()
        logger.info("Stored correlation matrix in database")
    
    def get_correlation(
        self,
        feature_1: str,
        feature_2: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get correlation between two features.
        
        Args:
            feature_1: First feature name
            feature_2: Second feature name
            
        Returns:
            Dictionary with correlation data or None
        """
        record = self.db_session.query(FeatureCorrelation).filter(
            and_(
                FeatureCorrelation.feature_1 == feature_1,
                FeatureCorrelation.feature_2 == feature_2
            )
        ).first()
        
        if not record:
            # Try reverse order
            record = self.db_session.query(FeatureCorrelation).filter(
                and_(
                    FeatureCorrelation.feature_1 == feature_2,
                    FeatureCorrelation.feature_2 == feature_1
                )
            ).first()
        
        if not record:
            return None
        
        return {
            'feature_1': record.feature_1,
            'feature_2': record.feature_2,
            'correlation_coefficient': record.correlation_coefficient,
            'p_value': record.p_value,
            'sample_size': record.sample_size,
            'computed_at': record.computed_at,
        }
    
    def get_all_correlations(
        self,
        min_correlation: float = 0.0
    ) -> pd.DataFrame:
        """
        Get all correlations above a threshold.
        
        Args:
            min_correlation: Minimum absolute correlation
            
        Returns:
            DataFrame with correlations
        """
        query = self.db_session.query(FeatureCorrelation).filter(
            func.abs(FeatureCorrelation.correlation_coefficient) >= min_correlation
        )
        
        records = query.all()
        
        if not records:
            return pd.DataFrame()
        
        data = []
        for record in records:
            data.append({
                'feature_1': record.feature_1,
                'feature_2': record.feature_2,
                'correlation': record.correlation_coefficient,
                'abs_correlation': abs(record.correlation_coefficient),
                'p_value': record.p_value,
                'sample_size': record.sample_size,
                'computed_at': record.computed_at,
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('abs_correlation', ascending=False)
        
        return df
