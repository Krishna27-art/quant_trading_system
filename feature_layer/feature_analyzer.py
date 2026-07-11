"""
Feature Analyzer

Analyzes feature performance to determine predictive quality.
Calculates win rate, returns, accuracy, and other metrics for each feature.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from database.models import Feature, FeatureQuality, Prediction


logger = logging.getLogger(__name__)


class FeatureAnalyzer:
    """
    Feature Analyzer Engine.
    
    Responsibilities:
    1. Calculate feature performance metrics (win rate, returns, accuracy)
    2. Compare features against each other
    3. Identify which features are predictive
    4. Generate feature quality scores
    5. Track feature performance over time
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize Feature Analyzer.
        
        Args:
            db_session: SQLAlchemy database session
        """
        self.db_session = db_session
    
    def calculate_feature_performance(
        self,
        feature_name: str,
        start_date: date,
        end_date: date,
        prediction_horizon_days: int = 5,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate performance metrics for a single feature.
        
        Args:
            feature_name: Name of the feature to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            prediction_horizon_days: Number of days to look ahead for returns
            threshold: Optional threshold for binary classification (e.g., RSI > 70)
            
        Returns:
            Dictionary with performance metrics
        """
        logger.info(f"Analyzing feature: {feature_name} from {start_date} to {end_date}")
        
        # Get feature values
        feature_data = self._get_feature_data(feature_name, start_date, end_date)
        
        if feature_data.empty:
            logger.warning(f"No data found for feature: {feature_name}")
            return {}
        
        # Get price data for returns calculation
        price_data = self._get_price_data_for_symbols(
            feature_data['symbol'].unique(),
            start_date,
            end_date + timedelta(days=prediction_horizon_days)
        )
        
        if price_data.empty:
            logger.warning(f"No price data found for feature analysis")
            return {}
        
        # Calculate forward returns
        results = self._calculate_metrics(
            feature_data,
            price_data,
            prediction_horizon_days,
            threshold
        )
        
        return results
    
    def _get_feature_data(
        self,
        feature_name: str,
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """Get feature data from database."""
        query = self.db_session.query(Feature).filter(
            and_(
                Feature.feature_name == feature_name,
                Feature.date >= start_date,
                Feature.date <= end_date
            )
        )
        
        records = query.all()
        
        if not records:
            return pd.DataFrame()
        
        data = []
        for record in records:
            data.append({
                'symbol': record.symbol,
                'date': record.date,
                'feature_value': record.feature_value,
                'feature_version': record.feature_version,
            })
        
        return pd.DataFrame(data)
    
    def _get_price_data_for_symbols(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """
        Get price data for specified symbols.
        
        Note: This is a placeholder. In production, this would query
        the actual price data table (e.g., OHLCV data).
        """
        # TODO: Implement actual price data retrieval
        # For now, return empty DataFrame
        return pd.DataFrame()
    
    def _calculate_metrics(
        self,
        feature_data: pd.DataFrame,
        price_data: pd.DataFrame,
        horizon_days: int,
        threshold: Optional[float]
    ) -> Dict[str, Any]:
        """
        Calculate performance metrics.
        
        Args:
            feature_data: DataFrame with feature values
            price_data: DataFrame with price data
            horizon_days: Prediction horizon in days
            threshold: Optional threshold for binary classification
            
        Returns:
            Dictionary with calculated metrics
        """
        # This is a simplified implementation
        # In production, this would:
        # 1. Merge feature data with price data
        # 2. Calculate forward returns
        # 3. Apply threshold if provided
        # 4. Calculate win rate, average return, sharpe ratio, etc.
        
        results = {
            'feature_name': feature_data['feature_version'].iloc[0] if 'feature_version' in feature_data.columns else 'unknown',
            'sample_size': len(feature_data),
            'win_rate': 0.0,
            'average_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
        }
        
        return results
    
    def compare_features(
        self,
        feature_names: List[str],
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """
        Compare multiple features side by side.
        
        Args:
            feature_names: List of feature names to compare
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        
        for feature_name in feature_names:
            metrics = self.calculate_feature_performance(
                feature_name,
                start_date,
                end_date
            )
            
            if metrics:
                comparison_data.append(metrics)
        
        if not comparison_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(comparison_data)
        
        # Rank features by win rate
        if 'win_rate' in df.columns:
            df['rank'] = df['win_rate'].rank(ascending=False)
        
        return df
    
    def calculate_feature_quality_score(
        self,
        feature_name: str,
        start_date: date,
        end_date: date
    ) -> float:
        """
        Calculate overall quality score for a feature (0-100).
        
        Quality score is based on:
        - Win rate (40% weight)
        - Sharpe ratio (30% weight)
        - Profit factor (20% weight)
        - Sample size (10% weight)
        
        Args:
            feature_name: Name of the feature
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Quality score between 0 and 100
        """
        metrics = self.calculate_feature_performance(
            feature_name,
            start_date,
            end_date
        )
        
        if not metrics:
            return 0.0
        
        # Normalize and weight components
        win_rate_score = min(metrics.get('win_rate', 0) * 100, 100) * 0.4
        sharpe_score = min(max(metrics.get('sharpe_ratio', 0) * 10, 0), 100) * 0.3
        profit_factor_score = min(metrics.get('profit_factor', 0) * 10, 100) * 0.2
        sample_score = min(metrics.get('sample_size', 0) / 100, 100) * 0.1
        
        quality_score = win_rate_score + sharpe_score + profit_factor_score + sample_score
        
        return round(quality_score, 2)
    
    def update_feature_quality_in_db(
        self,
        feature_name: str,
        start_date: date,
        end_date: date
    ) -> None:
        """
        Calculate and store feature quality in database.
        
        Args:
            feature_name: Name of the feature
            start_date: Start date for analysis
            end_date: End date for analysis
        """
        metrics = self.calculate_feature_performance(
            feature_name,
            start_date,
            end_date
        )
        
        if not metrics:
            logger.warning(f"No metrics to store for feature: {feature_name}")
            return
        
        quality_score = self.calculate_feature_quality_score(
            feature_name,
            start_date,
            end_date
        )
        
        # Create or update feature quality record
        existing = self.db_session.query(FeatureQuality).filter(
            and_(
                FeatureQuality.feature_name == feature_name,
                FeatureQuality.evaluation_period_end == end_date
            )
        ).first()
        
        if existing:
            existing.quality_score = quality_score
            existing.win_rate = metrics.get('win_rate')
            existing.average_return = metrics.get('average_return')
            existing.sharpe_ratio = metrics.get('sharpe_ratio')
            existing.sortino_ratio = metrics.get('sortino_ratio')
            existing.max_drawdown = metrics.get('max_drawdown')
            existing.profit_factor = metrics.get('profit_factor')
            existing.sample_size = metrics.get('sample_size')
            existing.computed_at = datetime.now()
        else:
            quality_record = FeatureQuality(
                feature_name=feature_name,
                feature_version=metrics.get('feature_name', '1.0'),
                quality_score=quality_score,
                win_rate=metrics.get('win_rate'),
                average_return=metrics.get('average_return'),
                sharpe_ratio=metrics.get('sharpe_ratio'),
                sortino_ratio=metrics.get('sortino_ratio'),
                max_drawdown=metrics.get('max_drawdown'),
                profit_factor=metrics.get('profit_factor'),
                sample_size=metrics.get('sample_size'),
                evaluation_period_start=start_date,
                evaluation_period_end=end_date,
                computed_at=datetime.now()
            )
            self.db_session.add(quality_record)
        
        self.db_session.commit()
        logger.info(f"Updated quality score for {feature_name}: {quality_score}")
    
    def get_top_features(
        self,
        n: int = 10,
        category: Optional[str] = None,
        min_sample_size: int = 100
    ) -> pd.DataFrame:
        """
        Get top performing features by quality score.
        
        Args:
            n: Number of top features to return
            category: Optional category filter
            min_sample_size: Minimum sample size required
            
        Returns:
            DataFrame with top features
        """
        query = self.db_session.query(FeatureQuality).filter(
            FeatureQuality.sample_size >= min_sample_size
        )
        
        if category:
            # Join with feature_metadata to filter by category
            from database.models import FeatureMetadata as DBFeatureMetadata
            query = query.join(DBFeatureMetadata).filter(
                DBFeatureMetadata.feature_category == category
            )
        
        # Order by quality score descending and limit
        query = query.order_by(FeatureQuality.quality_score.desc()).limit(n)
        
        records = query.all()
        
        if not records:
            return pd.DataFrame()
        
        data = []
        for record in records:
            data.append({
                'feature_name': record.feature_name,
                'quality_score': record.quality_score,
                'win_rate': record.win_rate,
                'average_return': record.average_return,
                'sharpe_ratio': record.sharpe_ratio,
                'sample_size': record.sample_size,
                'evaluation_period_end': record.evaluation_period_end,
            })
        
        return pd.DataFrame(data)
