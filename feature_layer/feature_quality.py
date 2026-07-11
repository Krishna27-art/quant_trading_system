"""
Feature Quality Scoring System

Comprehensive quality scoring for features based on multiple dimensions.
Combines predictive power, stability, and uniqueness into a single score.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from database.models import FeatureQuality, FeatureMetadata as DBFeatureMetadata


logger = logging.getLogger(__name__)


class FeatureQualityScorer:
    """
    Feature Quality Scoring System.
    
    Responsibilities:
    1. Calculate comprehensive quality scores (0-100)
    2. Track quality over time
    3. Identify degrading features
    4. Recommend feature enablement/disablement
    5. Generate quality reports
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize Feature Quality Scorer.
        
        Args:
            db_session: SQLAlchemy database session
        """
        self.db_session = db_session
    
    def calculate_comprehensive_quality_score(
        self,
        feature_name: str,
        performance_metrics: Dict[str, float],
        importance_score: Optional[float] = None,
        uniqueness_score: Optional[float] = None,
        stability_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive quality score for a feature.
        
        Quality score is based on:
        - Predictive Power (40%): Win rate, Sharpe ratio, returns
        - Importance (25%): ML model feature importance
        - Uniqueness (20%): Low correlation with other features
        - Stability (15%): Consistent performance over time
        
        Args:
            feature_name: Name of the feature
            performance_metrics: Performance metrics from FeatureAnalyzer
            importance_score: Optional ML importance score (0-1)
            uniqueness_score: Optional uniqueness score (0-1)
            stability_score: Optional stability score (0-1)
            
        Returns:
            Dictionary with quality breakdown
        """
        logger.info(f"Calculating comprehensive quality score for {feature_name}")
        
        # Predictive Power Score (40%)
        predictive_score = self._calculate_predictive_score(performance_metrics)
        
        # Importance Score (25%)
        if importance_score is not None:
            importance_score_normalized = importance_score * 100
        else:
            importance_score_normalized = 50.0  # Default if not available
        
        # Uniqueness Score (20%)
        if uniqueness_score is not None:
            uniqueness_score_normalized = uniqueness_score * 100
        else:
            uniqueness_score_normalized = 50.0  # Default if not available
        
        # Stability Score (15%)
        if stability_score is not None:
            stability_score_normalized = stability_score * 100
        else:
            stability_score_normalized = 50.0  # Default if not available
        
        # Calculate weighted score
        total_score = (
            predictive_score * 0.4 +
            importance_score_normalized * 0.25 +
            uniqueness_score_normalized * 0.2 +
            stability_score_normalized * 0.15
        )
        
        return {
            'feature_name': feature_name,
            'total_quality_score': round(total_score, 2),
            'predictive_score': round(predictive_score, 2),
            'importance_score': round(importance_score_normalized, 2),
            'uniqueness_score': round(uniqueness_score_normalized, 2),
            'stability_score': round(stability_score_normalized, 2),
            'grade': self._get_quality_grade(total_score),
        }
    
    def _calculate_predictive_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate predictive power score from performance metrics.
        
        Args:
            metrics: Performance metrics dictionary
            
        Returns:
            Predictive score (0-100)
        """
        win_rate = metrics.get('win_rate', 0.5)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        profit_factor = metrics.get('profit_factor', 1)
        average_return = metrics.get('average_return', 0)
        
        # Normalize each component
        win_rate_score = min(win_rate * 100, 100)
        sharpe_score = min(max(sharpe_ratio * 10, 0), 100)
        profit_factor_score = min(max((profit_factor - 1) * 20, 0), 100)
        return_score = min(max(average_return * 100, 0), 100)
        
        # Weighted average
        predictive_score = (
            win_rate_score * 0.4 +
            sharpe_score * 0.3 +
            profit_factor_score * 0.2 +
            return_score * 0.1
        )
        
        return predictive_score
    
    def _get_quality_grade(self, score: float) -> str:
        """Get letter grade for quality score."""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def calculate_stability_score(
        self,
        feature_name: str,
        lookback_periods: int = 4
    ) -> float:
        """
        Calculate stability score based on historical quality scores.
        
        Args:
            feature_name: Name of the feature
            lookback_periods: Number of historical periods to consider
            
        Returns:
            Stability score (0-100)
        """
        query = self.db_session.query(FeatureQuality).filter(
            FeatureQuality.feature_name == feature_name
        ).order_by(FeatureQuality.evaluation_period_end.desc()).limit(lookback_periods)
        
        records = query.all()
        
        if len(records) < 2:
            return 50.0  # Default if insufficient history
        
        scores = [record.quality_score for record in records]
        
        # Calculate standard deviation (lower is more stable)
        std_dev = np.std(scores)
        
        # Convert to stability score (inverse of std dev)
        stability_score = max(100 - (std_dev * 2), 0)
        
        return stability_score
    
    def calculate_uniqueness_score(
        self,
        feature_name: str,
        correlation_threshold: float = 0.7
    ) -> float:
        """
        Calculate uniqueness score based on correlation with other features.
        
        Args:
            feature_name: Name of the feature
            correlation_threshold: Correlation threshold for uniqueness
            
        Returns:
            Uniqueness score (0-100)
        """
        from database.models import FeatureCorrelation
        
        # Get correlations with this feature
        query = self.db_session.query(FeatureCorrelation).filter(
            and_(
                FeatureCorrelation.feature_1 == feature_name,
                func.abs(FeatureCorrelation.correlation_coefficient) >= correlation_threshold
            )
        )
        
        high_corr_count = query.count()
        
        # More high correlations = less unique
        if high_corr_count == 0:
            return 100.0
        elif high_corr_count <= 2:
            return 80.0
        elif high_corr_count <= 5:
            return 60.0
        elif high_corr_count <= 10:
            return 40.0
        else:
            return 20.0
    
    def update_feature_quality(
        self,
        feature_name: str,
        performance_metrics: Dict[str, float],
        importance_score: Optional[float] = None,
        evaluation_period_start: date = None,
        evaluation_period_end: date = None
    ) -> None:
        """
        Update feature quality in database.
        
        Args:
            feature_name: Name of the feature
            performance_metrics: Performance metrics
            importance_score: Optional importance score
            evaluation_period_start: Start date of evaluation
            evaluation_period_end: End date of evaluation
        """
        if evaluation_period_end is None:
            evaluation_period_end = date.today()
        if evaluation_period_start is None:
            evaluation_period_start = evaluation_period_end - timedelta(days=90)
        
        # Calculate component scores
        stability_score = self.calculate_stability_score(feature_name)
        uniqueness_score = self.calculate_uniqueness_score(feature_name)
        
        # Calculate comprehensive score
        quality_breakdown = self.calculate_comprehensive_quality_score(
            feature_name,
            performance_metrics,
            importance_score,
            uniqueness_score,
            stability_score
        )
        
        # Check if record exists
        existing = self.db_session.query(FeatureQuality).filter(
            and_(
                FeatureQuality.feature_name == feature_name,
                FeatureQuality.evaluation_period_end == evaluation_period_end
            )
        ).first()
        
        if existing:
            existing.quality_score = quality_breakdown['total_quality_score']
            existing.win_rate = performance_metrics.get('win_rate')
            existing.average_return = performance_metrics.get('average_return')
            existing.sharpe_ratio = performance_metrics.get('sharpe_ratio')
            existing.sortino_ratio = performance_metrics.get('sortino_ratio')
            existing.max_drawdown = performance_metrics.get('max_drawdown')
            existing.profit_factor = performance_metrics.get('profit_factor')
            existing.sample_size = performance_metrics.get('sample_size')
            existing.computed_at = datetime.now()
        else:
            quality_record = FeatureQuality(
                feature_name=feature_name,
                feature_version='1.0',
                quality_score=quality_breakdown['total_quality_score'],
                win_rate=performance_metrics.get('win_rate'),
                average_return=performance_metrics.get('average_return'),
                sharpe_ratio=performance_metrics.get('sharpe_ratio'),
                sortino_ratio=performance_metrics.get('sortino_ratio'),
                max_drawdown=performance_metrics.get('max_drawdown'),
                profit_factor=performance_metrics.get('profit_factor'),
                sample_size=performance_metrics.get('sample_size'),
                evaluation_period_start=evaluation_period_start,
                evaluation_period_end=evaluation_period_end,
                computed_at=datetime.now()
            )
            self.db_session.add(quality_record)
        
        self.db_session.commit()
        logger.info(
            f"Updated quality for {feature_name}: "
            f"{quality_breakdown['total_quality_score']} ({quality_breakdown['grade']})"
        )
    
    def get_quality_report(
        self,
        category: Optional[str] = None,
        min_score: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Generate quality report for features.
        
        Args:
            category: Optional category filter
            min_score: Optional minimum quality score
            
        Returns:
            DataFrame with quality report
        """
        query = self.db_session.query(FeatureQuality)
        
        if category:
            query = query.join(DBFeatureMetadata).filter(
                DBFeatureMetadata.feature_category == category
            )
        
        if min_score:
            query = query.filter(FeatureQuality.quality_score >= min_score)
        
        query = query.order_by(FeatureQuality.quality_score.desc())
        
        records = query.all()
        
        if not records:
            return pd.DataFrame()
        
        data = []
        for record in records:
            data.append({
                'feature_name': record.feature_name,
                'quality_score': record.quality_score,
                'grade': self._get_quality_grade(record.quality_score),
                'win_rate': record.win_rate,
                'sharpe_ratio': record.sharpe_ratio,
                'profit_factor': record.profit_factor,
                'sample_size': record.sample_size,
                'evaluation_period_end': record.evaluation_period_end,
            })
        
        return pd.DataFrame(data)
    
    def recommend_feature_actions(self) -> Dict[str, List[str]]:
        """
        Recommend actions for features based on quality scores.
        
        Returns:
            Dictionary with action recommendations
        """
        report = self.get_quality_report()
        
        if report.empty:
            return {
                'disable': [],
                'monitor': [],
                'keep': [],
            }
        
        disable = []
        monitor = []
        keep = []
        
        for _, row in report.iterrows():
            score = row['quality_score']
            
            if score < 50:
                disable.append(row['feature_name'])
            elif score < 70:
                monitor.append(row['feature_name'])
            else:
                keep.append(row['feature_name'])
        
        return {
            'disable': disable,
            'monitor': monitor,
            'keep': keep,
        }
