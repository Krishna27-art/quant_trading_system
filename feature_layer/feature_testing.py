"""
Feature Testing Framework

Tests individual features and feature combinations to identify alpha candidates.
Implements backtesting for feature-based strategies.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import and_
import uuid

from database.models import FeatureCombination, Feature
from feature_layer.feature_analyzer import FeatureAnalyzer


logger = logging.getLogger(__name__)


class FeatureTester:
    """
    Feature Testing Framework.
    
    Responsibilities:
    1. Test single features with thresholds
    2. Test feature combinations
    3. Calculate backtest metrics
    4. Save alpha candidates
    5. Compare strategies
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize Feature Tester.
        
        Args:
            db_session: SQLAlchemy database session
        """
        self.db_session = db_session
        self.analyzer = FeatureAnalyzer(db_session)
    
    def test_single_feature(
        self,
        feature_name: str,
        condition: str,
        start_date: date,
        end_date: date,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Test a single feature with a specific condition.
        
        Args:
            feature_name: Name of the feature to test
            condition: Condition string (e.g., "RSI14 > 70", "VWAP_Distance < -2")
            start_date: Start date for backtest
            end_date: End date for backtest
            symbols: Optional list of symbols to test
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Testing feature {feature_name} with condition: {condition}")
        
        # Get feature data
        feature_data = self._get_feature_data(
            feature_name,
            start_date,
            end_date,
            symbols
        )
        
        if feature_data.empty:
            return {
                "feature_name": feature_name,
                "condition": condition,
                "error": "No feature data available"
            }
        
        # Apply condition to filter signals
        signals = self._apply_condition(feature_data, condition)
        
        if signals.empty:
            return {
                "feature_name": feature_name,
                "condition": condition,
                "error": "No signals generated"
            }
        
        # Calculate returns (simplified - in production would use actual price data)
        results = self._calculate_backtest_metrics(signals, start_date, end_date)
        
        return {
            "feature_name": feature_name,
            "condition": condition,
            "total_signals": len(signals),
            **results
        }
    
    def test_feature_combination(
        self,
        features: List[str],
        conditions: List[str],
        combination_name: str,
        start_date: date,
        end_date: date,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Test a combination of features with multiple conditions.
        
        Args:
            features: List of feature names
            conditions: List of condition strings (one per feature)
            combination_name: Name for this combination
            start_date: Start date for backtest
            end_date: End date for backtest
            symbols: Optional list of symbols to test
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(
            f"Testing combination {combination_name} with {len(features)} features"
        )
        
        if len(features) != len(conditions):
            raise ValueError("Number of features must match number of conditions")
        
        # Get data for all features
        all_feature_data = {}
        for feature_name in features:
            feature_data = self._get_feature_data(
                feature_name,
                start_date,
                end_date,
                symbols
            )
            if not feature_data.empty:
                all_feature_data[feature_name] = feature_data
        
        if not all_feature_data:
            return {
                "combination_name": combination_name,
                "error": "No feature data available"
            }
        
        # Merge all feature data
        merged_data = self._merge_feature_data(all_feature_data)
        
        if merged_data.empty:
            return {
                "combination_name": combination_name,
                "error": "Failed to merge feature data"
            }
        
        # Apply all conditions (AND logic)
        signals = merged_data.copy()
        for feature_name, condition in zip(features, conditions):
            signals = self._apply_condition(signals, condition, feature_name)
        
        if signals.empty:
            return {
                "combination_name": combination_name,
                "error": "No signals generated after applying conditions"
            }
        
        # Calculate returns
        results = self._calculate_backtest_metrics(signals, start_date, end_date)
        
        return {
            "combination_name": combination_name,
            "features": features,
            "conditions": conditions,
            "total_signals": len(signals),
            **results
        }
    
    def _get_feature_data(
        self,
        feature_name: str,
        start_date: date,
        end_date: date,
        symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Get feature data from database."""
        query = self.db_session.query(Feature).filter(
            and_(
                Feature.feature_name == feature_name,
                Feature.date >= start_date,
                Feature.date <= end_date
            )
        )
        
        if symbols:
            query = query.filter(Feature.symbol.in_(symbols))
        
        records = query.all()
        
        if not records:
            return pd.DataFrame()
        
        data = []
        for record in records:
            data.append({
                'symbol': record.symbol,
                'date': record.date,
                'feature_value': record.feature_value,
            })
        
        return pd.DataFrame(data)
    
    def _apply_condition(
        self,
        data: pd.DataFrame,
        condition: str,
        feature_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Apply condition to filter data.
        
        Args:
            data: DataFrame with feature data
            condition: Condition string (e.g., "> 70", "< -2")
            feature_col: Optional feature column name (defaults to 'feature_value')
            
        Returns:
            Filtered DataFrame
        """
        if feature_col is None:
            feature_col = 'feature_value'
        
        # Parse condition (simplified - in production use proper expression parser)
        try:
            # Extract operator and value
            if '>' in condition:
                parts = condition.split('>')
                operator = '>'
                value = float(parts[1].strip())
            elif '<' in condition:
                parts = condition.split('<')
                operator = '<'
                value = float(parts[1].strip())
            elif '>=' in condition:
                parts = condition.split('>=')
                operator = '>='
                value = float(parts[1].strip())
            elif '<=' in condition:
                parts = condition.split('<=')
                operator = '<='
                value = float(parts[1].strip())
            else:
                raise ValueError(f"Unsupported condition: {condition}")
            
            # Apply filter
            if operator == '>':
                filtered = data[data[feature_col] > value]
            elif operator == '<':
                filtered = data[data[feature_col] < value]
            elif operator == '>=':
                filtered = data[data[feature_col] >= value]
            elif operator == '<=':
                filtered = data[data[feature_col] <= value]
            else:
                filtered = data
            
            return filtered
            
        except Exception as e:
            logger.error(f"Failed to apply condition {condition}: {e}")
            return pd.DataFrame()
    
    def _merge_feature_data(self, feature_data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple feature DataFrames."""
        if not feature_data_dict:
            return pd.DataFrame()
        
        # Start with first feature
        merged = list(feature_data_dict.values())[0].copy()
        merged = merged.rename(columns={'feature_value': f'feature_0'})
        
        # Merge remaining features
        for i, (feature_name, data) in enumerate(
            list(feature_data_dict.items())[1:], start=1
        ):
            data = data.rename(columns={'feature_value': f'feature_{i}'})
            merged = pd.merge(
                merged,
                data,
                on=['symbol', 'date'],
                how='inner'
            )
        
        return merged
    
    def _calculate_backtest_metrics(
        self,
        signals: pd.DataFrame,
        start_date: date,
        end_date: date
    ) -> Dict[str, Any]:
        """
        Calculate backtest metrics for signals.
        
        Note: This is a simplified implementation.
        In production, this would:
        1. Get actual price data for the symbols
        2. Calculate forward returns
        3. Compute Sharpe ratio, max drawdown, etc.
        
        Args:
            signals: DataFrame with signals
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with backtest metrics
        """
        # Placeholder implementation
        # In production, this would calculate actual returns
        
        return {
            "win_rate": 0.0,
            "average_return": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "total_trades": len(signals),
            "winning_trades": 0,
            "losing_trades": 0,
        }
    
    def save_alpha_candidate(
        self,
        combination_name: str,
        features: List[str],
        conditions: List[str],
        results: Dict[str, Any],
        notes: str = ""
    ) -> str:
        """
        Save a successful feature combination as an alpha candidate.
        
        Args:
            combination_name: Name of the combination
            features: List of feature names
            conditions: List of conditions
            results: Backtest results
            notes: Optional notes
            
        Returns:
            ID of the saved combination
        """
        import json
        
        combination_id = str(uuid.uuid4())
        
        combination = FeatureCombination(
            id=combination_id,
            combination_name=combination_name,
            features=json.dumps(features),
            conditions=json.dumps(conditions),
            win_rate=results.get('win_rate'),
            average_return=results.get('average_return'),
            sharpe_ratio=results.get('sharpe_ratio'),
            max_drawdown=results.get('max_drawdown'),
            sample_size=results.get('total_trades', 0),
            is_active=True,
            notes=notes,
            created_at=datetime.now(),
            last_tested=datetime.now()
        )
        
        self.db_session.add(combination)
        self.db_session.commit()
        
        logger.info(f"Saved alpha candidate: {combination_name} (ID: {combination_id})")
        
        return combination_id
    
    def get_alpha_candidates(
        self,
        min_win_rate: float = 0.6,
        min_sharpe: float = 1.0,
        active_only: bool = True
    ) -> pd.DataFrame:
        """
        Get alpha candidates that meet performance criteria.
        
        Args:
            min_win_rate: Minimum win rate
            min_sharpe: Minimum Sharpe ratio
            active_only: Whether to show only active candidates
            
        Returns:
            DataFrame with alpha candidates
        """
        query = self.db_session.query(FeatureCombination).filter(
            and_(
                FeatureCombination.win_rate >= min_win_rate,
                FeatureCombination.sharpe_ratio >= min_sharpe
            )
        )
        
        if active_only:
            query = query.filter(FeatureCombination.is_active == True)
        
        query = query.order_by(FeatureCombination.sharpe_ratio.desc())
        
        records = query.all()
        
        if not records:
            return pd.DataFrame()
        
        import json
        
        data = []
        for record in records:
            data.append({
                'id': record.id,
                'combination_name': record.combination_name,
                'features': json.loads(record.features),
                'conditions': json.loads(record.conditions),
                'win_rate': record.win_rate,
                'average_return': record.average_return,
                'sharpe_ratio': record.sharpe_ratio,
                'max_drawdown': record.max_drawdown,
                'sample_size': record.sample_size,
                'is_active': record.is_active,
                'last_tested': record.last_tested,
                'notes': record.notes,
            })
        
        return pd.DataFrame(data)
    
    def compare_strategies(
        self,
        combination_ids: List[str]
    ) -> pd.DataFrame:
        """
        Compare multiple alpha candidates.
        
        Args:
            combination_ids: List of combination IDs to compare
            
        Returns:
            DataFrame with comparison
        """
        query = self.db_session.query(FeatureCombination).filter(
            FeatureCombination.id.in_(combination_ids)
        )
        
        records = query.all()
        
        if not records:
            return pd.DataFrame()
        
        import json
        
        data = []
        for record in records:
            data.append({
                'combination_name': record.combination_name,
                'win_rate': record.win_rate,
                'average_return': record.average_return,
                'sharpe_ratio': record.sharpe_ratio,
                'sortino_ratio': record.sortino_ratio,
                'max_drawdown': record.max_drawdown,
                'profit_factor': record.profit_factor,
                'sample_size': record.sample_size,
            })
        
        df = pd.DataFrame(data)
        
        # Add rank column based on Sharpe ratio
        df['sharpe_rank'] = df['sharpe_ratio'].rank(ascending=False)
        
        return df.sort_values('sharpe_ratio', ascending=False)
