"""
Quality Calculator

Calculates quality scores for evidence based on historical performance.
Combines multiple quality dimensions into a single score.
"""

from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from meta_alpha.evidence_weighting.quality_score import QualityScore, QualityScoreBuilder
from utils.logger import get_logger

logger = get_logger("meta_alpha.evidence_weighting")


class QualityCalculator:
    """
    Calculates quality scores for evidence based on historical performance.
    
    Dimensions:
    - Data quality: Completeness, accuracy, timeliness
    - Historical IC: Information Coefficient over time
    - Historical Sharpe: Risk-adjusted returns
    - Regime stability: Performance across market regimes
    - Missing values: Penalty for incomplete data
    """
    
    def __init__(
        self,
        ic_threshold: float = 0.05,
        sharpe_threshold: float = 1.0,
        missing_penalty: float = 20.0,
    ):
        """
        Initialize quality calculator.
        
        Args:
            ic_threshold: Minimum IC for good quality
            sharpe_threshold: Minimum Sharpe for good quality
            missing_penalty: Penalty for missing data
        """
        self.ic_threshold = ic_threshold
        self.sharpe_threshold = sharpe_threshold
        self.missing_penalty = missing_penalty
        self._logger = get_logger("meta_alpha.evidence_weighting")
    
    def calculate_quality(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        market_contexts: Optional[pd.DataFrame] = None,
    ) -> QualityScore:
        """
        Calculate quality score for evidence.
        
        Args:
            factor_values: Series of factor values
            returns: Series of forward returns
            market_contexts: Optional DataFrame with market context
            
        Returns:
            QualityScore
        """
        # Calculate data quality
        data_quality = self._calculate_data_quality(factor_values)
        
        # Calculate historical IC
        historical_ic = self._calculate_historical_ic(factor_values, returns)
        
        # Calculate historical Sharpe
        historical_sharpe = self._calculate_historical_sharpe(returns)
        
        # Calculate regime stability
        regime_stability = self._calculate_regime_stability(
            factor_values,
            returns,
            market_contexts,
        )
        
        # Calculate missing values penalty
        missing_values = self._calculate_missing_values_penalty(factor_values)
        
        # Build quality score
        builder = QualityScoreBuilder()
        quality_score = (
            builder.data_quality(data_quality)
            .historical_ic(historical_ic)
            .historical_sharpe(historical_sharpe)
            .regime_stability(regime_stability)
            .missing_values(missing_values)
            .build()
        )
        
        return quality_score
    
    def _calculate_data_quality(self, factor_values: pd.Series) -> float:
        """
        Calculate data quality score.
        
        Args:
            factor_values: Series of factor values
            
        Returns:
            Data quality score (0-100)
        """
        if len(factor_values) == 0:
            return 0.0
        
        # Check for missing values
        missing_ratio = factor_values.isna().sum() / len(factor_values)
        
        # Check for infinite values
        infinite_ratio = np.isinf(factor_values).sum() / len(factor_values)
        
        # Calculate base score
        base_score = 100.0
        
        # Penalize missing values
        base_score -= missing_ratio * 100.0
        
        # Penalize infinite values
        base_score -= infinite_ratio * 100.0
        
        # Check for reasonable range
        if not factor_values.empty:
            finite_values = factor_values[np.isfinite(factor_values)]
            if len(finite_values) > 0:
                q1 = finite_values.quantile(0.25)
                q3 = finite_values.quantile(0.75)
                iqr = q3 - q1
                
                # Check for outliers
                outliers = ((finite_values < (q1 - 3 * iqr)) | (finite_values > (q3 + 3 * iqr))).sum()
                outlier_ratio = outliers / len(finite_values)
                base_score -= outlier_ratio * 50.0
        
        return max(0.0, min(100.0, base_score))
    
    def _calculate_historical_ic(self, factor_values: pd.Series, returns: pd.Series) -> float:
        """
        Calculate historical IC score.
        
        Args:
            factor_values: Series of factor values
            returns: Series of forward returns
            
        Returns:
            Historical IC score (0-100)
        """
        if len(factor_values) < 10 or len(returns) < 10:
            return 50.0  # Neutral score for insufficient data
        
        # Align data
        aligned_factor, aligned_returns = factor_values.align(returns, join='inner')
        
        if len(aligned_factor) < 10:
            return 50.0
        
        # Calculate IC
        ic = aligned_factor.corr(aligned_returns)
        
        if pd.isna(ic):
            return 50.0
        
        # Convert to score (0-100)
        # IC of 0.1 is good, IC of 0.05 is threshold
        if ic >= self.ic_threshold * 2:
            return 100.0
        elif ic >= self.ic_threshold:
            return 50.0 + (ic - self.ic_threshold) / self.ic_threshold * 50.0
        elif ic >= 0:
            return 50.0 * (ic / self.ic_threshold)
        else:
            return max(0.0, 50.0 - abs(ic) / self.ic_threshold * 50.0)
    
    def _calculate_historical_sharpe(self, returns: pd.Series) -> float:
        """
        Calculate historical Sharpe score.
        
        Args:
            returns: Series of forward returns
            
        Returns:
            Historical Sharpe score (0-100)
        """
        if len(returns) < 10:
            return 50.0
        
        # Calculate Sharpe
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0
        
        if pd.isna(sharpe):
            return 50.0
        
        # Convert to score (0-100)
        # Sharpe of 2.0 is excellent, Sharpe of 1.0 is threshold
        if sharpe >= self.sharpe_threshold * 2:
            return 100.0
        elif sharpe >= self.sharpe_threshold:
            return 50.0 + (sharpe - self.sharpe_threshold) / self.sharpe_threshold * 50.0
        elif sharpe >= 0:
            return 50.0 * (sharpe / self.sharpe_threshold)
        else:
            return max(0.0, 50.0 - abs(sharpe) / self.sharpe_threshold * 50.0)
    
    def _calculate_regime_stability(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        market_contexts: Optional[pd.DataFrame],
    ) -> float:
        """
        Calculate regime stability score.
        
        Args:
            factor_values: Series of factor values
            returns: Series of forward returns
            market_contexts: Optional DataFrame with market context
            
        Returns:
            Regime stability score (0-100)
        """
        if market_contexts is None or len(market_contexts) == 0:
            return 50.0  # Neutral score if no context
        
        # Align data
        aligned_factor, aligned_returns = factor_values.align(returns, join='inner')
        aligned_context = market_contexts.loc[aligned_factor.index]
        
        if len(aligned_factor) < 10 or len(aligned_context) < 10:
            return 50.0
        
        # Group by trend if available
        if "trend" in aligned_context.columns:
            trend_groups = aligned_context.groupby("trend")
            
            regime_scores = []
            
            for trend, group in trend_groups:
                if len(group) < 5:
                    continue
                
                # Get corresponding factor and returns
                group_factor = aligned_factor[group.index]
                group_returns = aligned_returns[group.index]
                
                # Calculate IC for this regime
                ic = group_factor.corr(group_returns)
                
                if not pd.isna(ic):
                    regime_scores.append(abs(ic))
            
            if regime_scores:
                # Stability is inverse of variance
                stability = 1.0 - np.std(regime_scores) if regime_scores else 0.0
                return max(0.0, min(100.0, stability * 100.0))
        
        return 50.0
    
    def _calculate_missing_values_penalty(self, factor_values: pd.Series) -> float:
        """
        Calculate missing values penalty score.
        
        Args:
            factor_values: Series of factor values
            
        Returns:
            Missing values penalty score (0-100)
        """
        if len(factor_values) == 0:
            return 0.0
        
        missing_ratio = factor_values.isna().sum() / len(factor_values)
        
        # Penalty: 100 - missing_ratio * 100
        penalty = 100.0 - missing_ratio * 100.0
        
        return max(0.0, min(100.0, penalty))
    
    def calculate_batch_quality(
        self,
        factor_data: Dict[str, pd.Series],
        returns: pd.Series,
        market_contexts: Optional[pd.DataFrame] = None,
    ) -> Dict[str, QualityScore]:
        """
        Calculate quality scores for multiple factors.
        
        Args:
            factor_data: Dictionary mapping factor names to factor values
            returns: Series of forward returns
            market_contexts: Optional DataFrame with market context
            
        Returns:
            Dictionary mapping factor names to QualityScore
        """
        quality_scores = {}
        
        for factor_name, factor_values in factor_data.items():
            try:
                quality_score = self.calculate_quality(
                    factor_values,
                    returns,
                    market_contexts,
                )
                quality_scores[factor_name] = quality_score
            except Exception as e:
                self._logger.error(f"Failed to calculate quality for {factor_name}: {e}")
                # Return neutral score on error
                quality_scores[factor_name] = QualityScore(
                    data_quality=50.0,
                    historical_ic=50.0,
                    historical_sharpe=50.0,
                    regime_stability=50.0,
                    missing_values=100.0,
                    overall_score=50.0,
                )
        
        return quality_scores


def calculate_evidence_quality(
    factor_values: pd.Series,
    returns: pd.Series,
    market_contexts: Optional[pd.DataFrame] = None,
) -> QualityScore:
    """
    Convenience function to calculate evidence quality.
    
    Args:
        factor_values: Series of factor values
        returns: Series of forward returns
        market_contexts: Optional DataFrame with market context
        
    Returns:
        QualityScore
    """
    calculator = QualityCalculator()
    return calculator.calculate_quality(factor_values, returns, market_contexts)
