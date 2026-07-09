"""
Performance Metrics Calculator

Comprehensive performance metrics for factor evaluation.
Calculates Sharpe, Win Rate, Precision, Recall, Lift, Stability, Turnover, Mutual Information.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import precision_score, recall_score

from utils.logger import get_logger

logger = get_logger("research.performance_metrics")


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a factor."""
    sharpe_ratio: float
    win_rate: float
    precision: float
    recall: float
    lift: float
    stability: float
    turnover: float
    mutual_information: float
    max_drawdown: float
    calmar_ratio: float
    sortino_ratio: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "win_rate": round(self.win_rate, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "lift": round(self.lift, 4),
            "stability": round(self.stability, 4),
            "turnover": round(self.turnover, 4),
            "mutual_information": round(self.mutual_information, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
        }


class PerformanceMetricsCalculator:
    """
    Calculates comprehensive performance metrics for factor evaluation.
    
    Metrics:
    - Sharpe Ratio: Risk-adjusted return
    - Win Rate: Percentage of winning trades
    - Precision: True positive rate for binary predictions
    - Recall: True positive rate for actual positives
    - Lift: Improvement over random baseline
    - Stability: Consistency of performance over time
    - Turnover: Rate of signal changes
    - Mutual Information: Non-linear dependence
    - Max Drawdown: Maximum peak-to-trough decline
    - Calmar Ratio: Return / Max Drawdown
    - Sortino Ratio: Downside risk-adjusted return
    """
    
    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize metrics calculator.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe calculation
        """
        self.risk_free_rate = risk_free_rate
        self._logger = get_logger("research.performance_metrics")
    
    def calculate(
        self,
        factor_values: pd.Series,
        future_returns: pd.Series,
        predictions: Optional[pd.Series] = None,
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            factor_values: Series with factor values
            future_returns: Series with future returns
            predictions: Optional binary predictions (0/1)
            
        Returns:
            PerformanceMetrics
        """
        # Remove NaN
        valid_mask = factor_values.notna() & future_returns.notna()
        factor_clean = factor_values[valid_mask]
        returns_clean = future_returns[valid_mask]
        
        if predictions is not None:
            predictions_clean = predictions[valid_mask]
        else:
            predictions_clean = None
        
        # Calculate metrics
        sharpe_ratio = self._calculate_sharpe(returns_clean)
        win_rate = self._calculate_win_rate(returns_clean)
        precision = self._calculate_precision(predictions_clean, returns_clean)
        recall = self._calculate_recall(predictions_clean, returns_clean)
        lift = self._calculate_lift(factor_clean, returns_clean)
        stability = self._calculate_stability(factor_clean, returns_clean)
        turnover = self._calculate_turnover(factor_clean)
        mutual_information = self._calculate_mutual_information(factor_clean, returns_clean)
        max_drawdown = self._calculate_max_drawdown(returns_clean)
        calmar_ratio = self._calculate_calmar(returns_clean, max_drawdown)
        sortino_ratio = self._calculate_sortino(returns_clean)
        
        return PerformanceMetrics(
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            precision=precision,
            recall=recall,
            lift=lift,
            stability=stability,
            turnover=turnover,
            mutual_information=mutual_information,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
        )
    
    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate
        sharpe = excess_returns.mean() / returns.std()
        
        # Annualize (assuming daily returns)
        sharpe_annualized = sharpe * np.sqrt(252)
        
        return float(sharpe_annualized)
    
    def _calculate_win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate (percentage of positive returns)."""
        if len(returns) == 0:
            return 0.0
        
        win_rate = (returns > 0).mean()
        return float(win_rate)
    
    def _calculate_precision(
        self,
        predictions: Optional[pd.Series],
        returns: pd.Series,
    ) -> float:
        """Calculate precision (true positive rate for predictions)."""
        if predictions is None:
            return 0.0
        
        # Convert returns to binary labels (1 if positive, 0 otherwise)
        labels = (returns > 0).astype(int)
        
        try:
            precision = precision_score(labels, predictions, zero_division=0)
            return float(precision)
        except Exception:
            return 0.0
    
    def _calculate_recall(
        self,
        predictions: Optional[pd.Series],
        returns: pd.Series,
    ) -> float:
        """Calculate recall (true positive rate for actual positives)."""
        if predictions is None:
            return 0.0
        
        # Convert returns to binary labels (1 if positive, 0 otherwise)
        labels = (returns > 0).astype(int)
        
        try:
            recall = recall_score(labels, predictions, zero_division=0)
            return float(recall)
        except Exception:
            return 0.0
    
    def _calculate_lift(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
    ) -> float:
        """
        Calculate lift (improvement over random baseline).
        
        Lift = Average return when factor is positive / Average return overall
        """
        if len(factor_values) == 0 or len(returns) == 0:
            return 0.0
        
        # Binary signal based on factor sign
        signal = (factor_values > 0).astype(int)
        
        # Average return when signal is positive
        positive_signal_returns = returns[signal == 1]
        if len(positive_signal_returns) == 0:
            return 0.0
        
        avg_positive_return = positive_signal_returns.mean()
        avg_overall_return = returns.mean()
        
        if avg_overall_return == 0:
            return 0.0
        
        lift = avg_positive_return / avg_overall_return
        return float(lift)
    
    def _calculate_stability(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        n_splits: int = 5,
    ) -> float:
        """
        Calculate stability (consistency of performance over time).
        
        Stability = 1 - (std of split Sharpe ratios / mean of split Sharpe ratios)
        """
        if len(factor_values) < n_splits * 10:
            return 0.0
        
        split_size = len(factor_values) // n_splits
        sharpe_ratios = []
        
        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = start_idx + split_size if i < n_splits - 1 else len(factor_values)
            
            split_returns = returns.iloc[start_idx:end_idx]
            split_sharpe = self._calculate_sharpe(split_returns)
            sharpe_ratios.append(split_sharpe)
        
        sharpe_array = np.array(sharpe_ratios)
        
        if np.mean(sharpe_array) == 0:
            return 0.0
        
        stability = 1 - (np.std(sharpe_array) / abs(np.mean(sharpe_array)))
        return max(0.0, float(stability))
    
    def _calculate_turnover(self, factor_values: pd.Series) -> float:
        """
        Calculate turnover (rate of signal changes).
        
        Turnover = Number of sign changes / Total observations
        """
        if len(factor_values) < 2:
            return 0.0
        
        # Binary signal based on factor sign
        signal = (factor_values > 0).astype(int)
        
        # Count sign changes
        sign_changes = (signal.diff() != 0).sum()
        
        turnover = sign_changes / len(signal)
        return float(turnover)
    
    def _calculate_mutual_information(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        bins: int = 10,
    ) -> float:
        """
        Calculate mutual information (non-linear dependence).
        
        Discretizes continuous variables and calculates mutual information.
        """
        if len(factor_values) == 0 or len(returns) == 0:
            return 0.0
        
        try:
            # Discretize variables
            factor_binned = pd.cut(factor_values, bins=bins, labels=False)
            returns_binned = pd.cut(returns, bins=bins, labels=False)
            
            # Remove NaN
            valid_mask = factor_binned.notna() & returns_binned.notna()
            factor_binned = factor_binned[valid_mask]
            returns_binned = returns_binned[valid_mask]
            
            # Calculate mutual information
            mi = stats.mutual_info_regression(
                factor_binned.values.reshape(-1, 1),
                returns_binned.values,
            )
            
            return float(mi[0])
        except Exception:
            return 0.0
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        
        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max
        
        # Maximum drawdown
        max_dd = drawdown.min()
        
        return float(max_dd)
    
    def _calculate_calmar(
        self,
        returns: pd.Series,
        max_drawdown: float,
    ) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        if len(returns) == 0 or max_drawdown == 0:
            return 0.0
        
        # Annualized return
        annualized_return = (1 + returns.mean()) ** 252 - 1
        
        calmar = annualized_return / abs(max_drawdown)
        return float(calmar)
    
    def _calculate_sortino(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (downside risk-adjusted return)."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate
        
        # Downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return 0.0
        
        downside_deviation = downside_returns.std()
        
        if downside_deviation == 0:
            return 0.0
        
        sortino = excess_returns.mean() / downside_deviation
        
        # Annualize
        sortino_annualized = sortino * np.sqrt(252)
        
        return float(sortino_annualized)


def calculate_performance_metrics(
    factor_values: pd.Series,
    future_returns: pd.Series,
    predictions: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """
    Convenience function to calculate performance metrics.
    
    Args:
        factor_values: Series with factor values
        future_returns: Series with future returns
        predictions: Optional binary predictions
        risk_free_rate: Risk-free rate for Sharpe calculation
        
    Returns:
        PerformanceMetrics
    """
    calculator = PerformanceMetricsCalculator(risk_free_rate=risk_free_rate)
    return calculator.calculate(factor_values, future_returns, predictions)
