"""
Metrics Logger

Logs both training metrics and trading metrics for experiments.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from research_platform.experiments.base import (
    TrainingMetrics,
    TradingMetrics,
)
from utils.logger import get_logger

logger = get_logger("experiments.metrics_logger")


class MetricsLogger:
    """
    Metrics Logger.
    
    Logs:
    - Training metrics (accuracy, precision, recall, F1, ROC, LogLoss)
    - Trading metrics (win rate, returns, Sharpe, Sortino, drawdown, etc.)
    """
    
    def __init__(self):
        """Initialize metrics logger."""
        self.training_metrics: Dict[str, TrainingMetrics] = {}
        self.trading_metrics: Dict[str, TradingMetrics] = {}
        self._logger = get_logger("experiments.metrics_logger")
    
    def log_training_metrics(
        self,
        experiment_id: str,
        accuracy: float,
        precision: float,
        recall: float,
        f1_score: float,
        roc_auc: float,
        log_loss: float,
    ) -> TrainingMetrics:
        """
        Log training metrics.
        
        Args:
            experiment_id: Experiment ID
            accuracy: Accuracy score
            precision: Precision score
            recall: Recall score
            f1_score: F1 score
            roc_auc: ROC AUC score
            log_loss: Log loss
            
        Returns:
            TrainingMetrics object
        """
        metrics_id = f"TM-{uuid.uuid4().hex[:8].upper()}"
        
        metrics = TrainingMetrics(
            metrics_id=metrics_id,
            experiment_id=experiment_id,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            roc_auc=roc_auc,
            log_loss=log_loss,
        )
        
        self.training_metrics[metrics_id] = metrics
        
        self._logger.info(
            f"Logged training metrics {metrics_id}: "
            f"accuracy={accuracy:.4f}, f1={f1_score:.4f}"
        )
        
        return metrics
    
    def log_trading_metrics(
        self,
        experiment_id: str,
        win_rate: float,
        average_return: float,
        average_loss: float,
        profit_factor: float,
        sharpe_ratio: float,
        sortino_ratio: float,
        max_drawdown: float,
        expectancy: float,
        calmar_ratio: float,
        total_trades: int,
    ) -> TradingMetrics:
        """
        Log trading metrics.
        
        Args:
            experiment_id: Experiment ID
            win_rate: Win rate
            average_return: Average return
            average_loss: Average loss
            profit_factor: Profit factor
            sharpe_ratio: Sharpe ratio
            sortino_ratio: Sortino ratio
            max_drawdown: Maximum drawdown
            expectancy: Expectancy
            calmar_ratio: Calmar ratio
            total_trades: Total trades
            
        Returns:
            TradingMetrics object
        """
        metrics_id = f"TRM-{uuid.uuid4().hex[:8].upper()}"
        
        metrics = TradingMetrics(
            metrics_id=metrics_id,
            experiment_id=experiment_id,
            win_rate=win_rate,
            average_return=average_return,
            average_loss=average_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            expectancy=expectancy,
            calmar_ratio=calmar_ratio,
            total_trades=total_trades,
        )
        
        self.trading_metrics[metrics_id] = metrics
        
        self._logger.info(
            f"Logged trading metrics {metrics_id}: "
            f"win_rate={win_rate:.2%}, sharpe={sharpe_ratio:.2f}"
        )
        
        return metrics
    
    def calculate_trading_metrics_from_returns(
        self,
        experiment_id: str,
        returns: List[float],
        risk_free_rate: float = 0.0,
    ) -> TradingMetrics:
        """
        Calculate trading metrics from a list of returns.
        
        Args:
            experiment_id: Experiment ID
            returns: List of returns (in percentage)
            risk_free_rate: Risk-free rate for Sharpe calculation
            
        Returns:
            TradingMetrics object
        """
        if not returns:
            raise ValueError("Returns list cannot be empty")
        
        returns_array = np.array(returns)
        
        # Basic metrics
        total_trades = len(returns)
        winning_trades = sum(1 for r in returns if r > 0)
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        winning_returns = returns_array[returns_array > 0]
        losing_returns = returns_array[returns_array < 0]
        
        average_return = np.mean(returns_array)
        average_win = np.mean(winning_returns) if len(winning_returns) > 0 else 0.0
        average_loss = np.mean(losing_returns) if len(losing_returns) > 0 else 0.0
        
        # Profit factor
        total_wins = np.sum(winning_returns)
        total_losses = abs(np.sum(losing_returns))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Sharpe ratio
        if len(returns) > 1:
            excess_returns = returns_array - risk_free_rate
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Sortino ratio
        if len(losing_returns) > 0:
            downside_returns = returns_array[returns_array < 0]
            downside_deviation = np.std(downside_returns)
            sortino_ratio = np.mean(returns_array) / downside_deviation if downside_deviation > 0 else 0.0
        else:
            sortino_ratio = 0.0
        
        # Maximum drawdown
        cumulative_returns = np.cumsum(returns_array / 100)  # Convert to decimal
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / peak
        max_drawdown = np.max(drawdown) * 100 if len(drawdown) > 0 else 0.0
        
        # Expectancy
        expectancy = average_return
        
        # Calmar ratio
        calmar_ratio = average_return / max_drawdown if max_drawdown > 0 else 0.0
        
        return self.log_trading_metrics(
            experiment_id=experiment_id,
            win_rate=win_rate,
            average_return=average_return,
            average_loss=average_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            expectancy=expectancy,
            calmar_ratio=calmar_ratio,
            total_trades=total_trades,
        )
    
    def get_training_metrics(self, metrics_id: str) -> Optional[TrainingMetrics]:
        """Get training metrics by ID."""
        return self.training_metrics.get(metrics_id)
    
    def get_trading_metrics(self, metrics_id: str) -> Optional[TradingMetrics]:
        """Get trading metrics by ID."""
        return self.trading_metrics.get(metrics_id)
    
    def get_training_metrics_by_experiment(self, experiment_id: str) -> List[TrainingMetrics]:
        """Get all training metrics for an experiment."""
        return [
            metrics for metrics in self.training_metrics.values()
            if metrics.experiment_id == experiment_id
        ]
    
    def get_trading_metrics_by_experiment(self, experiment_id: str) -> List[TradingMetrics]:
        """Get all trading metrics for an experiment."""
        return [
            metrics for metrics in self.trading_metrics.values()
            if metrics.experiment_id == experiment_id
        ]
    
    def compare_trading_metrics(
        self,
        metrics_id_1: str,
        metrics_id_2: str,
    ) -> Optional[Dict]:
        """
        Compare two trading metrics.
        
        Args:
            metrics_id_1: First metrics ID
            metrics_id_2: Second metrics ID
            
        Returns:
            Dictionary with comparison results
        """
        metrics_1 = self.trading_metrics.get(metrics_id_1)
        metrics_2 = self.trading_metrics.get(metrics_id_2)
        
        if not metrics_1 or not metrics_2:
            return None
        
        comparison = {
            'metrics_1': metrics_1.to_dict(),
            'metrics_2': metrics_2.to_dict(),
            'differences': {
                'win_rate_change': metrics_2.win_rate - metrics_1.win_rate,
                'sharpe_change': metrics_2.sharpe_ratio - metrics_1.sharpe_ratio,
                'drawdown_change': metrics_2.max_drawdown - metrics_1.max_drawdown,
                'return_change': metrics_2.average_return - metrics_1.average_return,
                'trades_change': metrics_2.total_trades - metrics_1.total_trades,
            },
        }
        
        return comparison
    
    def get_best_trading_metrics(
        self,
        metric: str = "sharpe_ratio",
        top_n: int = 10,
    ) -> List[TradingMetrics]:
        """
        Get best trading metrics by a specific metric.
        
        Args:
            metric: Metric to compare (sharpe_ratio, win_rate, average_return, etc.)
            top_n: Number of top results to return
            
        Returns:
            List of top TradingMetrics objects
        """
        all_metrics = list(self.trading_metrics.values())
        
        # Sort by metric descending
        all_metrics.sort(key=lambda x: getattr(x, metric, 0), reverse=True)
        
        return all_metrics[:top_n]
