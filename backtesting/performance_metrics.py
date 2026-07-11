"""
Performance Metrics Module

Calculates comprehensive performance metrics for backtesting results.
Includes:
- Prediction Accuracy
- Win Rate
- Average Return/Loss
- Profit Factor
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Expectancy
- Average Holding Period
- Calmar Ratio
- Information Ratio
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("performance_metrics")


@dataclass
class PerformanceMetrics:
    """Container for all performance metrics."""
    
    # Basic metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # Return metrics
    total_return: float = 0.0
    avg_return: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    
    # Risk-adjusted metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    
    # Trade metrics
    avg_holding_period: float = 0.0
    avg_win_holding_period: float = 0.0
    avg_loss_holding_period: float = 0.0
    
    # Distribution metrics
    skewness: float = 0.0
    kurtosis: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "total_return": self.total_return,
            "avg_return": self.avg_return,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "best_trade": self.best_trade,
            "worst_trade": self.worst_trade,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "information_ratio": self.information_ratio,
            "avg_holding_period": self.avg_holding_period,
            "avg_win_holding_period": self.avg_win_holding_period,
            "avg_loss_holding_period": self.avg_loss_holding_period,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
        }


class PerformanceCalculator:
    """
    Calculates performance metrics from trade results.
    
    Uses institutional-grade metrics for comprehensive evaluation.
    """
    
    def __init__(self, risk_free_rate: float = 0.06):
        """
        Initialize the performance calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 6% for India)
        """
        self.risk_free_rate = risk_free_rate
        self.logger = logger
    
    def calculate_metrics(
        self,
        trades: list[dict[str, Any]],
        equity_curve: list[float] | None = None,
    ) -> PerformanceMetrics:
        """
        Calculate all performance metrics.
        
        Args:
            trades: List of trade dicts with pnl, pnl_pct, holding_days
            equity_curve: List of equity values over time (optional)
            
        Returns:
            PerformanceMetrics object
        """
        metrics = PerformanceMetrics()
        
        if not trades:
            self.logger.warning("No trades provided, returning empty metrics")
            return metrics
        
        # Filter completed trades
        completed_trades = [t for t in trades if t.get('pnl') is not None]
        
        if not completed_trades:
            self.logger.warning("No completed trades")
            return metrics
        
        pnls = [t['pnl'] for t in completed_trades]
        pnl_pcts = [t['pnl_pct'] for t in completed_trades if t.get('pnl_pct') is not None]
        holding_days = [t.get('holding_days', 0) for t in completed_trades if t.get('holding_days') is not None]
        
        # Basic metrics
        metrics.total_trades = len(completed_trades)
        metrics.winning_trades = sum(1 for pnl in pnls if pnl > 0)
        metrics.losing_trades = sum(1 for pnl in pnls if pnl < 0)
        metrics.win_rate = (metrics.winning_trades / metrics.total_trades * 100) if metrics.total_trades > 0 else 0
        
        # Return metrics
        metrics.total_return = sum(pnls)
        metrics.avg_return = np.mean(pnls) if pnls else 0
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl < 0]
        metrics.avg_win = np.mean(wins) if wins else 0
        metrics.avg_loss = np.mean(losses) if losses else 0
        metrics.best_trade = max(pnls) if pnls else 0
        metrics.worst_trade = min(pnls) if pnls else 0
        
        # Risk metrics
        metrics.max_drawdown, metrics.max_drawdown_duration = self._calculate_max_drawdown(
            equity_curve if equity_curve else pnls
        )
        
        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        metrics.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Expectancy
        metrics.expectancy = metrics.avg_return if metrics.total_trades > 0 else 0
        
        # Risk-adjusted metrics
        if pnl_pcts:
            returns_array = np.array(pnl_pcts)
            metrics.sharpe_ratio = self._calculate_sharpe_ratio(returns_array)
            metrics.sortino_ratio = self._calculate_sortino_ratio(returns_array)
            metrics.calmar_ratio = self._calculate_calmar_ratio(returns_array, metrics.max_drawdown)
            metrics.information_ratio = self._calculate_information_ratio(returns_array)
        
        # Distribution metrics
        if pnl_pcts:
            metrics.skewness = float(pd.Series(pnl_pcts).skew())
            metrics.kurtosis = float(pd.Series(pnl_pcts).kurtosis())
        
        # Trade metrics
        if holding_days:
            metrics.avg_holding_period = np.mean(holding_days)
            win_days = [holding_days[i] for i, pnl in enumerate(pnls) if pnl > 0]
            loss_days = [holding_days[i] for i, pnl in enumerate(pnls) if pnl < 0]
            metrics.avg_win_holding_period = np.mean(win_days) if win_days else 0
            metrics.avg_loss_holding_period = np.mean(loss_days) if loss_days else 0
        
        return metrics
    
    def _calculate_max_drawdown(
        self,
        equity_curve: list[float] | list,
    ) -> tuple[float, int]:
        """
        Calculate maximum drawdown and duration.
        
        Args:
            equity_curve: List of equity values or PnLs
            
        Returns:
            Tuple of (max_drawdown_pct, max_drawdown_duration)
        """
        if not equity_curve or len(equity_curve) < 2:
            return 0.0, 0
        
        # Convert to cumulative if PnLs
        if isinstance(equity_curve[0], (int, float)) and abs(equity_curve[0]) < 1:
            # Assume these are PnLs, convert to cumulative
            equity = np.cumsum(equity_curve)
        else:
            equity = np.array(equity_curve)
        
        peak = equity[0]
        max_dd = 0.0
        max_dd_duration = 0
        current_dd_duration = 0
        
        for i in range(1, len(equity)):
            if equity[i] > peak:
                peak = equity[i]
                current_dd_duration = 0
            else:
                current_dd_duration += 1
                dd = (peak - equity[i]) / peak * 100 if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
                    max_dd_duration = current_dd_duration
        
        return max_dd, max_dd_duration
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate Sharpe Ratio.
        
        Sharpe = (mean_return - risk_free_rate) / std_return
        """
        if len(returns) < 2:
            return 0.0
        
        # Annualize (assuming daily returns)
        mean_return = np.mean(returns) * 252  # Annualized
        std_return = np.std(returns) * np.sqrt(252)  # Annualized
        
        if std_return == 0:
            return 0.0
        
        sharpe = (mean_return - self.risk_free_rate) / std_return
        return sharpe
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate Sortino Ratio.
        
        Sortino = (mean_return - risk_free_rate) / downside_deviation
        """
        if len(returns) < 2:
            return 0.0
        
        # Annualize
        mean_return = np.mean(returns) * 252
        
        # Downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf') if mean_return > self.risk_free_rate else 0.0
        
        downside_std = np.std(downside_returns) * np.sqrt(252)
        
        if downside_std == 0:
            return 0.0
        
        sortino = (mean_return - self.risk_free_rate) / downside_std
        return sortino
    
    def _calculate_calmar_ratio(self, returns: np.ndarray, max_drawdown: float) -> float:
        """
        Calculate Calmar Ratio.
        
        Calmar = annual_return / max_drawdown
        """
        if max_drawdown == 0:
            return 0.0
        
        annual_return = np.mean(returns) * 252
        calmar = annual_return / abs(max_drawdown)
        return calmar
    
    def _calculate_information_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate Information Ratio.
        
        IR = (portfolio_return - benchmark_return) / tracking_error
        For simplicity, assume benchmark return is 0 (excess returns).
        """
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns) * 252
        tracking_error = np.std(returns) * np.sqrt(252)
        
        if tracking_error == 0:
            return 0.0
        
        ir = mean_return / tracking_error
        return ir
    
    def calculate_prediction_accuracy(
        self,
        predictions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Calculate prediction accuracy metrics.
        
        Args:
            predictions: List of prediction dicts with actual outcome
            
        Returns:
            Dict with accuracy metrics
        """
        if not predictions:
            return {"accuracy": 0.0, "total": 0, "correct": 0}
        
        correct = sum(1 for p in predictions if p.get('correct', False))
        total = len(predictions)
        accuracy = correct / total * 100 if total > 0 else 0
        
        # Calculate by confidence buckets
        confidence_buckets = {}
        for bucket in [50, 60, 70, 80, 90]:
            bucket_preds = [p for p in predictions if p.get('confidence', 0) >= bucket]
            if bucket_preds:
                bucket_correct = sum(1 for p in bucket_preds if p.get('correct', False))
                confidence_buckets[f"confidence_{bucket}+"] = {
                    "accuracy": bucket_correct / len(bucket_preds) * 100,
                    "count": len(bucket_preds),
                }
        
        return {
            "accuracy": accuracy,
            "total": total,
            "correct": correct,
            "incorrect": total - correct,
            "confidence_buckets": confidence_buckets,
        }


def calculate_performance(
    trades: list[dict[str, Any]],
    equity_curve: list[float] | None = None,
    risk_free_rate: float = 0.06,
) -> dict[str, Any]:
    """
    Convenience function to calculate performance metrics.
    
    Args:
        trades: List of trade dicts
        equity_curve: Optional equity curve
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Dict with all performance metrics
    """
    calculator = PerformanceCalculator(risk_free_rate=risk_free_rate)
    metrics = calculator.calculate_metrics(trades, equity_curve)
    
    return metrics.to_dict()
