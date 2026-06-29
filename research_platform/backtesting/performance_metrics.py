"""
Performance Metrics Calculation

Institutional-grade performance metrics for backtesting.
Calculates Sharpe, Sortino, Calmar, and other key metrics.
"""

from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from utils.logger import get_logger

logger = get_logger("performance_metrics")


class PerformanceMetrics(BaseModel):
    """Performance metrics for a backtest."""

    # Return metrics
    total_return: float = Field(..., description="Total return")
    annualized_return: float = Field(..., description="Annualized return")
    cagr: float = Field(..., description="Compound annual growth rate")

    # Risk metrics
    volatility: float = Field(..., description="Annualized volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    calmar_ratio: float = Field(..., description="Calmar ratio")

    # Drawdown metrics
    max_drawdown: float = Field(..., description="Maximum drawdown")
    avg_drawdown: float = Field(..., description="Average drawdown")
    max_drawdown_duration: int = Field(..., description="Maximum drawdown duration in days")

    # Trade metrics
    total_trades: int = Field(..., description="Total number of trades")
    win_rate: float = Field(..., description="Win rate")
    avg_win: float = Field(..., description="Average win")
    avg_loss: float = Field(..., description="Average loss")
    profit_factor: float = Field(..., description="Profit factor")

    # Portfolio metrics
    avg_positions: float = Field(..., description="Average number of positions")
    turnover: float = Field(..., description="Portfolio turnover")

    # Benchmark metrics
    alpha: float | None = Field(None, description="Alpha vs benchmark")
    beta: float | None = Field(None, description="Beta vs benchmark")
    information_ratio: float | None = Field(None, description="Information ratio")

    # Additional metrics
    skewness: float | None = Field(None, description="Return skewness")
    kurtosis: float | None = Field(None, description="Return kurtosis")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class PerformanceCalculator:
    """
    Performance calculator for backtesting results.

    Calculates institutional-grade performance metrics.
    """

    def __init__(self):
        """Initialize the performance calculator."""
        self.logger = logger

    def calculate_metrics(
        self,
        returns: list[float],
        equity_curve: list[tuple[datetime, float]],
        positions: list[Any],
        benchmark_returns: list[float] | None = None,
    ) -> PerformanceMetrics:
        """
        Calculate performance metrics.

        Args:
            returns: List of daily returns
            equity_curve: List of (date, value) tuples
            positions: List of positions
            benchmark_returns: Optional benchmark returns

        Returns:
            PerformanceMetrics
        """
        self.logger.info("Calculating performance metrics")

        returns_array = np.array(returns)

        # Return metrics
        total_return = self._calculate_total_return(equity_curve)
        annualized_return = self._calculate_annualized_return(returns_array)
        cagr = self._calculate_cagr(equity_curve)

        # Risk metrics
        volatility = self._calculate_volatility(returns_array)
        sharpe_ratio = self._calculate_sharpe_ratio(returns_array)
        sortino_ratio = self._calculate_sortino_ratio(returns_array)
        calmar_ratio = self._calculate_calmar_ratio(returns_array, equity_curve)

        # Drawdown metrics
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        avg_drawdown = self._calculate_avg_drawdown(equity_curve)
        max_drawdown_duration = self._calculate_max_drawdown_duration(equity_curve)

        # Trade metrics
        total_trades = len(positions)
        win_rate = self._calculate_win_rate(positions)
        avg_win = self._calculate_avg_win(positions)
        avg_loss = self._calculate_avg_loss(positions)
        profit_factor = self._calculate_profit_factor(positions)

        # Portfolio metrics
        avg_positions = self._calculate_avg_positions(positions)
        turnover = self._calculate_turnover(positions, equity_curve)

        # Benchmark metrics
        alpha = None
        beta = None
        information_ratio = None

        if benchmark_returns:
            alpha, beta, information_ratio = self._calculate_benchmark_metrics(
                returns_array, np.array(benchmark_returns)
            )

        # Additional metrics
        skewness = self._calculate_skewness(returns_array)
        kurtosis = self._calculate_kurtosis(returns_array)

        metrics = PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            cagr=cagr,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            avg_drawdown=avg_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            total_trades=total_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_positions=avg_positions,
            turnover=turnover,
            alpha=alpha,
            beta=beta,
            information_ratio=information_ratio,
            skewness=skewness,
            kurtosis=kurtosis,
        )

        self.logger.info(
            f"Performance metrics calculated: Sharpe={sharpe_ratio:.2f}, Total Return={total_return:.2%}"
        )

        return metrics

    def _calculate_total_return(self, equity_curve: list[tuple[datetime, float]]) -> float:
        """Calculate total return."""
        if len(equity_curve) < 2:
            return 0.0

        initial_value = equity_curve[0][1]
        final_value = equity_curve[-1][1]

        return (final_value - initial_value) / initial_value

    def _calculate_annualized_return(self, returns: np.ndarray) -> float:
        """Calculate annualized return."""
        if len(returns) == 0:
            return 0.0

        mean_return = np.mean(returns)
        return mean_return * 252

    def _calculate_cagr(self, equity_curve: list[tuple[datetime, float]]) -> float:
        """Calculate compound annual growth rate."""
        if len(equity_curve) < 2:
            return 0.0

        initial_value = equity_curve[0][1]
        final_value = equity_curve[-1][1]

        start_date = equity_curve[0][0]
        end_date = equity_curve[-1][0]

        years = (end_date - start_date).days / 365.25

        if years <= 0:
            return 0.0

        return (final_value / initial_value) ** (1 / years) - 1

    def _calculate_volatility(self, returns: np.ndarray) -> float:
        """Calculate annualized volatility."""
        if len(returns) == 0:
            return 0.0

        return np.std(returns) * np.sqrt(252)

    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.06) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.06) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return 0.0

        downside_std = np.std(downside_returns)

        if downside_std == 0:
            return 0.0

        return np.mean(excess_returns) / downside_std * np.sqrt(252)

    def _calculate_calmar_ratio(
        self, returns: np.ndarray, equity_curve: list[tuple[datetime, float]]
    ) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        annualized_return = self._calculate_annualized_return(returns)
        max_drawdown = abs(self._calculate_max_drawdown(equity_curve))

        if max_drawdown == 0:
            return 0.0

        return annualized_return / max_drawdown

    def _calculate_max_drawdown(self, equity_curve: list[tuple[datetime, float]]) -> float:
        """Calculate maximum drawdown."""
        if len(equity_curve) < 2:
            return 0.0

        values = [v for _, v in equity_curve]
        cumulative = np.array(values)

        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max

        return drawdown.min()

    def _calculate_avg_drawdown(self, equity_curve: list[tuple[datetime, float]]) -> float:
        """Calculate average drawdown."""
        if len(equity_curve) < 2:
            return 0.0

        values = [v for _, v in equity_curve]
        cumulative = np.array(values)

        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max

        # Only consider drawdowns (negative values)
        drawdowns = drawdown[drawdown < 0]

        if len(drawdowns) == 0:
            return 0.0

        return drawdowns.mean()

    def _calculate_max_drawdown_duration(self, equity_curve: list[tuple[datetime, float]]) -> int:
        """Calculate maximum drawdown duration in days."""
        if len(equity_curve) < 2:
            return 0

        values = [v for _, v in equity_curve]
        dates = [d for d, _ in equity_curve]

        cumulative = np.array(values)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max

        # Find drawdown periods
        in_drawdown = False
        drawdown_start = None
        max_duration = 0

        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                drawdown_start = dates[i]
            elif dd >= 0 and in_drawdown:
                in_drawdown = False
                duration = (dates[i] - drawdown_start).days
                max_duration = max(max_duration, duration)

        return max_duration

    def _calculate_win_rate(self, positions: list[Any]) -> float:
        """Calculate win rate."""
        if not positions:
            return 0.0

        winning_positions = [p for p in positions if p.pnl > 0]
        return len(winning_positions) / len(positions)

    def _calculate_avg_win(self, positions: list[Any]) -> float:
        """Calculate average win."""
        winning_positions = [p for p in positions if p.pnl > 0]

        if not winning_positions:
            return 0.0

        return np.mean([p.pnl for p in winning_positions])

    def _calculate_avg_loss(self, positions: list[Any]) -> float:
        """Calculate average loss."""
        losing_positions = [p for p in positions if p.pnl < 0]

        if not losing_positions:
            return 0.0

        return np.mean([p.pnl for p in losing_positions])

    def _calculate_profit_factor(self, positions: list[Any]) -> float:
        """Calculate profit factor."""
        gross_profit = sum(p.pnl for p in positions if p.pnl > 0)
        gross_loss = abs(sum(p.pnl for p in positions if p.pnl < 0))

        if gross_loss == 0:
            return 0.0

        return gross_profit / gross_loss

    def _calculate_avg_positions(self, positions: list[Any]) -> float:
        """Calculate average number of positions."""
        if not positions:
            return 0.0

        # This is a simplified calculation
        # In practice, would need to track positions over time
        return len(positions) / 10.0  # Placeholder

    def _calculate_turnover(
        self, positions: list[Any], equity_curve: list[tuple[datetime, float]]
    ) -> float:
        """Calculate portfolio turnover."""
        if not positions or len(equity_curve) < 2:
            return 0.0

        # Simplified turnover calculation
        total_trades = len(positions)
        avg_portfolio_value = np.mean([v for _, v in equity_curve])

        return (total_trades * avg_portfolio_value * 0.02) / (
            avg_portfolio_value * len(equity_curve)
        )

    def _calculate_benchmark_metrics(
        self, returns: np.ndarray, benchmark_returns: np.ndarray
    ) -> tuple[float, float, float]:
        """Calculate benchmark-relative metrics."""
        if len(returns) != len(benchmark_returns):
            return None, None, None

        # Calculate beta
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)

        beta = covariance / benchmark_variance if benchmark_variance > 0 else None

        # Calculate alpha
        excess_returns = returns - benchmark_returns
        alpha = np.mean(excess_returns) * 252 if beta is not None else None

        # Calculate information ratio
        tracking_error = np.std(excess_returns) * np.sqrt(252)
        information_ratio = alpha / tracking_error if tracking_error > 0 else None

        return alpha, beta, information_ratio

    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate return skewness."""
        if len(returns) == 0:
            return 0.0

        from scipy.stats import skew

        return skew(returns)

    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate return kurtosis."""
        if len(returns) == 0:
            return 0.0

        from scipy.stats import kurtosis

        return kurtosis(returns)


# Global performance calculator instance
performance_calculator = PerformanceCalculator()
