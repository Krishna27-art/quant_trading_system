"""
Backtest Comparison and Benchmarking

Institutional-grade comparison of backtest results against benchmarks.
Enables strategy evaluation and comparison.
"""

from datetime import datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from utils.logger import get_logger

logger = get_logger("benchmarking")


class BenchmarkData(BaseModel):
    """Benchmark data for comparison."""

    name: str = Field(..., description="Benchmark name")
    returns: list[float] = Field(..., description="Benchmark returns")
    equity_curve: list[tuple[datetime, float]] = Field(
        default_factory=list, description="Benchmark equity curve"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ComparisonResult(BaseModel):
    """Result of backtest comparison."""

    strategy_name: str = Field(..., description="Strategy name")
    benchmark_name: str = Field(..., description="Benchmark name")

    # Performance comparison
    strategy_return: float = Field(..., description="Strategy total return")
    benchmark_return: float = Field(..., description="Benchmark total return")
    excess_return: float = Field(..., description="Excess return over benchmark")

    # Risk-adjusted comparison
    strategy_sharpe: float = Field(..., description="Strategy Sharpe ratio")
    benchmark_sharpe: float = Field(..., description="Benchmark Sharpe ratio")
    sharpe_difference: float = Field(..., description="Sharpe ratio difference")

    # Drawdown comparison
    strategy_max_dd: float = Field(..., description="Strategy max drawdown")
    benchmark_max_dd: float = Field(..., description="Benchmark max drawdown")

    # Beta and Alpha
    beta: float = Field(..., description="Strategy beta vs benchmark")
    alpha: float = Field(..., description="Strategy alpha vs benchmark")

    # Information ratio
    information_ratio: float = Field(..., description="Information ratio")
    tracking_error: float = Field(..., description="Tracking error")

    # Win rate vs benchmark
    win_rate_vs_benchmark: float = Field(..., description="Win rate vs benchmark")

    # Up/Down capture
    up_capture: float = Field(..., description="Up capture ratio")
    down_capture: float = Field(..., description="Down capture ratio")

    # Overall assessment
    outperformed: bool = Field(..., description="Whether strategy outperformed benchmark")
    assessment: str = Field(..., description="Overall assessment")

    compared_at: datetime = Field(
        default_factory=datetime.now, description="When comparison was performed"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class BenchmarkComparator:
    """
    Comparator for backtest results vs benchmarks.

    Enables strategy evaluation and comparison.
    """

    def __init__(self):
        """Initialize the benchmark comparator."""
        self.logger = logger

    def compare_to_benchmark(
        self,
        strategy_returns: list[float],
        strategy_equity: list[tuple[datetime, float]],
        benchmark: BenchmarkData,
    ) -> ComparisonResult:
        """
        Compare strategy to benchmark.

        Args:
            strategy_returns: Strategy returns
            strategy_equity: Strategy equity curve
            benchmark: Benchmark data

        Returns:
            ComparisonResult
        """
        self.logger.info(f"Comparing strategy to {benchmark.name}")

        strategy_returns_array = np.array(strategy_returns)
        benchmark_returns_array = np.array(benchmark.returns)

        # Calculate returns
        strategy_return = self._calculate_total_return(strategy_equity)
        benchmark_return = self._calculate_total_return(benchmark.equity_curve)
        excess_return = strategy_return - benchmark_return

        # Calculate Sharpe ratios
        strategy_sharpe = self._calculate_sharpe(strategy_returns_array)
        benchmark_sharpe = self._calculate_sharpe(benchmark_returns_array)
        sharpe_difference = strategy_sharpe - benchmark_sharpe

        # Calculate max drawdowns
        strategy_max_dd = abs(self._calculate_max_drawdown(strategy_equity))
        benchmark_max_dd = abs(self._calculate_max_drawdown(benchmark.equity_curve))

        # Calculate beta and alpha
        beta = self._calculate_beta(strategy_returns_array, benchmark_returns_array)
        alpha = self._calculate_alpha(strategy_returns_array, benchmark_returns_array)

        # Calculate information ratio and tracking error
        information_ratio, tracking_error = self._calculate_information_metrics(
            strategy_returns_array, benchmark_returns_array
        )

        # Calculate win rate vs benchmark
        win_rate_vs_benchmark = self._calculate_win_rate_vs_benchmark(
            strategy_returns_array, benchmark_returns_array
        )

        # Calculate up/down capture
        up_capture, down_capture = self._calculate_capture_ratios(
            strategy_returns_array, benchmark_returns_array
        )

        # Determine overall assessment
        outperformed = excess_return > 0 and sharpe_difference > 0
        assessment = self._generate_assessment(
            excess_return, sharpe_difference, strategy_max_dd, benchmark_max_dd
        )

        result = ComparisonResult(
            strategy_name="Strategy",
            benchmark_name=benchmark.name,
            strategy_return=strategy_return,
            benchmark_return=benchmark_return,
            excess_return=excess_return,
            strategy_sharpe=strategy_sharpe,
            benchmark_sharpe=benchmark_sharpe,
            sharpe_difference=sharpe_difference,
            strategy_max_dd=strategy_max_dd,
            benchmark_max_dd=benchmark_max_dd,
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio,
            tracking_error=tracking_error,
            win_rate_vs_benchmark=win_rate_vs_benchmark,
            up_capture=up_capture,
            down_capture=down_capture,
            outperformed=outperformed,
            assessment=assessment,
        )

        self.logger.info(f"Comparison completed: {assessment}")

        return result

    def compare_strategies(
        self, strategies: dict[str, tuple[list[float], list[tuple[datetime, float]]]]
    ) -> pd.DataFrame:
        """
        Compare multiple strategies.

        Args:
            strategies: Dictionary mapping strategy names to (returns, equity_curve) tuples

        Returns:
            DataFrame with comparison results
        """
        self.logger.info(f"Comparing {len(strategies)} strategies")

        results = []

        for strategy_name, (returns, equity_curve) in strategies.items():
            metrics = {
                "strategy": strategy_name,
                "total_return": self._calculate_total_return(equity_curve),
                "sharpe_ratio": self._calculate_sharpe(np.array(returns)),
                "max_drawdown": abs(self._calculate_max_drawdown(equity_curve)),
                "volatility": np.std(returns) * np.sqrt(252),
            }
            results.append(metrics)

        df = pd.DataFrame(results)

        # Sort by Sharpe ratio
        df = df.sort_values("sharpe_ratio", ascending=False)

        return df

    def _calculate_total_return(self, equity_curve: list[tuple[datetime, float]]) -> float:
        """Calculate total return from equity curve."""
        if len(equity_curve) < 2:
            return 0.0

        initial_value = equity_curve[0][1]
        final_value = equity_curve[-1][1]

        return (final_value - initial_value) / initial_value

    def _calculate_sharpe(self, returns: np.ndarray, risk_free_rate: float = 0.06) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def _calculate_max_drawdown(self, equity_curve: list[tuple[datetime, float]]) -> float:
        """Calculate maximum drawdown."""
        if len(equity_curve) < 2:
            return 0.0

        values = [v for _, v in equity_curve]
        cumulative = np.array(values)

        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max

        return drawdown.min()

    def _calculate_beta(self, strategy_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calculate beta."""
        if len(strategy_returns) != len(benchmark_returns) or len(strategy_returns) == 0:
            return 0.0

        covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)

        if benchmark_variance == 0:
            return 0.0

        return covariance / benchmark_variance

    def _calculate_alpha(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        risk_free_rate: float = 0.06,
    ) -> float:
        """Calculate alpha."""
        if len(strategy_returns) != len(benchmark_returns) or len(strategy_returns) == 0:
            return 0.0

        beta = self._calculate_beta(strategy_returns, benchmark_returns)

        daily_rf = risk_free_rate / 252
        strategy_excess = strategy_returns - daily_rf
        benchmark_excess = benchmark_returns - daily_rf

        alpha = np.mean(strategy_excess) - beta * np.mean(benchmark_excess)

        return alpha * 252  # Annualize

    def _calculate_information_metrics(
        self, strategy_returns: np.ndarray, benchmark_returns: np.ndarray
    ) -> tuple[float, float]:
        """Calculate information ratio and tracking error."""
        if len(strategy_returns) != len(benchmark_returns) or len(strategy_returns) == 0:
            return 0.0, 0.0

        excess_returns = strategy_returns - benchmark_returns
        tracking_error = np.std(excess_returns) * np.sqrt(252)

        if tracking_error == 0:
            return 0.0, 0.0

        information_ratio = np.mean(excess_returns) * 252 / tracking_error

        return information_ratio, tracking_error

    def _calculate_win_rate_vs_benchmark(
        self, strategy_returns: np.ndarray, benchmark_returns: np.ndarray
    ) -> float:
        """Calculate win rate vs benchmark."""
        if len(strategy_returns) != len(benchmark_returns) or len(strategy_returns) == 0:
            return 0.0

        wins = np.sum(strategy_returns > benchmark_returns)
        total = len(strategy_returns)

        return wins / total if total > 0 else 0.0

    def _calculate_capture_ratios(
        self, strategy_returns: np.ndarray, benchmark_returns: np.ndarray
    ) -> tuple[float, float]:
        """Calculate up and down capture ratios."""
        if len(strategy_returns) != len(benchmark_returns) or len(strategy_returns) == 0:
            return 0.0, 0.0

        # Up capture
        up_mask = benchmark_returns > 0
        if np.sum(up_mask) > 0:
            strategy_up = strategy_returns[up_mask].sum()
            benchmark_up = benchmark_returns[up_mask].sum()
            up_capture = strategy_up / benchmark_up if benchmark_up > 0 else 0.0
        else:
            up_capture = 0.0

        # Down capture
        down_mask = benchmark_returns < 0
        if np.sum(down_mask) > 0:
            strategy_down = strategy_returns[down_mask].sum()
            benchmark_down = benchmark_returns[down_mask].sum()
            down_capture = strategy_down / benchmark_down if benchmark_down < 0 else 0.0
        else:
            down_capture = 0.0

        return up_capture, down_capture

    def _generate_assessment(
        self,
        excess_return: float,
        sharpe_difference: float,
        strategy_max_dd: float,
        benchmark_max_dd: float,
    ) -> str:
        """Generate overall assessment."""
        if excess_return > 0.1 and sharpe_difference > 0.5:
            return "Strong outperformance with superior risk-adjusted returns"
        elif excess_return > 0.05 and sharpe_difference > 0.2:
            return "Moderate outperformance with good risk-adjusted returns"
        elif excess_return > 0:
            return "Outperforms benchmark but with similar risk-adjusted returns"
        elif excess_return < -0.05:
            return "Significant underperformance vs benchmark"
        elif excess_return < 0:
            return "Underperforms benchmark"
        else:
            return "Performance similar to benchmark"


# Global benchmark comparator instance
benchmark_comparator = BenchmarkComparator()
