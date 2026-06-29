"""
Deflated Sharpe for Backtests

Every backtest must compute Deflated Sharpe to reduce data-mining risk.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from scipy import stats

from utils.logger import get_logger

logger = get_logger("deflated_sharpe")


@dataclass
class DeflatedSharpeResult:
    """Deflated Sharpe result."""

    sharpe: float
    deflated_sharpe: float
    sharpe_deflation: float  # Deflation factor
    p_value: float  # P-value for Sharpe significance
    is_significant: bool
    num_returns: int
    num_benchmark_returns: int
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sharpe": self.sharpe,
            "deflated_sharpe": self.deflated_sharpe,
            "sharpe_deflation": self.sharpe_deflation,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "num_returns": self.num_returns,
            "num_benchmark_returns": self.num_benchmark_returns,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DeflatedSharpeConfig:
    """Deflated Sharpe configuration."""

    # Significance level
    significance_level: float = 0.05  # 5% significance level

    # Minimum observations
    min_returns: int = 30
    min_benchmark_returns: int = 30

    # Benchmark type
    use_sharpe_ratio_benchmark: bool = True  # Use Sharpe ratio benchmark
    use_returns_benchmark: bool = True  # Use returns benchmark

    # Risk-free rate
    risk_free_rate: float = 0.06  # 6% annual risk-free rate

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "significance_level": self.significance_level,
            "min_returns": self.min_returns,
            "min_benchmark_returns": self.min_benchmark_returns,
            "use_sharpe_ratio_benchmark": self.use_sharpe_ratio_benchmark,
            "use_returns_benchmark": self.use_returns_benchmark,
            "risk_free_rate": self.risk_free_rate,
        }


class DeflatedSharpeCalculator:
    """
    Deflated Sharpe calculator.

    Every backtest must compute Deflated Sharpe to reduce data-mining risk.
    """

    def __init__(self, config: DeflatedSharpeConfig | None = None):
        """Initialize deflated Sharpe calculator."""
        self.config = config or DeflatedSharpeConfig()
        self.logger = logger

        # Benchmark Sharpe ratios (strategy -> Sharpe)
        self.benchmark_sharpes: dict[str, float] = {}

        # Calculation history
        self.calculation_history: list[DeflatedSharpeResult] = []

        # Statistics
        self.total_calculations: int = 0

        self.logger.info("DeflatedSharpeCalculator initialized")

    def add_benchmark_sharpe(self, strategy_name: str, sharpe: float):
        """
        Add benchmark Sharpe ratio.

        Args:
            strategy_name: Strategy name
            sharpe: Sharpe ratio
        """
        self.benchmark_sharpes[strategy_name] = sharpe
        self.logger.debug(f"Added benchmark Sharpe for {strategy_name}: {sharpe:.3f}")

    def calculate_deflated_sharpe(
        self,
        returns: list[float],
        benchmark_returns: list[float] | None = None,
        strategy_name: str | None = None,
    ) -> DeflatedSharpeResult:
        """
        Calculate deflated Sharpe.

        Args:
            returns: Strategy returns
            benchmark_returns: Optional benchmark returns
            strategy_name: Optional strategy name for benchmark Sharpe lookup

        Returns:
            Deflated Sharpe result
        """
        self.total_calculations += 1

        if len(returns) < self.config.min_returns:
            raise ValueError(f"Insufficient returns: {len(returns)} < {self.config.min_returns}")

        # Calculate Sharpe
        sharpe = self._calculate_sharpe(returns)

        # Calculate deflation factor
        deflation_factor = self._calculate_deflation_factor(
            returns, benchmark_returns, strategy_name
        )

        # Calculate deflated Sharpe
        deflated_sharpe = sharpe / deflation_factor if deflation_factor > 0 else 0

        # Calculate p-value
        p_value = self._calculate_sharpe_p_value(returns, sharpe)

        # Check significance
        is_significant = p_value < self.config.significance_level

        result = DeflatedSharpeResult(
            sharpe=sharpe,
            deflated_sharpe=deflated_sharpe,
            sharpe_deflation=deflation_factor,
            p_value=p_value,
            is_significant=is_significant,
            num_returns=len(returns),
            num_benchmark_returns=len(benchmark_returns) if benchmark_returns else 0,
            timestamp=datetime.utcnow(),
        )

        self.calculation_history.append(result)

        # Keep only last 1000 results
        if len(self.calculation_history) > 1000:
            self.calculation_history = self.calculation_history[-1000:]

        self.logger.info(
            f"Deflated Sharpe: sharpe={sharpe:.3f}, "
            f"deflated={deflated_sharpe:.3f}, "
            f"deflation={deflation_factor:.3f}, "
            f"p_value={p_value:.4f}, "
            f"significant={is_significant}"
        )

        return result

    def _calculate_sharpe(self, returns: list[float]) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Returns

        Returns:
            Sharpe ratio
        """
        returns_array = np.array(returns)

        # Calculate annualized return (assuming daily returns)
        mean_return = np.mean(returns_array)
        annualized_return = mean_return * 252

        # Calculate annualized volatility
        volatility = np.std(returns_array)
        annualized_volatility = volatility * np.sqrt(252)

        # Calculate excess return
        excess_return = annualized_return - self.config.risk_free_rate

        # Calculate Sharpe
        if annualized_volatility == 0:
            return 0.0

        sharpe = excess_return / annualized_volatility

        return sharpe

    def _calculate_deflation_factor(
        self,
        returns: list[float],
        benchmark_returns: list[float] | None,
        strategy_name: str | None,
    ) -> float:
        """
        Calculate Sharpe deflation factor.

        Args:
            returns: Strategy returns
            benchmark_returns: Optional benchmark returns
            strategy_name: Optional strategy name

        Returns:
            Deflation factor
        """
        deflation_factors = []

        # Use benchmark Sharpe if available
        if strategy_name and strategy_name in self.benchmark_sharpes:
            benchmark_sharpe = self.benchmark_sharpes[strategy_name]
            strategy_sharpe = self._calculate_sharpe(returns)

            if benchmark_sharpe > 0:
                # Deflation factor based on Sharpe ratio
                sharpe_deflation = 1 + (strategy_sharpe / benchmark_sharpe)
                deflation_factors.append(sharpe_deflation)

        # Use benchmark returns if available
        if benchmark_returns and len(benchmark_returns) >= self.config.min_benchmark_returns:
            benchmark_sharpe = self._calculate_sharpe(benchmark_returns)
            strategy_sharpe = self._calculate_sharpe(returns)

            if benchmark_sharpe > 0:
                # Deflation factor based on returns
                returns_deflation = 1 + (strategy_sharpe / benchmark_sharpe)
                deflation_factors.append(returns_deflation)

        # If no benchmark available, use default deflation
        if not deflation_factors:
            # Default deflation based on number of trials (data mining bias)
            # Assume 100 trials for backtesting
            num_trials = 100
            default_deflation = np.sqrt(1 + np.log(num_trials))
            deflation_factors.append(default_deflation)

        # Use average deflation factor
        deflation_factor = np.mean(deflation_factors)

        return deflation_factor

    def _calculate_sharpe_p_value(self, returns: list[float], sharpe: float) -> float:
        """
        Calculate p-value for Sharpe significance.

        Args:
            returns: Returns
            sharpe: Calculated Sharpe

        Returns:
            P-value
        """
        returns_array = np.array(returns)
        n = len(returns_array)

        # Calculate t-statistic for Sharpe
        # t = Sharpe * sqrt(n)
        t_stat = sharpe * np.sqrt(n)

        # Calculate two-tailed p-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))

        return p_value

    def get_calculation_history(self, limit: int = 100) -> list[DeflatedSharpeResult]:
        """Get calculation history."""
        return self.calculation_history[-limit:]

    def get_latest_result(self) -> DeflatedSharpeResult | None:
        """Get latest calculation result."""
        if self.calculation_history:
            return self.calculation_history[-1]
        return None

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics."""
        significant_count = sum(1 for r in self.calculation_history if r.is_significant)

        return {
            "total_calculations": self.total_calculations,
            "significant_count": significant_count,
            "significance_rate": significant_count / max(1, self.total_calculations),
            "benchmark_strategies": list(self.benchmark_sharpes.keys()),
            "config": self.config.to_dict(),
        }

    def reset(self):
        """Reset calculator."""
        self.benchmark_sharpes.clear()
        self.calculation_history.clear()
        self.total_calculations = 0

        self.logger.info("DeflatedSharpeCalculator reset")


def create_deflated_sharpe_calculator(
    config: DeflatedSharpeConfig | None = None,
) -> DeflatedSharpeCalculator:
    """Factory function to create deflated Sharpe calculator."""
    return DeflatedSharpeCalculator(config)
