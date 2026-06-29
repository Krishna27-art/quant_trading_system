"""
Risk Metrics Calculation

Institutional-grade risk metrics for portfolio analysis.
Calculates VaR, CVaR, beta, and other risk measures.
"""

from datetime import datetime
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field

from utils.logger import get_logger

logger = get_logger("risk_metrics")


class RiskMetricType(str, Enum):
    """Types of risk metrics."""

    VAR = "var"
    CVAR = "cvar"
    BETA = "beta"
    ALPHA = "alpha"
    TRACKING_ERROR = "tracking_error"
    INFORMATION_RATIO = "information_ratio"
    TURNOVER = "turnover"
    CONCENTRATION = "concentration"


class VaRMethod(str, Enum):
    """VaR calculation methods."""

    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"


class RiskMetrics(BaseModel):
    """Risk metrics for a portfolio."""

    # VaR metrics
    var_95: float = Field(..., description="95% Value at Risk")
    var_99: float = Field(..., description="99% Value at Risk")
    var_99_5: float = Field(..., description="99.5% Value at Risk")

    # CVaR metrics
    cvar_95: float = Field(..., description="95% Conditional VaR")
    cvar_99: float = Field(..., description="99% Conditional VaR")

    # Beta and Alpha
    beta: float | None = Field(None, description="Beta vs benchmark")
    alpha: float | None = Field(None, description="Alpha vs benchmark")

    # Tracking error
    tracking_error: float | None = Field(None, description="Tracking error vs benchmark")

    # Concentration
    herfindahl_index: float = Field(..., description="Herfindahl-Hirschman Index")
    max_position_weight: float = Field(..., description="Maximum position weight")

    # Volatility metrics
    daily_volatility: float = Field(..., description="Daily volatility")
    annualized_volatility: float = Field(..., description="Annualized volatility")

    # Skewness and kurtosis
    skewness: float | None = Field(None, description="Return skewness")
    kurtosis: float | None = Field(None, description="Return kurtosis")

    # Tail risk
    expected_shortfall: float | None = Field(None, description="Expected shortfall")
    tail_ratio: float | None = Field(None, description="Tail ratio (95th/5th percentile)")

    calculated_at: datetime = Field(
        default_factory=datetime.now, description="When metrics were calculated"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class RiskCalculator:
    """
    Risk calculator for portfolio analysis.

    Calculates institutional-grade risk metrics.
    """

    def __init__(self):
        """Initialize the risk calculator."""
        self.logger = logger

    def calculate_var(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95,
        method: VaRMethod = VaRMethod.HISTORICAL,
    ) -> float:
        """
        Calculate Value at Risk.

        Args:
            returns: Array of returns
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            method: VaR calculation method

        Returns:
            VaR value
        """
        if len(returns) == 0:
            return 0.0

        if method == VaRMethod.HISTORICAL:
            return self._historical_var(returns, confidence_level)
        elif method == VaRMethod.PARAMETRIC:
            return self._parametric_var(returns, confidence_level)
        elif method == VaRMethod.MONTE_CARLO:
            return self._monte_carlo_var(returns, confidence_level)
        else:
            raise ValueError(f"Unsupported VaR method: {method}")

    def _historical_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate historical VaR."""
        return np.percentile(returns, (1 - confidence_level) * 100)

    def _parametric_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate parametric VaR assuming normal distribution."""
        from scipy.stats import norm

        mean = np.mean(returns)
        std = np.std(returns)

        z_score = norm.ppf(1 - confidence_level)
        var = mean + z_score * std

        return var

    def _monte_carlo_var(
        self, returns: np.ndarray, confidence_level: float, n_simulations: int = 10000
    ) -> float:
        """Calculate Monte Carlo VaR."""
        mean = np.mean(returns)
        std = np.std(returns)

        # Simulate returns
        simulated_returns = np.random.normal(mean, std, n_simulations)

        return np.percentile(simulated_returns, (1 - confidence_level) * 100)

    def calculate_cvar(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).

        Args:
            returns: Array of returns
            confidence_level: Confidence level

        Returns:
            CVaR value
        """
        if len(returns) == 0:
            return 0.0

        var = self.calculate_var(returns, confidence_level)

        # Average of returns worse than VaR
        worst_returns = returns[returns <= var]

        if len(worst_returns) == 0:
            return var

        return np.mean(worst_returns)

    def calculate_beta(self, portfolio_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """
        Calculate beta vs benchmark.

        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns

        Returns:
            Beta value
        """
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) == 0:
            return 0.0

        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)

        if benchmark_variance == 0:
            return 0.0

        return covariance / benchmark_variance

    def calculate_alpha(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        risk_free_rate: float = 0.06,
    ) -> float:
        """
        Calculate alpha vs benchmark.

        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            risk_free_rate: Risk-free rate (annual)

        Returns:
            Alpha value (annualized)
        """
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) == 0:
            return 0.0

        beta = self.calculate_beta(portfolio_returns, benchmark_returns)

        # Daily risk-free rate
        daily_rf = risk_free_rate / 252

        # Calculate excess returns
        portfolio_excess = portfolio_returns - daily_rf
        benchmark_excess = benchmark_returns - daily_rf

        # Alpha = mean(portfolio_excess) - beta * mean(benchmark_excess)
        alpha = np.mean(portfolio_excess) - beta * np.mean(benchmark_excess)

        # Annualize
        return alpha * 252

    def calculate_tracking_error(
        self, portfolio_returns: np.ndarray, benchmark_returns: np.ndarray
    ) -> float:
        """
        Calculate tracking error vs benchmark.

        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns

        Returns:
            Tracking error (annualized)
        """
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) == 0:
            return 0.0

        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = np.std(excess_returns)

        # Annualize
        return tracking_error * np.sqrt(252)

    def calculate_concentration(self, weights: np.ndarray) -> tuple[float, float]:
        """
        Calculate concentration metrics.

        Args:
            weights: Portfolio weights

        Returns:
            Tuple of (Herfindahl index, max position weight)
        """
        if len(weights) == 0:
            return 0.0, 0.0

        # Herfindahl-Hirschman Index
        hhi = np.sum(weights**2)

        # Max position weight
        max_weight = np.max(weights)

        return hhi, max_weight

    def calculate_tail_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate tail ratio (95th percentile / 5th percentile).

        Args:
            returns: Array of returns

        Returns:
            Tail ratio
        """
        if len(returns) == 0:
            return 0.0

        percentile_95 = np.percentile(returns, 95)
        percentile_5 = np.percentile(returns, 5)

        if percentile_5 == 0:
            return 0.0

        return abs(percentile_95 / percentile_5)

    def calculate_all_risk_metrics(
        self,
        returns: list[float],
        weights: list[float] | None = None,
        benchmark_returns: list[float] | None = None,
    ) -> RiskMetrics:
        """
        Calculate all risk metrics.

        Args:
            returns: Portfolio returns
            weights: Portfolio weights (for concentration)
            benchmark_returns: Benchmark returns (for beta/alpha)

        Returns:
            RiskMetrics
        """
        self.logger.info("Calculating risk metrics")

        returns_array = np.array(returns)

        # VaR metrics
        var_95 = self.calculate_var(returns_array, 0.95)
        var_99 = self.calculate_var(returns_array, 0.99)
        var_99_5 = self.calculate_var(returns_array, 0.995)

        # CVaR metrics
        cvar_95 = self.calculate_cvar(returns_array, 0.95)
        cvar_99 = self.calculate_cvar(returns_array, 0.99)

        # Beta and Alpha
        beta = None
        alpha = None
        tracking_error = None

        if benchmark_returns:
            benchmark_array = np.array(benchmark_returns)
            beta = self.calculate_beta(returns_array, benchmark_array)
            alpha = self.calculate_alpha(returns_array, benchmark_array)
            tracking_error = self.calculate_tracking_error(returns_array, benchmark_array)

        # Concentration
        if weights:
            weights_array = np.array(weights)
            hhi, max_weight = self.calculate_concentration(weights_array)
        else:
            hhi = 0.0
            max_weight = 0.0

        # Volatility
        daily_volatility = np.std(returns_array)
        annualized_volatility = daily_volatility * np.sqrt(252)

        # Skewness and kurtosis
        try:
            from scipy.stats import kurtosis, skew

            skewness = skew(returns_array)
            kurt = kurtosis(returns_array)
        except ImportError:
            skewness = None
            kurt = None

        # Tail risk
        expected_shortfall = cvar_95
        tail_ratio = self.calculate_tail_ratio(returns_array)

        metrics = RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            var_99_5=var_99_5,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            beta=beta,
            alpha=alpha,
            tracking_error=tracking_error,
            herfindahl_index=hhi,
            max_position_weight=max_weight,
            daily_volatility=daily_volatility,
            annualized_volatility=annualized_volatility,
            skewness=skewness,
            kurtosis=kurt,
            expected_shortfall=expected_shortfall,
            tail_ratio=tail_ratio,
        )

        self.logger.info(f"Risk metrics calculated: VaR95={var_95:.4f}, Beta={beta:.2f}")

        return metrics


# Global risk calculator instance
risk_calculator = RiskCalculator()
