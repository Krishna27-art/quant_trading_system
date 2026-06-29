"""
Quantstats Evaluation

The only numbers that matter for backtest evaluation.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("quantstats_eval")


@dataclass
class EvaluationMetrics:
    """Evaluation metrics from quantstats."""

    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    cagr: float
    calmar_ratio: float
    win_rate: float

    # Additional metrics
    volatility: float = 0.0
    best_day: float = 0.0
    worst_day: float = 0.0
    avg_return: float = 0.0
    total_return: float = 0.0

    # Benchmark comparison
    sharpe_vs_benchmark: float | None = None
    alpha: float | None = None
    beta: float | None = None

    # Decision status
    decision: str = "UNKNOWN"  # GREEN, RED, YELLOW
    decision_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "cagr": self.cagr,
            "calmar_ratio": self.calmar_ratio,
            "win_rate": self.win_rate,
            "volatility": self.volatility,
            "best_day": self.best_day,
            "worst_day": self.worst_day,
            "avg_return": self.avg_return,
            "total_return": self.total_return,
            "sharpe_vs_benchmark": self.sharpe_vs_benchmark,
            "alpha": self.alpha,
            "beta": self.beta,
            "decision": self.decision,
            "decision_reasons": self.decision_reasons,
        }


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: Daily returns series
        risk_free_rate: Risk-free rate (annualized)

    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / 252
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

    return sharpe


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sortino ratio (downside deviation).

    Args:
        returns: Daily returns series
        risk_free_rate: Risk-free rate (annualized)

    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return 0.0

    downside_std = downside_returns.std()
    if downside_std == 0:
        return 0.0

    sortino = excess_returns.mean() / downside_std * np.sqrt(252)

    return sortino


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown.

    Args:
        returns: Daily returns series

    Returns:
        Maximum drawdown
    """
    if len(returns) == 0:
        return 0.0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max

    return drawdown.min()


def calculate_cagr(returns: pd.Series) -> float:
    """
    Calculate Compound Annual Growth Rate.

    Args:
        returns: Daily returns series

    Returns:
        CAGR
    """
    if len(returns) == 0:
        return 0.0

    total_return = (1 + returns).prod() - 1
    years = len(returns) / 252

    if years == 0:
        return 0.0

    cagr = (1 + total_return) ** (1 / years) - 1

    return cagr


def calculate_calmar_ratio(returns: pd.Series) -> float:
    """
    Calculate Calmar ratio (CAGR / Max Drawdown).

    Args:
        returns: Daily returns series

    Returns:
        Calmar ratio
    """
    cagr = calculate_cagr(returns)
    max_dd = calculate_max_drawdown(returns)

    if max_dd == 0:
        return 0.0

    calmar = cagr / abs(max_dd)

    return calmar


def calculate_win_rate(returns: pd.Series) -> float:
    """
    Calculate win rate (percentage of positive days).

    Args:
        returns: Daily returns series

    Returns:
        Win rate
    """
    if len(returns) == 0:
        return 0.0

    win_rate = (returns > 0).mean() * 100

    return win_rate


def calculate_volatility(returns: pd.Series) -> float:
    """
    Calculate annualized volatility.

    Args:
        returns: Daily returns series

    Returns:
        Annualized volatility
    """
    if len(returns) == 0:
        return 0.0

    volatility = returns.std() * np.sqrt(252)

    return volatility


def calculate_alpha_beta(returns: pd.Series, benchmark_returns: pd.Series) -> tuple:
    """
    Calculate alpha and beta.

    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns

    Returns:
        Tuple of (alpha, beta)
    """
    if len(returns) == 0 or len(benchmark_returns) == 0:
        return 0.0, 0.0

    # Align series
    common_index = returns.index.intersection(benchmark_returns.index)
    if len(common_index) == 0:
        return 0.0, 0.0

    strategy_aligned = returns.loc[common_index]
    benchmark_aligned = benchmark_returns.loc[common_index]

    # Calculate beta
    covariance = np.cov(strategy_aligned, benchmark_aligned)[0, 1]
    benchmark_variance = benchmark_aligned.var()

    beta = 0.0 if benchmark_variance == 0 else covariance / benchmark_variance

    # Calculate alpha (annualized)
    strategy_mean = strategy_aligned.mean() * 252
    benchmark_mean = benchmark_aligned.mean() * 252
    alpha = strategy_mean - beta * benchmark_mean

    return alpha, beta


def evaluate_strategy(
    returns: pd.Series, benchmark_returns: pd.Series | None = None, risk_free_rate: float = 0.0
) -> EvaluationMetrics:
    """
    Evaluate strategy with quantstats-style metrics.

    Args:
        returns: Daily returns series
        benchmark_returns: Benchmark returns (optional)
        risk_free_rate: Risk-free rate (annualized)

    Returns:
        Evaluation metrics
    """
    # Calculate core metrics
    sharpe_ratio = calculate_sharpe_ratio(returns, risk_free_rate)
    sortino_ratio = calculate_sortino_ratio(returns, risk_free_rate)
    max_drawdown = calculate_max_drawdown(returns)
    cagr = calculate_cagr(returns)
    calmar_ratio = calculate_calmar_ratio(returns)
    win_rate = calculate_win_rate(returns)

    # Calculate additional metrics
    volatility = calculate_volatility(returns)
    best_day = returns.max()
    worst_day = returns.min()
    avg_return = returns.mean()
    total_return = (1 + returns).prod() - 1

    # Calculate benchmark comparison if provided
    sharpe_vs_benchmark = None
    alpha = None
    beta = None

    if benchmark_returns is not None:
        sharpe_vs_benchmark = sharpe_ratio - calculate_sharpe_ratio(
            benchmark_returns, risk_free_rate
        )
        alpha, beta = calculate_alpha_beta(returns, benchmark_returns)

    # Determine decision
    decision, decision_reasons = make_decision(
        sharpe_ratio, max_drawdown, calmar_ratio, cagr, len(returns)
    )

    metrics = EvaluationMetrics(
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown,
        cagr=cagr,
        calmar_ratio=calmar_ratio,
        win_rate=win_rate,
        volatility=volatility,
        best_day=best_day,
        worst_day=worst_day,
        avg_return=avg_return,
        total_return=total_return,
        sharpe_vs_benchmark=sharpe_vs_benchmark,
        alpha=alpha,
        beta=beta,
        decision=decision,
        decision_reasons=decision_reasons,
    )

    # Print decision numbers
    print("\n=== DECISION NUMBERS ===")
    print(f"Sharpe Ratio:        {sharpe_ratio:.2f}   (need > 1.0)")
    print(f"Sortino Ratio:       {sortino_ratio:.2f}   (need > 1.5)")
    print(f"Max Drawdown:        {max_drawdown:.1%}  (need > -20%)")
    print(f"CAGR:                {cagr:.1%}")
    print(f"Calmar Ratio:        {calmar_ratio:.2f}   (need > 0.5)")
    print(f"Win Rate:            {win_rate:.1%}")
    print(f"\nDecision:            {decision}")
    if decision_reasons:
        print("Reasons:")
        for reason in decision_reasons:
            print(f"  - {reason}")

    return metrics


def make_decision(
    sharpe_ratio: float, max_drawdown: float, calmar_ratio: float, cagr: float, n_days: int
) -> tuple:
    """
    Make deployment decision based on metrics.

    Green light — deploy:
    - Sharpe > 1.0
    - Max DD < 20%
    - Calmar > 0.5
    - Positive for 3+ years

    Red light — investigate:
    - Sharpe < 0.7
    - Max DD > 30%
    - Returns clustered in 1 year
    - Works only pre-2020

    Args:
        sharpe_ratio: Sharpe ratio
        max_drawdown: Maximum drawdown
        calmar_ratio: Calmar ratio
        cagr: CAGR
        n_days: Number of days in backtest

    Returns:
        Tuple of (decision, reasons)
    """
    decision = "YELLOW"
    reasons = []

    # Check for insufficient data first (red flag regardless of other metrics)
    years = n_days / 252
    if years < 1:
        decision = "RED"
        reasons.append(f"Insufficient data: {years:.1f} years")
        return decision, reasons

    # Check green light criteria
    green_checks = 0
    if sharpe_ratio > 1.0:
        green_checks += 1
    if max_drawdown > -0.20:
        green_checks += 1
    if calmar_ratio > 0.5:
        green_checks += 1
    if years >= 3 and cagr > 0:
        green_checks += 1

    # Check red light criteria
    red_checks = 0
    if sharpe_ratio < 0.7:
        red_checks += 1
        reasons.append(f"Sharpe ratio {sharpe_ratio:.2f} < 0.7")
    if max_drawdown < -0.30:
        red_checks += 1
        reasons.append(f"Max drawdown {max_drawdown:.1%} > 30%")

    # Make decision
    if red_checks >= 2:
        decision = "RED"
    elif green_checks >= 3:
        decision = "GREEN"
        reasons.append("Meets deployment criteria")
    else:
        reasons.append("Borderline - needs more analysis")

    return decision, reasons


def validate_deployment(metrics: EvaluationMetrics) -> bool:
    """
    Validate if strategy is ready for deployment.

    Args:
        metrics: Evaluation metrics

    Returns:
        True if ready for deployment, False otherwise
    """
    if metrics.decision == "GREEN":
        logger.info(f"Strategy ready for deployment: {metrics.decision}")
        return True

    logger.warning(f"Strategy not ready for deployment: {metrics.decision}")
    for reason in metrics.decision_reasons:
        logger.warning(f"  - {reason}")

    return False
