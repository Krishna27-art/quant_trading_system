"""
Low Turnover Rebalancing

Reduce turnover — costs are killing your Sharpe.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("low_turnover")


@dataclass
class TurnoverMetrics:
    """Metrics for turnover analysis."""

    average_turnover: float
    annual_cost: float
    weekly_turnovers: list[float]
    n_periods: int
    cost_per_trade: float = 0.0012  # 0.12% per trade

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "average_turnover": self.average_turnover,
            "annual_cost": self.annual_cost,
            "weekly_turnovers": self.weekly_turnovers,
            "n_periods": self.n_periods,
            "cost_per_trade": self.cost_per_trade,
        }


@dataclass
class RebalanceConfig:
    """Configuration for low turnover rebalancing."""

    top_n: int = 15
    entry_threshold: float = 0.7  # only enter if in top 70th percentile
    exit_threshold: float = 0.3  # only exit if falls below 30th percentile
    max_turnover: float = 0.25  # max 25% portfolio change per rebalance
    transaction_cost: float = 0.0012  # 0.12% per trade

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "top_n": self.top_n,
            "entry_threshold": self.entry_threshold,
            "exit_threshold": self.exit_threshold,
            "max_turnover": self.max_turnover,
            "transaction_cost": self.transaction_cost,
        }


def low_turnover_rebalance(
    current_positions: dict[str, float],
    new_scores: pd.Series,
    config: RebalanceConfig | None = None,
) -> dict[str, float]:
    """
    Portfolio buffer rebalancing — the highest single ROI change for cost reduction.

    Standard approach: replace bottom stocks every week → 60-80% turnover
    Buffer approach:   only trade when a stock moves significantly → 15-25% turnover

    The buffer zone: stock ranked 11-19 out of 20 stays in portfolio.
    It only exits when it drops below rank 6 (exit_threshold=0.3).

    Args:
        current_positions: Current positions {symbol: weight}
        new_scores: New prediction scores
        config: Rebalance configuration

    Returns:
        New positions {symbol: weight}
    """
    if config is None:
        config = RebalanceConfig()

    percentile_rank = new_scores.rank(pct=True)
    current_set = set(current_positions.keys())

    # Forced exits: current holdings that scored very badly
    force_exit = {
        s
        for s in current_set
        if s in percentile_rank.index and percentile_rank[s] < config.exit_threshold
    }

    # Strong entry candidates: not already held, scored very well
    entry_candidates = percentile_rank[
        (percentile_rank > config.entry_threshold) & (~percentile_rank.index.isin(current_set))
    ].sort_values(ascending=False)

    # Build new portfolio
    keep = current_set - force_exit
    n_slots = config.top_n - len(keep)
    entries = list(entry_candidates.head(n_slots).index)

    new_positions = list(keep) + entries

    # Enforce max turnover
    exits = current_set - set(new_positions)
    additions = set(new_positions) - current_set
    actual_turnover = (len(exits) + len(additions)) / (2 * config.top_n)

    if actual_turnover > config.max_turnover:
        # Too much change — be conservative, keep more existing positions
        n_reduce = int((actual_turnover - config.max_turnover) * 2 * config.top_n)
        entries = entries[:-n_reduce] if n_reduce > 0 else entries
        new_positions = list(keep) + entries

    # Equal weight positions
    return {s: 1.0 / len(new_positions) for s in new_positions}


def measure_turnover(
    rebalance_history: list[set[str]], cost_per_trade: float = 0.0012, rebalance_freq: str = "W"
) -> TurnoverMetrics:
    """
    Measure turnover from rebalance history.

    Args:
        rebalance_history: List of sets of position names per period
        cost_per_trade: Cost per trade (default 0.12%)
        rebalance_freq: Rebalance frequency (D, W, M)

    Returns:
        Turnover metrics
    """
    turnovers = []

    for i in range(1, len(rebalance_history)):
        prev = rebalance_history[i - 1]
        curr = rebalance_history[i]
        n = max(len(prev), len(curr))

        if n == 0:
            continue

        turnover = len(prev.symmetric_difference(curr)) / (2 * n)
        turnovers.append(turnover)

    if not turnovers:
        return TurnoverMetrics(
            average_turnover=0.0,
            annual_cost=0.0,
            weekly_turnovers=[],
            n_periods=0,
            cost_per_trade=cost_per_trade,
        )

    avg_turnover = np.mean(turnovers)

    # Calculate annual cost based on rebalance frequency
    if rebalance_freq == "D":
        periods_per_year = 252
    elif rebalance_freq == "W":
        periods_per_year = 52
    elif rebalance_freq == "M":
        periods_per_year = 12
    else:
        periods_per_year = 52  # default to weekly

    annual_cost = avg_turnover * periods_per_year * cost_per_trade

    logger.info(f"Average turnover per rebalance: {avg_turnover:.1%}")
    logger.info(f"Annual cost (@ {cost_per_trade:.2%} per trade): {annual_cost:.1%}")

    return TurnoverMetrics(
        average_turnover=avg_turnover,
        annual_cost=annual_cost,
        weekly_turnovers=turnovers,
        n_periods=len(turnovers),
        cost_per_trade=cost_per_trade,
    )


def calculate_transaction_cost(
    turnover: float, portfolio_value: float, cost_per_trade: float = 0.0012
) -> float:
    """
    Calculate transaction cost for a given turnover.

    Args:
        turnover: Portfolio turnover (0-1)
        portfolio_value: Total portfolio value
        cost_per_trade: Cost per trade

    Returns:
        Transaction cost in currency units
    """
    # Turnover represents the fraction of portfolio traded
    # Each trade costs cost_per_trade
    cost = turnover * portfolio_value * cost_per_trade

    return cost


def optimize_turnover_parameters(
    predictions_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    config: RebalanceConfig,
    n_trials: int = 10,
) -> RebalanceConfig:
    """
    Optimize turnover parameters to maximize Sharpe.

    Args:
        predictions_df: Predictions DataFrame
        returns_df: Returns DataFrame
        config: Base configuration
        n_trials: Number of trials

    Returns:
        Optimized configuration
    """
    best_config = config
    best_sharpe = 0.0

    # Try different parameter combinations
    for entry_thresh in [0.6, 0.65, 0.7, 0.75, 0.8]:
        for exit_thresh in [0.2, 0.25, 0.3, 0.35, 0.4]:
            for max_turnover in [0.15, 0.2, 0.25, 0.3]:
                test_config = RebalanceConfig(
                    top_n=config.top_n,
                    entry_threshold=entry_thresh,
                    exit_threshold=exit_thresh,
                    max_turnover=max_turnover,
                    transaction_cost=config.transaction_cost,
                )

                # Simulate and calculate Sharpe (simplified)
                # In production, this would run a full backtest
                sharpe = _estimate_sharpe_with_config(predictions_df, returns_df, test_config)

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_config = test_config

    logger.info(
        f"Optimized config: entry={best_config.entry_threshold:.2f}, "
        f"exit={best_config.exit_threshold:.2f}, max_turnover={best_config.max_turnover:.2f}"
    )

    return best_config


def _estimate_sharpe_with_config(
    predictions_df: pd.DataFrame, returns_df: pd.DataFrame, config: RebalanceConfig
) -> float:
    """
    Estimate Sharpe ratio with given configuration (simplified).

    Args:
        predictions_df: Predictions DataFrame
        returns_df: Returns DataFrame
        config: Rebalance configuration

    Returns:
        Estimated Sharpe ratio
    """
    # Simplified estimation - in production, run full backtest
    # This is a placeholder for the actual optimization logic
    return 0.0


def get_turnover_recommendations(current_turnover: float, annual_cost: float) -> list[str]:
    """
    Get recommendations based on turnover analysis.

    Args:
        current_turnover: Current average turnover
        annual_cost: Current annual cost

    Returns:
        List of recommendations
    """
    recommendations = []

    if current_turnover > 0.40:
        recommendations.append(
            "Turnover > 40% - this is likely your biggest Sharpe killer. "
            "Implement buffer rebalancing to cut costs by 40-60%."
        )

    if annual_cost > 0.10:
        recommendations.append(
            f"Annual cost {annual_cost:.1%} is high. "
            "Consider reducing turnover or optimizing execution."
        )

    if current_turnover < 0.15:
        recommendations.append(
            "Turnover is low - good for cost control. "
            "Ensure signal is not being diluted by holding too long."
        )

    if 0.15 <= current_turnover <= 0.25:
        recommendations.append(
            "Turnover is in optimal range (15-25%). Buffer rebalancing is working well."
        )

    return recommendations


def compare_rebalance_strategies(
    predictions_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    standard_config: RebalanceConfig,
    buffer_config: RebalanceConfig,
) -> dict[str, Any]:
    """
    Compare standard vs buffer rebalancing strategies.

    Args:
        predictions_df: Predictions DataFrame
        returns_df: Returns DataFrame
        standard_config: Standard rebalance configuration
        buffer_config: Buffer rebalance configuration

    Returns:
        Comparison results
    """
    # Simulate both strategies (simplified)
    # In production, this would run full backtests

    standard_turnover = 0.60  # Typical for standard rebalancing
    buffer_turnover = 0.20  # Typical for buffer rebalancing

    standard_cost = standard_turnover * 52 * standard_config.transaction_cost
    buffer_cost = buffer_turnover * 52 * buffer_config.transaction_cost

    cost_savings = standard_cost - buffer_cost

    return {
        "standard_turnover": standard_turnover,
        "buffer_turnover": buffer_turnover,
        "standard_annual_cost": standard_cost,
        "buffer_annual_cost": buffer_cost,
        "cost_savings": cost_savings,
        "cost_reduction_pct": cost_savings / standard_cost if standard_cost > 0 else 0.0,
    }
