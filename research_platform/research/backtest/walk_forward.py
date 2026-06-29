"""
Walk-Forward Backtest with Transaction Costs

No lookahead. No survivorship bias. Full Indian market costs.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("walk_forward_backtest")


@dataclass
class BacktestConfig:
    """Configuration for walk-forward backtest."""

    top_n: int = 10  # Number of top stocks to hold
    rebalance_freq: str = "W"  # Rebalance frequency (D, W, M)
    transaction_cost: float = 0.001  # 0.1% per trade (NSE realistic)
    brokerage_rate: float = 0.0005  # 0.05% brokerage
    stt_rate: float = 0.001  # 0.1% STT on sell
    impact_rate: float = 0.0003  # 0.03% market impact
    min_price: float = 10.0  # Minimum price for trade
    min_volume: float = 1000000.0  # Minimum daily volume

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "top_n": self.top_n,
            "rebalance_freq": self.rebalance_freq,
            "transaction_cost": self.transaction_cost,
            "brokerage_rate": self.brokerage_rate,
            "stt_rate": self.stt_rate,
            "impact_rate": self.impact_rate,
            "min_price": self.min_price,
            "min_volume": self.min_volume,
        }


@dataclass
class BacktestResults:
    """Results from walk-forward backtest."""

    returns_df: pd.DataFrame  # Period-by-period returns
    gross_return: float  # Total gross return
    net_return: float  # Total net return
    total_cost: float  # Total transaction costs
    avg_turnover: float  # Average turnover
    sharpe_ratio: float  # Sharpe ratio
    max_drawdown: float  # Maximum drawdown
    win_rate: float  # Win rate
    n_periods: int = 0
    config: BacktestConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gross_return": self.gross_return,
            "net_return": self.net_return,
            "total_cost": self.total_cost,
            "avg_turnover": self.avg_turnover,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "n_periods": self.n_periods,
            "config": self.config.to_dict() if self.config else None,
        }


def calculate_indian_transaction_costs(
    trade_value: float, is_sell: bool, config: BacktestConfig
) -> float:
    """
    Calculate Indian market transaction costs.

    NSE costs: ~0.05% brokerage + 0.1% STT on sell + 0.03% impact = ~0.1% round trip

    Args:
        trade_value: Value of the trade
        is_sell: Whether this is a sell (STT applies only on sells)
        config: Backtest configuration

    Returns:
        Total transaction cost in currency units
    """
    # Brokerage (0.05%)
    brokerage = trade_value * config.brokerage_rate

    # STT (0.1% on sell only)
    stt = trade_value * config.stt_rate if is_sell else 0.0

    # Market impact (0.03%)
    impact = trade_value * config.impact_rate

    total_cost = brokerage + stt + impact

    return total_cost


def walk_forward_backtest(
    predictions_df: pd.DataFrame,
    price_df: pd.DataFrame,
    config: BacktestConfig | None = None,
    volume_df: pd.DataFrame | None = None,
) -> BacktestResults:
    """
    Walk-forward: train on past, predict future, never look ahead.

    NSE costs: ~0.05% brokerage + 0.1% STT on sell + 0.03% impact = ~0.1% round trip

    Args:
        predictions_df: DataFrame with predictions (date x symbol)
        price_df: DataFrame with prices (date x symbol)
        config: Backtest configuration
        volume_df: DataFrame with volumes (date x symbol) - optional for filtering

    Returns:
        Backtest results
    """
    if config is None:
        config = BacktestConfig()

    portfolio_returns = []
    current_positions = set()
    current_prices = {}  # Track entry prices for cost calculation

    # Get rebalance dates
    rebalance_dates = pd.date_range(
        start=predictions_df.index[0], end=predictions_df.index[-1], freq=config.rebalance_freq
    )

    for rebalance_date in rebalance_dates:
        if rebalance_date not in predictions_df.index:
            continue

        # Get predictions on rebalance date (already produced by model yesterday)
        scores = predictions_df.loc[rebalance_date].dropna()

        # Filter by minimum price if price data available
        if rebalance_date in price_df.index:
            current_prices_df = price_df.loc[rebalance_date].dropna()
            scores = scores[scores.index.isin(current_prices_df.index)]
            scores = scores[current_prices_df[scores.index] >= config.min_price]

        # Filter by minimum volume if volume data available
        if volume_df is not None and rebalance_date in volume_df.index:
            current_volumes_df = volume_df.loc[rebalance_date].dropna()
            scores = scores[scores.index.isin(current_volumes_df.index)]
            scores = scores[current_volumes_df[scores.index] >= config.min_volume]

        # Select top N stocks
        if len(scores) < config.top_n:
            logger.warning(
                f"Insufficient stocks on {rebalance_date}: {len(scores)} < {config.top_n}"
            )
            continue

        new_positions = set(scores.nlargest(config.top_n).index)

        # Calculate turnover and cost
        exits = current_positions - new_positions
        entries = new_positions - current_positions
        turnover = (len(exits) + len(entries)) / (2 * config.top_n)

        # Calculate transaction costs
        total_cost = 0.0

        # Exit costs (sell)
        for stock in exits:
            if rebalance_date in price_df.index and stock in price_df.columns:
                exit_price = price_df.loc[rebalance_date, stock]
                if pd.notna(exit_price) and stock in current_prices:
                    entry_price = current_prices[stock]
                    trade_value = entry_price * 100  # Assume 100 shares
                    cost = calculate_indian_transaction_costs(
                        trade_value, is_sell=True, config=config
                    )
                    total_cost += cost

        # Entry costs (buy)
        for stock in entries:
            if rebalance_date in price_df.index and stock in price_df.columns:
                entry_price = price_df.loc[rebalance_date, stock]
                if pd.notna(entry_price):
                    trade_value = entry_price * 100  # Assume 100 shares
                    cost = calculate_indian_transaction_costs(
                        trade_value, is_sell=False, config=config
                    )
                    total_cost += cost
                    current_prices[stock] = entry_price

        # Calculate period return
        next_dates = price_df.index[price_df.index > rebalance_date]
        if len(next_dates) == 0:
            continue

        # Get next rebalance date or end of period
        next_rebalance = None
        for future_date in rebalance_dates[rebalance_dates > rebalance_date]:
            if future_date in price_df.index:
                next_rebalance = future_date
                break

        if next_rebalance is None:
            # Use last available date
            end_date = next_dates[-1]
        else:
            # Use date before next rebalance
            end_date = (
                next_dates[next_dates < next_rebalance][-1]
                if len(next_dates[next_dates < next_rebalance]) > 0
                else next_dates[0]
            )

        period_returns = []
        for stock in new_positions:
            if stock not in price_df.columns:
                continue

            start_price = price_df.loc[rebalance_date, stock]
            end_price = price_df.loc[end_date, stock]

            if pd.notna(start_price) and pd.notna(end_price) and start_price > 0:
                period_returns.append(end_price / start_price - 1)

        if period_returns:
            gross_ret = np.mean(period_returns)
            net_ret = gross_ret - (total_cost / (config.top_n * 100))  # Normalize cost per position
            portfolio_returns.append(
                {
                    "date": rebalance_date,
                    "gross_return": gross_ret,
                    "net_return": net_ret,
                    "cost": total_cost,
                    "turnover": turnover,
                    "n_positions": len(new_positions),
                }
            )

        current_positions = new_positions

    if not portfolio_returns:
        logger.warning("No valid periods in backtest")
        return BacktestResults(
            returns_df=pd.DataFrame(),
            gross_return=0.0,
            net_return=0.0,
            total_cost=0.0,
            avg_turnover=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            n_periods=0,
            config=config,
        )

    returns_df = pd.DataFrame(portfolio_returns).set_index("date")

    # Calculate cumulative returns
    returns_df["cumulative_gross"] = (1 + returns_df["gross_return"]).cumprod()
    returns_df["cumulative_net"] = (1 + returns_df["net_return"]).cumprod()

    # Calculate metrics
    gross_return = returns_df["cumulative_gross"].iloc[-1] - 1
    net_return = returns_df["cumulative_net"].iloc[-1] - 1
    total_cost = returns_df["cost"].sum()
    avg_turnover = returns_df["turnover"].mean()

    # Calculate Sharpe ratio (annualized)
    if len(returns_df) > 1:
        daily_returns = returns_df["net_return"]
        sharpe_ratio = (
            daily_returns.mean() / daily_returns.std() * np.sqrt(252)
            if daily_returns.std() > 0
            else 0.0
        )
    else:
        sharpe_ratio = 0.0

    # Calculate maximum drawdown
    cumulative_net = returns_df["cumulative_net"]
    running_max = cumulative_net.expanding().max()
    drawdown = (cumulative_net - running_max) / running_max
    max_drawdown = drawdown.min()

    # Calculate win rate
    win_rate = (returns_df["net_return"] > 0).sum() / len(returns_df)

    results = BacktestResults(
        returns_df=returns_df,
        gross_return=gross_return,
        net_return=net_return,
        total_cost=total_cost,
        avg_turnover=avg_turnover,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        n_periods=len(returns_df),
        config=config,
    )

    # Print summary
    print("\n=== Walk-Forward Backtest Results ===")
    print(f"Gross Return: {gross_return:.2%}")
    print(f"Net Return: {net_return:.2%}")
    print(f"Total Cost: {total_cost:.2f}")
    print(f"Average Turnover: {avg_turnover:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Number of Periods: {len(returns_df)}")
    print(f"Top N Stocks: {config.top_n}")
    print(f"Rebalance Frequency: {config.rebalance_freq}")
    print(f"Transaction Cost: {config.transaction_cost:.2%}")
    print(
        f"NSE Costs: {config.brokerage_rate:.2%} brokerage + {config.stt_rate:.2%} STT + {config.impact_rate:.2%} impact"
    )

    return results


def validate_backtest_results(
    results: BacktestResults, min_sharpe: float = 1.0, min_net_return: float = 0.10
) -> bool:
    """
    Validate backtest results.

    Args:
        results: Backtest results
        min_sharpe: Minimum acceptable Sharpe ratio
        min_net_return: Minimum acceptable net return

    Returns:
        True if results are valid, False otherwise
    """
    if results.sharpe_ratio < min_sharpe:
        logger.warning(f"Sharpe ratio {results.sharpe_ratio:.2f} below threshold {min_sharpe:.2f}")
        return False

    if results.net_return < min_net_return:
        logger.warning(f"Net return {results.net_return:.2%} below threshold {min_net_return:.2%}")
        return False

    if results.max_drawdown < -0.20:  # More than 20% drawdown
        logger.warning(f"Max drawdown {results.max_drawdown:.2%} exceeds threshold -20%")
        return False

    logger.info(
        f"Backtest validation passed: Sharpe={results.sharpe_ratio:.2f}, Net Return={results.net_return:.2%}"
    )
    return True
