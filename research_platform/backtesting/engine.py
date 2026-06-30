"""
Backtesting Engine

Institutional-grade backtesting for strategy evaluation.
Follows the frozen trading contract specifications.

Performance:
- BLAS optimization enabled for NumPy operations
"""

from collections import deque
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from research_platform.backtesting.transaction_costs import (
    TransactionCostCalculator,
    TransactionCostModel,
)
from research_platform.research.deflated_sharpe import (
    DeflatedSharpeCalculator,
    DeflatedSharpeConfig,
)
from utils.logger import get_logger

logger = get_logger("backtesting_engine")

# Enable BLAS optimization for NumPy
try:
    from utils.blas_config import auto_configure

    blas_library = auto_configure()
    logger.info(f"BLAS optimization enabled: {blas_library}")
except ImportError:
    logger.warning("BLAS configuration not available, using default NumPy")


class RebalanceFrequency(str, Enum):
    """Rebalancing frequency."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


# Periods-per-year used for annualization/Sharpe scaling, keyed to the actual
# spacing of the returns series. Previously the engine always used 252
# (trading days/year) regardless of rebalance_frequency, which overstates
# annualized return and inflates the Sharpe ratio by sqrt(252/periods) for
# any non-daily rebalance — e.g. ~2.2x inflation for weekly rebalancing.
PERIODS_PER_YEAR = {
    RebalanceFrequency.DAILY: 252,
    RebalanceFrequency.WEEKLY: 52,
    RebalanceFrequency.MONTHLY: 12,
    RebalanceFrequency.QUARTERLY: 4,
}


class BacktestConfig(BaseModel):
    """Configuration for backtesting."""

    # Time period
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")

    # Portfolio parameters
    initial_capital: float = Field(default=10000000, description="Initial capital (10M default)")
    max_positions: int = Field(default=50, description="Maximum number of positions")
    position_size: float = Field(default=0.02, description="Position size as fraction of capital")

    # Rebalancing
    rebalance_frequency: RebalanceFrequency = Field(
        default=RebalanceFrequency.WEEKLY, description="Rebalancing frequency"
    )

    # Transaction costs
    commission_rate: float = Field(default=0.001, description="Commission rate (0.1%)")
    slippage_rate: float = Field(default=0.001, description="Slippage rate (0.1%)")

    # Risk management
    max_position_size: float = Field(
        default=0.1, description="Maximum position size as fraction of capital"
    )
    stop_loss: float | None = Field(
        default=None,
        description="Stop loss threshold as a negative fraction, e.g. -0.05 for -5%. "
        "Checked against current mark-to-market price every rebalance, not just at exit.",
    )
    take_profit: float | None = Field(
        default=None, description="Optional take-profit threshold as a positive fraction"
    )

    # Date alignment
    max_date_alignment_lookback_days: int = Field(
        default=5,
        description="If a rebalance date isn't a trading day, look back up to this many "
        "calendar days for the most recent available trading day instead of silently "
        "skipping the rebalance.",
    )

    # Benchmark
    benchmark: str | None = Field(default="NIFTY50", description="Benchmark index")

    @model_validator(mode="after")
    def validate_dates(self) -> "BacktestConfig":
        if self.start_date > self.end_date:
            raise ValueError("start_date cannot be after end_date")
        return self

    @property
    def periods_per_year(self) -> int:
        return PERIODS_PER_YEAR[self.rebalance_frequency]

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class Position(BaseModel):
    """Position in a security."""

    symbol: str = Field(..., description="Stock symbol")
    entry_date: datetime = Field(..., description="Entry date")
    exit_date: datetime | None = Field(None, description="Exit date")

    # Position details
    shares: float = Field(..., description="Number of shares")
    entry_price: float = Field(..., description="Entry price")
    exit_price: float | None = Field(None, description="Exit price")

    # P&L
    pnl: float = Field(default=0.0, description="Profit/Loss")
    pnl_pct: float = Field(default=0.0, description="Profit/Loss percentage")

    # Metadata
    signal: int = Field(default=1, description="Signal (1: long, -1: short)")
    confidence: float = Field(default=1.0, description="Prediction confidence")
    exit_reason: str = Field(default="", description="signal_change | stop_loss | take_profit | end_of_backtest")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class Portfolio(BaseModel):
    """Portfolio state at a point in time."""

    date: datetime = Field(..., description="Portfolio date")
    cash: float = Field(..., description="Cash available")
    positions: list[Position] = Field(default_factory=list, description="Current positions")

    # Portfolio value
    total_value: float = Field(..., description="Total portfolio value")
    long_value: float = Field(default=0.0, description="Long positions value")
    short_value: float = Field(default=0.0, description="Short positions value")

    # Performance
    daily_return: float = Field(default=0.0, description="Daily return")
    cumulative_return: float = Field(default=0.0, description="Cumulative return")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class BacktestResult(BaseModel):
    """Result of backtesting."""

    config: BacktestConfig = Field(..., description="Backtest configuration")

    # Performance metrics
    total_return: float = Field(..., description="Total return")
    annualized_return: float = Field(..., description="Annualized return")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    deflated_sharpe: float | None = Field(None, description="Deflated Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown")

    # Trade statistics
    total_trades: int = Field(..., description="Total number of trades")
    win_rate: float = Field(..., description="Win rate")
    avg_win: float = Field(..., description="Average win")
    avg_loss: float = Field(..., description="Average loss")
    profit_factor: float = Field(..., description="Profit factor")
    stop_loss_exits: int = Field(default=0, description="Trades closed by stop loss")
    take_profit_exits: int = Field(default=0, description="Trades closed by take profit")
    signal_change_exits: int = Field(default=0, description="Trades closed by signal change")

    # Portfolio statistics
    avg_positions: float = Field(..., description="Average number of positions")
    turnover: float = Field(..., description="Portfolio turnover (annualized)")

    # Time series - use deque with maxlen to prevent unbounded memory growth
    equity_curve: deque = Field(
        default_factory=lambda: deque(maxlen=10000), description="Equity curve"
    )
    returns_series: deque = Field(
        default_factory=lambda: deque(maxlen=10000), description="Daily returns"
    )

    # Positions - use deque with maxlen to prevent unbounded memory growth
    all_positions: deque = Field(
        default_factory=lambda: deque(maxlen=50000), description="All positions"
    )

    # Metadata
    backtest_start: datetime = Field(
        default_factory=datetime.now, description="When backtest started"
    )
    backtest_end: datetime = Field(default_factory=datetime.now, description="When backtest ended")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class BacktestingEngine:
    """
    Institutional-grade backtesting engine.

    Follows the frozen trading contract specifications.
    """

    def __init__(self, config: BacktestConfig):
        """Initialize the backtesting engine."""
        self.config = config
        self.logger = logger

        self._portfolio_history: list[Portfolio] = []
        self._all_positions: list[Position] = []
        self._current_portfolio: Portfolio | None = None

        # Pre-indexed available trading dates per symbol, populated in
        # run_backtest() from price_data. Used by _resolve_trading_date to
        # align a calendar rebalance_date to the most recent actual trading
        # day instead of silently returning an empty price dict.
        self._available_dates: pd.DatetimeIndex | None = None

        # Initialize transaction cost calculator
        cost_model = TransactionCostModel(
            commission_rate=config.commission_rate, slippage_rate=config.slippage_rate
        )
        self.cost_calculator = TransactionCostCalculator(cost_model)

        # Initialize deflated Sharpe calculator
        self.deflated_sharpe_calc = DeflatedSharpeCalculator(DeflatedSharpeConfig())

    def run_backtest(self, predictions: list[Any], price_data: pd.DataFrame) -> BacktestResult:
        """
        Run backtest with given predictions.

        Args:
            predictions: List of predictions from model
            price_data: DataFrame with adjusted prices

        Returns:
            BacktestResult
        """
        self.logger.info(
            f"Starting backtest from {self.config.start_date} to {self.config.end_date}"
        )

        # Level 1: Data Validation - Check for NaN before proceeding
        if price_data.isna().mean().max() > 0.05:
            raise ValueError("Price data contains >5% NaN values - cannot proceed with backtest")

        if price_data.isna().all().any():
            raise ValueError("Price data contains all-NaN columns - cannot proceed with backtest")

        if len(price_data) < 100:
            raise ValueError(f"Insufficient observations: {len(price_data)} < 100")

        self._available_dates = pd.DatetimeIndex(sorted(price_data["date"].unique()))

        # Initialize portfolio
        self._initialize_portfolio()

        # Generate rebalance dates, aligned to actual trading days
        rebalance_dates = self._generate_rebalance_dates()

        # Run backtest
        for raw_date in rebalance_dates:
            resolved_date = self._resolve_trading_date(raw_date)
            if resolved_date is None:
                self.logger.warning(
                    f"No trading day found within "
                    f"{self.config.max_date_alignment_lookback_days} days of {raw_date}; "
                    f"skipping this rebalance explicitly (logged, not silent)."
                )
                continue

            date_predictions = self._get_predictions_for_date(predictions, resolved_date)
            self._rebalance_portfolio(resolved_date, date_predictions, price_data)

        # Liquidate all remaining open positions at end of backtest
        self._liquidate_open_positions(price_data)

        # Calculate results
        result = self._calculate_results()

        self.logger.info(f"Backtest completed. Total return: {result.total_return:.2%}")

        return result

    def _resolve_trading_date(self, raw_date: datetime) -> datetime | None:
        """
        Map a calendar rebalance date to the nearest actual trading day
        on or before it (never after — that would be lookahead).

        Previously _get_symbol_prices used an exact equality match against
        price_data["date"], so any rebalance_date landing on a weekend or
        market holiday (guaranteed to happen routinely for MONTHLY/QUARTERLY
        rebalancing, which steps by fixed 30/90 calendar days) silently
        returned an empty price dict and the rebalance was skipped with no
        log line — i.e. large unexplained gaps in the strategy's trading
        activity.
        """
        if self._available_dates is None or len(self._available_dates) == 0:
            return None

        raw_ts = pd.Timestamp(raw_date)
        candidates = self._available_dates[self._available_dates <= raw_ts]
        if candidates.empty:
            return None

        resolved = candidates[-1]
        lag_days = (raw_ts - resolved).days
        if lag_days > self.config.max_date_alignment_lookback_days:
            return None

        if lag_days > 0:
            self.logger.info(
                f"Rebalance date {raw_date.date()} is not a trading day; "
                f"aligned to {resolved.date()} ({lag_days}d lookback)"
            )
        return resolved.to_pydatetime()

    def _initialize_portfolio(self):
        """Initialize portfolio with initial capital."""
        self._current_portfolio = Portfolio(
            date=self.config.start_date,
            cash=self.config.initial_capital,
            positions=[],
            total_value=self.config.initial_capital,
            long_value=0.0,
            short_value=0.0,
            daily_return=0.0,
            cumulative_return=0.0,
        )

        self._portfolio_history.append(self._current_portfolio)

    def _generate_rebalance_dates(self) -> list[datetime]:
        """Generate rebalance dates based on frequency."""
        dates = []
        current_date = self.config.start_date

        while current_date <= self.config.end_date:
            dates.append(current_date)

            if self.config.rebalance_frequency == RebalanceFrequency.DAILY:
                current_date += timedelta(days=1)
            elif self.config.rebalance_frequency == RebalanceFrequency.WEEKLY:
                current_date += timedelta(weeks=1)
            elif self.config.rebalance_frequency == RebalanceFrequency.MONTHLY:
                current_date += timedelta(days=30)
            elif self.config.rebalance_frequency == RebalanceFrequency.QUARTERLY:
                current_date += timedelta(days=90)

        return dates

    def _get_predictions_for_date(self, predictions: list[Any], date: datetime) -> list[Any]:
        """Get predictions for a specific date."""
        date_predictions = [p for p in predictions if p.date == date]
        return date_predictions

    def _rebalance_portfolio(
        self, date: datetime, predictions: list[Any], price_data: pd.DataFrame
    ):
        """
        Rebalance portfolio based on predictions.

        Args:
            date: Rebalance date (already resolved to an actual trading day)
            predictions: Predictions for this date
            price_data: Price data
        """
        self.logger.info(f"Rebalancing portfolio on {date}")

        current_positions = self._current_portfolio.positions.copy()
        symbol_prices = self._get_symbol_prices(date, price_data)

        new_positions = []
        for position in current_positions:
            current_price = symbol_prices.get(position.symbol, position.entry_price)

            close_reason = self._evaluate_close_reason(position, predictions, current_price)

            if close_reason is not None:
                base_exit_price = current_price
                exit_price = base_exit_price * (1 - self.config.slippage_rate)

                position.exit_date = date
                position.exit_price = exit_price
                position.exit_reason = close_reason
                position.pnl = (exit_price - position.entry_price) * position.shares
                position.pnl_pct = (exit_price - position.entry_price) / position.entry_price

                sell_cost = self.cost_calculator.calculate_sell_cost(position.shares, exit_price)
                self._current_portfolio.cash += (position.shares * exit_price) - sell_cost

                self._all_positions.append(position)
            else:
                new_positions.append(position)

        # Open new positions based on predictions
        top_predictions = self._get_top_predictions(predictions)

        for pred in top_predictions:
            if len(new_positions) >= self.config.max_positions:
                break

            if any(p.symbol == pred.symbol for p in new_positions):
                continue

            base_price = symbol_prices.get(pred.symbol, 0)
            if base_price <= 0:
                continue

            price = base_price * (1 + self.config.slippage_rate)

            position_value = min(
                self.config.position_size * self._current_portfolio.total_value,
                self.config.max_position_size * self._current_portfolio.total_value,
            )

            shares = position_value / price

            buy_cost = self.cost_calculator.calculate_buy_cost(shares, price)

            total_cost = (shares * price) + buy_cost
            if total_cost > self._current_portfolio.cash:
                continue

            new_position = Position(
                symbol=pred.symbol,
                entry_date=date,
                shares=shares,
                entry_price=price,
                signal=1,
                confidence=pred.confidence,
            )

            new_positions.append(new_position)
            self._current_portfolio.cash -= total_cost

        long_value = sum(
            p.shares * symbol_prices.get(p.symbol, p.entry_price) for p in new_positions
        )
        short_value = 0.0  # Assuming long-only for now

        total_value = self._current_portfolio.cash + long_value - short_value

        self._current_portfolio = Portfolio(
            date=date,
            cash=self._current_portfolio.cash,
            positions=new_positions,
            total_value=total_value,
            long_value=long_value,
            short_value=short_value,
            daily_return=(total_value - self._portfolio_history[-1].total_value)
            / self._portfolio_history[-1].total_value,
            cumulative_return=(total_value - self.config.initial_capital)
            / self.config.initial_capital,
        )

        self._portfolio_history.append(self._current_portfolio)

    def _get_symbol_prices(self, date: datetime, price_data: pd.DataFrame) -> dict[str, float]:
        """Get prices for all symbols on a given (already-resolved trading) date."""
        date_data = price_data[price_data["date"] == date]

        prices = {}
        for _, row in date_data.iterrows():
            prices[row["symbol"]] = row["adjusted_close"]

        return prices

    def _evaluate_close_reason(
        self, position: Position, predictions: list[Any], current_price: float
    ) -> str | None:
        """
        Determine if a position should be closed, and why.

        Previously the stop_loss config field existed but was never actually
        checked (`if self.config.stop_loss: pass`) — meaning a configured
        stop loss had zero effect on the backtest and every "stop-loss
        protected" position could run to an arbitrary loss until the next
        signal change. This now actually enforces stop_loss/take_profit
        against the current mark-to-market price at every rebalance.
        """
        if position.entry_price <= 0:
            return None

        pnl_pct = (current_price - position.entry_price) / position.entry_price

        if self.config.stop_loss is not None and pnl_pct <= self.config.stop_loss:
            return "stop_loss"

        if self.config.take_profit is not None and pnl_pct >= self.config.take_profit:
            return "take_profit"

        signal_changed = any(
            pred.symbol == position.symbol and pred.prediction != 2 for pred in predictions
        )
        if signal_changed:
            return "signal_change"

        return None

    def _get_top_predictions(self, predictions: list[Any]) -> list[Any]:
        """Get top predictions for new positions."""
        top_predictions = [p for p in predictions if p.prediction == 2]
        top_predictions.sort(key=lambda x: x.confidence, reverse=True)
        return top_predictions

    def _liquidate_open_positions(self, price_data: pd.DataFrame) -> None:
        """
        Liquidate all remaining open positions at the end of the backtest.

        Previously, positions that were still open at the final backtest date
        were never closed or counted in total_trades, leading to meaningless
        win_rate calculations (division by zero when no trades closed during
        the backtest). This method ensures all positions are liquidated at
        the final available price and counted in the results.
        """
        if not self._current_portfolio or not self._current_portfolio.positions:
            return

        final_date = self._current_portfolio.date
        symbol_prices = self._get_symbol_prices(final_date, price_data)

        for position in self._current_portfolio.positions:
            if position.exit_date is not None:
                continue  # Already closed

            current_price = symbol_prices.get(position.symbol, position.entry_price)
            if current_price <= 0:
                self.logger.warning(
                    f"Cannot liquidate {position.symbol}: no price available at {final_date}"
                )
                continue

            exit_price = current_price * (1 - self.config.slippage_rate)

            position.exit_date = final_date
            position.exit_price = exit_price
            position.exit_reason = "end_of_backtest"
            position.pnl = (exit_price - position.entry_price) * position.shares
            position.pnl_pct = (exit_price - position.entry_price) / position.entry_price

            sell_cost = self.cost_calculator.calculate_sell_cost(position.shares, exit_price)
            self._current_portfolio.cash += (position.shares * exit_price) - sell_cost

            self._all_positions.append(position)
            self.logger.info(
                f"Liquidated {position.symbol} at end of backtest: "
                f"PnL={position.pnl:.2f} ({position.pnl_pct:.2%})"
            )

    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest results."""
        if not self._portfolio_history:
            raise ValueError("No portfolio history to calculate results")

        returns = [p.daily_return for p in self._portfolio_history[1:]]
        equity_curve = [(p.date, p.total_value) for p in self._portfolio_history]

        returns_clean = [r for r in returns if not np.isnan(r) and not np.isinf(r)]
        if len(returns_clean) < len(returns):
            self.logger.warning(
                f"Removed {len(returns) - len(returns_clean)} NaN/Inf values from returns"
            )

        if not returns_clean:
            raise ValueError("No valid returns after cleaning NaN/Inf values")

        periods_per_year = self.config.periods_per_year

        total_return = self._portfolio_history[-1].cumulative_return
        annualized_return = (1 + total_return) ** (periods_per_year / len(returns_clean)) - 1

        sharpe_ratio = self._calculate_sharpe_ratio(returns_clean, periods_per_year)
        sortino_ratio = self._calculate_sortino_ratio(returns_clean, periods_per_year)
        max_drawdown = self._calculate_max_drawdown(equity_curve)

        deflated_sharpe = None
        if len(returns_clean) >= 30:
            try:
                deflated_result = self.deflated_sharpe_calc.calculate_deflated_sharpe(returns_clean)
                deflated_sharpe = deflated_result.deflated_sharpe
                self.logger.info(f"Deflated Sharpe calculated: {deflated_sharpe:.3f}")
            except Exception as e:
                self.logger.warning(f"Failed to calculate deflated Sharpe: {e}")

        total_trades = len(self._all_positions)
        win_rate = self._calculate_win_rate()
        avg_win = self._calculate_avg_win()
        avg_loss = self._calculate_avg_loss()
        profit_factor = self._calculate_profit_factor()
        stop_loss_exits = sum(1 for p in self._all_positions if p.exit_reason == "stop_loss")
        take_profit_exits = sum(1 for p in self._all_positions if p.exit_reason == "take_profit")
        signal_change_exits = sum(
            1 for p in self._all_positions if p.exit_reason == "signal_change"
        )

        avg_positions = np.mean([len(p.positions) for p in self._portfolio_history])
        turnover = self._calculate_turnover(periods_per_year)

        result = BacktestResult(
            config=self.config,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            deflated_sharpe=deflated_sharpe,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            total_trades=total_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            stop_loss_exits=stop_loss_exits,
            take_profit_exits=take_profit_exits,
            signal_change_exits=signal_change_exits,
            avg_positions=avg_positions,
            turnover=turnover,
            equity_curve=equity_curve,
            returns_series=returns_clean,
            all_positions=self._all_positions,
        )

        return result

    def _calculate_sharpe_ratio(self, returns: list[float], periods_per_year: int) -> float:
        """Calculate Sharpe ratio, scaled by the actual return periodicity."""
        if not returns or np.std(returns) == 0:
            return 0.0

        return np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year)

    def _calculate_sortino_ratio(self, returns: list[float], periods_per_year: int) -> float:
        """Calculate Sortino ratio, scaled by the actual return periodicity."""
        if not returns:
            return 0.0

        downside_returns = [r for r in returns if r < 0]

        if not downside_returns:
            return 0.0

        downside_std = np.std(downside_returns)

        if downside_std == 0:
            return 0.0

        return np.mean(returns) / downside_std * np.sqrt(periods_per_year)

    def _calculate_max_drawdown(self, equity_curve: list[tuple[datetime, float]]) -> float:
        """Calculate maximum drawdown."""
        if not equity_curve:
            return 0.0

        values = [v for _, v in equity_curve]
        cumulative = np.array(values)

        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max

        return drawdown.min()

    def _calculate_win_rate(self) -> float:
        """Calculate win rate."""
        if not self._all_positions:
            return 0.0

        winning_positions = [p for p in self._all_positions if p.pnl > 0]
        return len(winning_positions) / len(self._all_positions)

    def _calculate_avg_win(self) -> float:
        """Calculate average win."""
        winning_positions = [p for p in self._all_positions if p.pnl > 0]

        if not winning_positions:
            return 0.0

        return np.mean([p.pnl for p in winning_positions])

    def _calculate_avg_loss(self) -> float:
        """Calculate average loss."""
        losing_positions = [p for p in self._all_positions if p.pnl < 0]

        if not losing_positions:
            return 0.0

        return np.mean([p.pnl for p in losing_positions])

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor."""
        gross_profit = sum(p.pnl for p in self._all_positions if p.pnl > 0)
        gross_loss = abs(sum(p.pnl for p in self._all_positions if p.pnl < 0))

        if gross_loss == 0:
            return 0.0

        return gross_profit / gross_loss

    def _calculate_turnover(self, periods_per_year: int) -> float:
        """
        Calculate annualized portfolio turnover as
        (total value bought+sold) / (2 * avg portfolio value) / years.

        Previously this was
        (total_trades * avg_portfolio_value * position_size) /
        (avg_portfolio_value * len(history))
        which algebraically reduces to just
        total_trades * position_size / len(history) — avg_portfolio_value
        cancels out entirely, so the "value" terms in that formula did
        nothing, and the result wasn't turnover by any standard definition
        (turnover should reflect *actual traded notional* relative to
        portfolio size, not just a trade count scaled by a constant).
        """
        if len(self._portfolio_history) < 2 or not self._all_positions:
            return 0.0

        avg_portfolio_value = np.mean([p.total_value for p in self._portfolio_history])
        if avg_portfolio_value <= 0:
            return 0.0

        traded_notional = sum(
            p.shares * p.entry_price + p.shares * (p.exit_price or p.entry_price)
            for p in self._all_positions
        )

        num_periods = len(self._portfolio_history) - 1
        years = num_periods / periods_per_year
        if years <= 0:
            return 0.0

        return (traded_notional / (2 * avg_portfolio_value)) / years