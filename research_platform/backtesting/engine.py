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

from utils.logger import get_logger

logger = get_logger("backtesting")
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

# Enable BLAS optimization for NumPy
try:
    from utils.blas_config import auto_configure

    blas_library = auto_configure()
    logger.info(f"BLAS optimization enabled: {blas_library}")
except ImportError:
    logger.warning("BLAS configuration not available, using default NumPy")

logger = get_logger("backtesting_engine")


class RebalanceFrequency(str, Enum):
    """Rebalancing frequency."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


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
    stop_loss: float | None = Field(default=None, description="Stop loss threshold")

    # Benchmark
    benchmark: str | None = Field(default="NIFTY50", description="Benchmark index")

    @model_validator(mode="after")
    def validate_dates(self) -> "BacktestConfig":
        if self.start_date > self.end_date:
            raise ValueError("start_date cannot be after end_date")
        return self

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

    # Portfolio statistics
    avg_positions: float = Field(..., description="Average number of positions")
    turnover: float = Field(..., description="Portfolio turnover")

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

        # Initialize portfolio
        self._initialize_portfolio()

        # Generate rebalance dates
        rebalance_dates = self._generate_rebalance_dates()

        # Run backtest
        for rebalance_date in rebalance_dates:
            # Get predictions for this date
            date_predictions = self._get_predictions_for_date(predictions, rebalance_date)

            # Rebalance portfolio
            self._rebalance_portfolio(rebalance_date, date_predictions, price_data)

        # Calculate results
        result = self._calculate_results()

        self.logger.info(f"Backtest completed. Total return: {result.total_return:.2%}")

        return result

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
        # Filter predictions for this date
        date_predictions = [p for p in predictions if p.date == date]
        return date_predictions

    def _rebalance_portfolio(
        self, date: datetime, predictions: list[Any], price_data: pd.DataFrame
    ):
        """
        Rebalance portfolio based on predictions.

        Args:
            date: Rebalance date
            predictions: Predictions for this date
            price_data: Price data
        """
        self.logger.info(f"Rebalancing portfolio on {date}")

        # Get current positions
        current_positions = self._current_portfolio.positions.copy()

        # Get prices for all symbols
        symbol_prices = self._get_symbol_prices(date, price_data)

        # Close positions that should be closed
        new_positions = []
        for position in current_positions:
            should_close = self._should_close_position(position, predictions, date)

            if should_close:
                # Close position
                base_exit_price = symbol_prices.get(position.symbol, position.entry_price)

                # Apply slippage to exit price (worse price for sell)
                exit_price = base_exit_price * (1 - self.config.slippage_rate)

                position.exit_date = date
                position.exit_price = exit_price
                position.pnl = (exit_price - position.entry_price) * position.shares
                position.pnl_pct = (exit_price - position.entry_price) / position.entry_price

                # Calculate transaction costs using cost calculator
                sell_cost = self.cost_calculator.calculate_sell_cost(position.shares, exit_price)

                # Update cash (net of costs)
                self._current_portfolio.cash += (position.shares * exit_price) - sell_cost

                # Add to all positions
                self._all_positions.append(position)
            else:
                new_positions.append(position)

        # Open new positions based on predictions
        top_predictions = self._get_top_predictions(predictions)

        for pred in top_predictions:
            if len(new_positions) >= self.config.max_positions:
                break

            # Check if already in position
            if any(p.symbol == pred.symbol for p in new_positions):
                continue

            # Calculate position size
            base_price = symbol_prices.get(pred.symbol, 0)
            if base_price <= 0:
                continue

            # Apply slippage to buy price (worse price for buy)
            price = base_price * (1 + self.config.slippage_rate)

            position_value = min(
                self.config.position_size * self._current_portfolio.total_value,
                self.config.max_position_size * self._current_portfolio.total_value,
            )

            shares = position_value / price

            # Calculate transaction costs using cost calculator
            buy_cost = self.cost_calculator.calculate_buy_cost(shares, price)

            # Check if enough cash (including costs)
            total_cost = (shares * price) + buy_cost
            if total_cost > self._current_portfolio.cash:
                continue

            # Open position
            new_position = Position(
                symbol=pred.symbol,
                entry_date=date,
                shares=shares,
                entry_price=price,
                signal=1,
                confidence=pred.confidence,
            )

            new_positions.append(new_position)

            # Update cash (net of costs)
            self._current_portfolio.cash -= total_cost

        # Calculate portfolio value
        long_value = sum(
            p.shares * symbol_prices.get(p.symbol, p.entry_price) for p in new_positions
        )
        short_value = 0.0  # Assuming long-only for now

        total_value = self._current_portfolio.cash + long_value - short_value

        # Update portfolio
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
        """Get prices for all symbols on a given date."""
        date_data = price_data[price_data["date"] == date]

        prices = {}
        for _, row in date_data.iterrows():
            prices[row["symbol"]] = row["adjusted_close"]

        return prices

    def _should_close_position(
        self, position: Position, predictions: list[Any], date: datetime
    ) -> bool:
        """Determine if a position should be closed."""
        # Check for stop loss
        if self.config.stop_loss:
            # This would require current price - simplified for now
            pass

        # Check if signal changed
        return any(pred.symbol == position.symbol and pred.prediction != 2 for pred in predictions)

    def _get_top_predictions(self, predictions: list[Any]) -> list[Any]:
        """Get top predictions for new positions."""
        # Filter for top class (2)
        top_predictions = [p for p in predictions if p.prediction == 2]

        # Sort by confidence
        top_predictions.sort(key=lambda x: x.confidence, reverse=True)

        return top_predictions

    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest results."""
        if not self._portfolio_history:
            raise ValueError("No portfolio history to calculate results")

        # Extract returns
        returns = [p.daily_return for p in self._portfolio_history[1:]]  # Skip first day
        equity_curve = [(p.date, p.total_value) for p in self._portfolio_history]

        # Level 3: NaN Repair - Handle NaN in returns before calculation
        returns_clean = [r for r in returns if not np.isnan(r) and not np.isinf(r)]
        if len(returns_clean) < len(returns):
            self.logger.warning(
                f"Removed {len(returns) - len(returns_clean)} NaN/Inf values from returns"
            )

        if not returns_clean:
            raise ValueError("No valid returns after cleaning NaN/Inf values")

        # Calculate metrics
        total_return = self._portfolio_history[-1].cumulative_return
        annualized_return = (1 + total_return) ** (252 / len(returns_clean)) - 1

        sharpe_ratio = self._calculate_sharpe_ratio(returns_clean)
        sortino_ratio = self._calculate_sortino_ratio(returns_clean)
        max_drawdown = self._calculate_max_drawdown(equity_curve)

        # Calculate deflated Sharpe (institutional requirement)
        deflated_sharpe = None
        if len(returns_clean) >= 30:  # Minimum observations required
            try:
                deflated_result = self.deflated_sharpe_calc.calculate_deflated_sharpe(returns_clean)
                deflated_sharpe = deflated_result.deflated_sharpe
                self.logger.info(f"Deflated Sharpe calculated: {deflated_sharpe:.3f}")
            except Exception as e:
                self.logger.warning(f"Failed to calculate deflated Sharpe: {e}")

        # Trade statistics
        total_trades = len(self._all_positions)
        win_rate = self._calculate_win_rate()
        avg_win = self._calculate_avg_win()
        avg_loss = self._calculate_avg_loss()
        profit_factor = self._calculate_profit_factor()

        # Portfolio statistics
        avg_positions = np.mean([len(p.positions) for p in self._portfolio_history])
        turnover = self._calculate_turnover()

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
            avg_positions=avg_positions,
            turnover=turnover,
            equity_curve=equity_curve,
            returns_series=returns_clean,
            all_positions=self._all_positions,
        )

        return result

    def _calculate_sharpe_ratio(self, returns: list[float]) -> float:
        """Calculate Sharpe ratio."""
        if not returns or np.std(returns) == 0:
            return 0.0

        return np.mean(returns) / np.std(returns) * np.sqrt(252)

    def _calculate_sortino_ratio(self, returns: list[float]) -> float:
        """Calculate Sortino ratio."""
        if not returns:
            return 0.0

        downside_returns = [r for r in returns if r < 0]

        if not downside_returns:
            return 0.0

        downside_std = np.std(downside_returns)

        if downside_std == 0:
            return 0.0

        return np.mean(returns) / downside_std * np.sqrt(252)

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

    def _calculate_turnover(self) -> float:
        """Calculate portfolio turnover."""
        if len(self._portfolio_history) < 2:
            return 0.0

        total_trades = len(self._all_positions)
        avg_portfolio_value = np.mean([p.total_value for p in self._portfolio_history])

        return (total_trades * avg_portfolio_value * self.config.position_size) / (
            avg_portfolio_value * len(self._portfolio_history)
        )
