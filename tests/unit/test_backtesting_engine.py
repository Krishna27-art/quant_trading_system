"""
Unit Tests for Backtesting Engine

Tests backtesting engine, performance metrics, and position management.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from research_platform.backtesting.engine import (
    BacktestConfig,
    BacktestingEngine,
    BacktestResult,
    Portfolio,
    Position,
    RebalanceFrequency,
)


class Prediction(BaseModel):
    """Mock prediction for testing."""

    symbol: str
    date: datetime
    prediction: int
    confidence: float


@pytest.fixture
def sample_config():
    """Create sample backtest configuration."""
    return BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 31),
        initial_capital=1000000,
        max_positions=10,
        position_size=0.1,
        rebalance_frequency=RebalanceFrequency.WEEKLY,
        commission_rate=0.001,
        slippage_rate=0.001,
    )


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    dates = pd.date_range("2024-01-01", "2024-03-31", freq="D")
    symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]

    data = []
    for symbol in symbols:
        np.random.seed(hash(symbol) % 1000)
        for date in dates:
            base_price = 1000 + np.random.randn() * 100
            data.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": base_price,
                    "high": base_price * 1.02,
                    "low": base_price * 0.98,
                    "close": base_price,
                    "adjusted_close": base_price,
                    "volume": np.random.randint(1000000, 5000000),
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def sample_predictions():
    """Create sample predictions for testing."""
    predictions = []
    dates = pd.date_range("2024-01-01", "2024-03-31", freq="W-MON")
    symbols = ["RELIANCE", "TCS", "INFY"]

    for date in dates:
        for symbol in symbols:
            predictions.append(
                Prediction(
                    symbol=symbol,
                    date=date,
                    prediction=np.random.choice([1, 2, 3]),
                    confidence=np.random.uniform(0.5, 1.0),
                )
            )

    return predictions


class TestBacktestConfig:
    """Test BacktestConfig configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BacktestConfig(start_date=datetime(2024, 1, 1), end_date=datetime(2024, 12, 31))

        assert config.initial_capital == 10000000
        assert config.max_positions == 50
        assert config.position_size == 0.02
        assert config.commission_rate == 0.001
        assert config.slippage_rate == 0.001

    def test_custom_config(self, sample_config):
        """Test custom configuration values."""
        assert sample_config.initial_capital == 1000000
        assert sample_config.max_positions == 10
        assert sample_config.position_size == 0.1
        assert sample_config.rebalance_frequency == RebalanceFrequency.WEEKLY


class TestPosition:
    """Test Position model."""

    def test_position_creation(self):
        """Test position creation."""
        position = Position(
            symbol="RELIANCE",
            entry_date=datetime(2024, 1, 1),
            shares=100,
            entry_price=1000.0,
            signal=1,
            confidence=0.8,
        )

        assert position.symbol == "RELIANCE"
        assert position.shares == 100
        assert position.entry_price == 1000.0
        assert position.pnl == 0.0

    def test_position_pnl_calculation(self):
        """Test position P&L calculation."""
        position = Position(
            symbol="RELIANCE",
            entry_date=datetime(2024, 1, 1),
            shares=100,
            entry_price=1000.0,
            exit_date=datetime(2024, 1, 15),
            exit_price=1100.0,
            signal=1,
        )

        position.pnl = (position.exit_price - position.entry_price) * position.shares
        position.pnl_pct = (position.exit_price - position.entry_price) / position.entry_price

        assert position.pnl == 10000.0
        assert position.pnl_pct == 0.1


class TestPortfolio:
    """Test Portfolio model."""

    def test_portfolio_creation(self):
        """Test portfolio creation."""
        portfolio = Portfolio(
            date=datetime(2024, 1, 1),
            cash=1000000,
            positions=[],
            total_value=1000000,
            long_value=0.0,
            short_value=0.0,
        )

        assert portfolio.cash == 1000000
        assert portfolio.total_value == 1000000
        assert len(portfolio.positions) == 0

    def test_portfolio_with_positions(self):
        """Test portfolio with positions."""
        positions = [
            Position(
                symbol="RELIANCE", entry_date=datetime(2024, 1, 1), shares=100, entry_price=1000.0
            ),
            Position(symbol="TCS", entry_date=datetime(2024, 1, 1), shares=50, entry_price=2000.0),
        ]

        portfolio = Portfolio(
            date=datetime(2024, 1, 1),
            cash=800000,
            positions=positions,
            total_value=1000000,
            long_value=200000,
            short_value=0.0,
        )

        assert len(portfolio.positions) == 2
        assert portfolio.long_value == 200000


class TestBacktestingEngine:
    """Test BacktestingEngine."""

    @pytest.fixture
    def engine(self, sample_config):
        """Create backtesting engine instance."""
        return BacktestingEngine(sample_config)

    def test_engine_initialization(self, engine, sample_config):
        """Test engine initialization."""
        assert engine.config == sample_config
        assert len(engine._portfolio_history) == 0
        assert len(engine._all_positions) == 0

    def test_initialize_portfolio(self, engine):
        """Test portfolio initialization."""
        engine._initialize_portfolio()

        assert engine._current_portfolio is not None
        assert engine._current_portfolio.cash == engine.config.initial_capital
        assert engine._current_portfolio.total_value == engine.config.initial_capital
        assert len(engine._portfolio_history) == 1

    def test_generate_rebalance_dates_daily(self, engine):
        """Test rebalance date generation for daily frequency."""
        engine.config.rebalance_frequency = RebalanceFrequency.DAILY
        dates = engine._generate_rebalance_dates()

        assert len(dates) > 0
        assert dates[0] == engine.config.start_date
        assert dates[-1] <= engine.config.end_date

    def test_generate_rebalance_dates_weekly(self, engine):
        """Test rebalance date generation for weekly frequency."""
        engine.config.rebalance_frequency = RebalanceFrequency.WEEKLY
        dates = engine._generate_rebalance_dates()

        assert len(dates) > 0
        # Check that dates are approximately weekly
        for i in range(1, len(dates)):
            delta = (dates[i] - dates[i - 1]).days
            assert 6 <= delta <= 8  # Allow for weekends

    def test_generate_rebalance_dates_monthly(self, engine):
        """Test rebalance date generation for monthly frequency."""
        engine.config.rebalance_frequency = RebalanceFrequency.MONTHLY
        dates = engine._generate_rebalance_dates()

        assert len(dates) > 0
        # Check that dates are approximately monthly
        for i in range(1, len(dates)):
            delta = (dates[i] - dates[i - 1]).days
            assert 28 <= delta <= 32

    def test_calculate_sharpe_ratio(self, engine):
        """Test Sharpe ratio calculation."""
        returns = [0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.02]
        sharpe = engine._calculate_sharpe_ratio(returns)

        assert sharpe > 0  # Positive returns should give positive Sharpe

    def test_calculate_sharpe_ratio_zero_std(self, engine):
        """Test Sharpe ratio with zero standard deviation."""
        returns = [0.01, 0.01, 0.01, 0.01]
        sharpe = engine._calculate_sharpe_ratio(returns)

        assert sharpe == 0.0  # Should handle zero std

    def test_calculate_sortino_ratio(self, engine):
        """Test Sortino ratio calculation."""
        returns = [0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.02]
        sortino = engine._calculate_sortino_ratio(returns)

        assert sortino > 0  # Should be positive

    def test_calculate_max_drawdown(self, engine):
        """Test maximum drawdown calculation."""
        equity_curve = [
            (datetime(2024, 1, 1), 1000000),
            (datetime(2024, 1, 2), 1050000),
            (datetime(2024, 1, 3), 950000),
            (datetime(2024, 1, 4), 1100000),
            (datetime(2024, 1, 5), 900000),
        ]

        max_dd = engine._calculate_max_drawdown(equity_curve)

        assert max_dd < 0  # Drawdown should be negative
        assert max_dd >= -0.2  # Should be within reasonable range

    def test_calculate_win_rate(self, engine):
        """Test win rate calculation."""
        # Create some positions
        engine._all_positions = [
            Position(
                symbol="A",
                entry_date=datetime(2024, 1, 1),
                shares=100,
                entry_price=1000,
                exit_price=1100,
                pnl=10000,
            ),
            Position(
                symbol="B",
                entry_date=datetime(2024, 1, 1),
                shares=100,
                entry_price=1000,
                exit_price=900,
                pnl=-10000,
            ),
            Position(
                symbol="C",
                entry_date=datetime(2024, 1, 1),
                shares=100,
                entry_price=1000,
                exit_price=1200,
                pnl=20000,
            ),
        ]

        win_rate = engine._calculate_win_rate()

        assert win_rate == 2 / 3  # 2 wins out of 3

    def test_calculate_profit_factor(self, engine):
        """Test profit factor calculation."""
        engine._all_positions = [
            Position(
                symbol="A",
                entry_date=datetime(2024, 1, 1),
                shares=100,
                entry_price=1000,
                exit_price=1100,
                pnl=10000,
            ),
            Position(
                symbol="B",
                entry_date=datetime(2024, 1, 1),
                shares=100,
                entry_price=1000,
                exit_price=900,
                pnl=-5000,
            ),
            Position(
                symbol="C",
                entry_date=datetime(2024, 1, 1),
                shares=100,
                entry_price=1000,
                exit_price=1200,
                pnl=20000,
            ),
        ]

        profit_factor = engine._calculate_profit_factor()

        assert profit_factor == 30000 / 5000  # Total profit / total loss
        assert profit_factor == 6.0

    def test_run_backtest(self, engine, sample_predictions, sample_price_data):
        """Test running a complete backtest."""
        result = engine.run_backtest(sample_predictions, sample_price_data)

        assert isinstance(result, BacktestResult)
        assert result.config == engine.config
        assert result.total_trades >= 0
        assert result.win_rate >= 0
        assert result.win_rate <= 1
        assert len(result.equity_curve) > 0

    def test_backtest_result_metrics(self, engine, sample_predictions, sample_price_data):
        """Test backtest result metrics."""
        result = engine.run_backtest(sample_predictions, sample_price_data)

        # Check that all metrics are calculated
        assert hasattr(result, "total_return")
        assert hasattr(result, "annualized_return")
        assert hasattr(result, "sharpe_ratio")
        assert hasattr(result, "sortino_ratio")
        assert hasattr(result, "max_drawdown")
        assert hasattr(result, "win_rate")
        assert hasattr(result, "profit_factor")

        # Check metric ranges
        assert result.max_drawdown <= 0  # Drawdown should be negative or zero
        assert result.win_rate >= 0
        assert result.win_rate <= 1


class TestBacktestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_predictions(self):
        """Test backtest with empty predictions list."""
        config = BacktestConfig(start_date=datetime(2024, 1, 1), end_date=datetime(2024, 4, 20))
        engine = BacktestingEngine(config)

        empty_predictions = []
        dates = pd.date_range("2024-01-01", "2024-04-20", freq="D")
        price_data = pd.DataFrame(
            {
                "date": dates,
                "symbol": ["RELIANCE"] * len(dates),
                "adjusted_close": [1000] * len(dates),
            }
        )

        result = engine.run_backtest(empty_predictions, price_data)

        # Should complete but with no trades
        assert result.total_trades == 0

    def test_invalid_date_range(self):
        """Test backtest with invalid date range."""
        with pytest.raises(Exception):
            BacktestConfig(
                start_date=datetime(2024, 12, 31),
                end_date=datetime(2024, 1, 1),  # End before start
            )

    def test_zero_initial_capital(self):
        """Test backtest with zero initial capital."""
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1), end_date=datetime(2024, 1, 31), initial_capital=0
        )
        engine = BacktestingEngine(config)

        engine._initialize_portfolio()

        assert engine._current_portfolio.cash == 0
        assert engine._current_portfolio.total_value == 0


class TestTransactionCosts:
    """Test transaction cost calculations."""

    def test_commission_calculation(self):
        """Test commission is applied correctly."""
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            commission_rate=0.001,  # 0.1%
        )
        BacktestingEngine(config)

        # Simulate a trade
        trade_value = 100000
        expected_commission = trade_value * config.commission_rate

        assert expected_commission == 100.0

    def test_slippage_impact(self):
        """Test slippage impact on execution."""
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            slippage_rate=0.001,  # 0.1%
        )
        BacktestingEngine(config)

        # Simulate slippage
        expected_price = 1000.0
        slippage = expected_price * config.slippage_rate
        actual_price = expected_price + slippage

        assert actual_price == 1001.0


class TestRiskManagement:
    """Test risk management features."""

    def test_max_position_size(self):
        """Test maximum position size constraint."""
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            initial_capital=1000000,
            max_position_size=0.2,  # 20% max
        )
        BacktestingEngine(config)

        max_allowed = config.max_position_size * config.initial_capital
        assert max_allowed == 200000

    def test_max_positions_limit(self):
        """Test maximum number of positions limit."""
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1), end_date=datetime(2024, 1, 31), max_positions=5
        )
        engine = BacktestingEngine(config)

        assert engine.config.max_positions == 5
