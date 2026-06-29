from typing import Any

import pandas as pd
import vectorbt as vbt


class VectorBTBacktester:
    """
    Wraps VectorBT for highly vectorized, high-speed backtesting.
    Ideal for walk-forward analysis and massive parameter sweeps.
    """

    def __init__(self, init_cash: float = 100000.0, fees: float = 0.001, slippage: float = 0.0005):
        self.init_cash = init_cash
        self.fees = fees
        self.slippage = slippage

    def run_backtest(
        self,
        prices: pd.Series,
        entries: pd.Series,
        exits: pd.Series,
        short_entries: pd.Series = None,
        short_exits: pd.Series = None,
        freq: str = "1D",
    ) -> vbt.Portfolio:
        """
        Runs a vectorized backtest on boolean entry/exit signals.
        Returns a vbt Portfolio object which contains tearsheets and metrics.

        Args:
            freq: Data frequency string (e.g., '1D', '1H', '5T'). Default is '1D'.
        """
        portfolio = vbt.Portfolio.from_signals(
            close=prices,
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            init_cash=self.init_cash,
            fees=self.fees,
            slippage=self.slippage,
            freq=freq,
        )
        return portfolio

    def get_metrics(self, portfolio: vbt.Portfolio) -> dict[str, Any]:
        """Extract key research metrics from the backtest."""
        return {
            "Total Return [%]": portfolio.total_return() * 100,
            "Sharpe Ratio": portfolio.sharpe_ratio(),
            "Max Drawdown [%]": portfolio.max_drawdown() * 100,
            "Win Rate [%]": portfolio.trades.win_rate() * 100,
            "Expectancy": portfolio.trades.expectancy(),
        }

    def run_walk_forward(
        self,
        prices: pd.Series,
        entries: pd.Series,
        exits: pd.Series,
        train_size: int = 252,
        test_size: int = 63,
        short_entries: pd.Series = None,
        short_exits: pd.Series = None,
        freq: str = "1D",
    ) -> list[dict[str, Any]]:
        """
        Walk-forward analysis: splits data into sequential train/test windows
        and runs a backtest on each test window.

        Args:
            prices: Price series for the full period.
            entries: Entry signals for the full period.
            exits: Exit signals for the full period.
            train_size: Number of bars in each training window.
            test_size: Number of bars in each testing window.
            short_entries: Optional short entry signals.
            short_exits: Optional short exit signals.
            freq: Data frequency string.

        Returns:
            A list of dicts, each containing 'window_index', 'train_range',
            'test_range', 'portfolio', and 'metrics' for that fold.
        """
        results: list[dict[str, Any]] = []
        total_bars = len(prices)
        window_index = 0

        start = 0
        while start + train_size + test_size <= total_bars:
            train_end = start + train_size
            test_end = train_end + test_size

            # Slice test window
            test_prices = prices.iloc[train_end:test_end]
            test_entries = entries.iloc[train_end:test_end]
            test_exits = exits.iloc[train_end:test_end]
            test_short_entries = (
                short_entries.iloc[train_end:test_end] if short_entries is not None else None
            )
            test_short_exits = (
                short_exits.iloc[train_end:test_end] if short_exits is not None else None
            )

            portfolio = self.run_backtest(
                prices=test_prices,
                entries=test_entries,
                exits=test_exits,
                short_entries=test_short_entries,
                short_exits=test_short_exits,
                freq=freq,
            )

            results.append(
                {
                    "window_index": window_index,
                    "train_range": (prices.index[start], prices.index[train_end - 1]),
                    "test_range": (prices.index[train_end], prices.index[test_end - 1]),
                    "portfolio": portfolio,
                    "metrics": self.get_metrics(portfolio),
                }
            )

            start += test_size  # Roll forward by test_size
            window_index += 1

        return results
