"""
Walk-Forward Backtest

Implements walk-forward validation (rolling window backtesting) instead of
a single train/test split. This provides more robust performance estimates by
testing the strategy across multiple non-overlapping periods.

Walk-forward process:
1. Train on initial window (e.g., first 6 months)
2. Test on subsequent window (e.g., next 3 months)
3. Roll forward: train on next window, test on next window
4. Aggregate results across all test periods
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prediction_intelligence.lightgbm_ranker import LightGBMRankerModel
from prediction_intelligence.signal_adapter import SignalPrediction, from_lightgbm_ranker_output
from research_platform.backtesting.engine import (
    BacktestConfig,
    BacktestingEngine,
    RebalanceFrequency,
)
from utils.logger import get_logger

logger = get_logger("walkforward_backtest")


def load_data():
    logger.info("Loading Parquet data from data lake...")
    df = pd.read_parquet("data/bronze/equity_history/equity_history.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = df["close"].astype(float)
    df["adjusted_close"] = df["close"]  # required by backtest engine

    symbols = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY"]
    df = df[df["symbol"].isin(symbols)].copy()

    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df


def feature_engineering(df):
    logger.info("Generating Machine Learning features...")
    df["returns_1d"] = df.groupby("symbol")["close"].pct_change(1)
    df["momentum_5d"] = df.groupby("symbol")["close"].pct_change(5)
    df["momentum_10d"] = df.groupby("symbol")["close"].pct_change(10)
    df["volatility_10d"] = (
        df.groupby("symbol")["returns_1d"].rolling(10).std().reset_index(0, drop=True)
    )

    # Calculate ATR for volatility-based stop/target sizing
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = (df["high"] - df["close"]).abs()
    df["low_close"] = (df["low"] - df["close"]).abs()
    df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)
    df["atr_14d"] = df.groupby("symbol")["tr"].rolling(14).mean().reset_index(0, drop=True)
    df["atr_pct"] = df["atr_14d"] / df["close"]

    df["fwd_return"] = df.groupby("symbol")["returns_1d"].shift(-1)
    df = df.dropna()
    return df


def generate_signals(test_df: pd.DataFrame) -> list[SignalPrediction]:
    """Generate trading signals from test predictions."""
    signals = []
    current_positions = set()
    test_df = test_df.sort_values("date")

    for _date, group in test_df.groupby("date"):
        top_picks = group.nlargest(2, "alpha_score")
        top_symbols = set(top_picks["symbol"].values)

        # Emit sell signals for symbols that dropped from top-N
        for symbol in current_positions:
            if symbol not in top_symbols:
                symbol_row = group[group["symbol"] == symbol]
                if not symbol_row.empty:
                    row = symbol_row.iloc[0]
                    sell_signal = from_lightgbm_ranker_output(row.to_dict(), row["date"])
                    sell_signal.prediction = 0
                    signals.append(sell_signal)

        # Emit buy signals for new top picks
        for _, row in top_picks.iterrows():
            if row["win_probability"] > 0.50:
                buy_signal = from_lightgbm_ranker_output(row.to_dict(), row["date"])
                buy_signal.prediction = 2
                signals.append(buy_signal)

        current_positions = top_symbols

    return signals


def run_walk_forward(
    df: pd.DataFrame,
    train_window_days: int = 180,  # 6 months training
    test_window_days: int = 90,  # 3 months testing
    step_days: int = 90,  # Roll forward by 3 months
):
    """
    Run walk-forward backtest.

    Args:
        df: Feature-engineered dataframe
        train_window_days: Training window size in days
        test_window_days: Testing window size in days
        step_days: Step size for rolling forward
    """
    features = ["returns_1d", "momentum_5d", "momentum_10d", "volatility_10d"]

    # Get date range
    min_date = df["date"].min()
    max_date = df["date"].max()

    logger.info(f"Date range: {min_date} to {max_date}")
    logger.info(f"Walk-forward config: train={train_window_days}d, test={test_window_days}d, step={step_days}d")

    # Generate walk-forward windows
    windows = []
    current_train_start = min_date

    while True:
        train_end = current_train_start + timedelta(days=train_window_days)
        test_start = train_end
        test_end = test_start + timedelta(days=test_window_days)

        if test_end > max_date:
            break

        windows.append({
            "train_start": current_train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })

        current_train_start += timedelta(days=step_days)

    logger.info(f"Generated {len(windows)} walk-forward windows")

    # Run backtest for each window
    all_results = []

    for i, window in enumerate(windows):
        logger.info(f"\n{'='*60}")
        logger.info(f"Window {i+1}/{len(windows)}")
        logger.info(f"Train: {window['train_start']} to {window['train_end']}")
        logger.info(f"Test:  {window['test_start']} to {window['test_end']}")
        logger.info(f"{'='*60}")

        # Split data
        train_df = df[
            (df["date"] >= window["train_start"]) & (df["date"] < window["train_end"])
        ].copy()
        test_df = df[
            (df["date"] >= window["test_start"]) & (df["date"] < window["test_end"])
        ].copy()

        if len(train_df) < 100 or len(test_df) < 50:
            logger.warning(f"Insufficient data for window {i+1}, skipping")
            continue

        logger.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

        # Train model
        model = LightGBMRankerModel()
        train_df_indexed = train_df.set_index(["date", "symbol"])

        try:
            model.train(
                features_df=train_df_indexed[features],
                labels_df=train_df_indexed["fwd_return"],
                feature_columns=features,
                params=None,
            )

            # Predict on test set
            test_df_indexed = test_df.set_index(["date", "symbol"])
            preds_df = model.predict(test_df_indexed[features])

            test_df = test_df.reset_index()
            test_df["alpha_score"] = preds_df["alpha_score"].values
            test_df["win_probability"] = preds_df["win_probability"].values

            # Generate signals
            signals = generate_signals(test_df)

            # ATR-based stop/target sizing
            median_atr_pct = test_df["atr_pct"].median()
            atr_stop_loss = -2.0 * median_atr_pct
            atr_take_profit = 1.5 * median_atr_pct

            # Run backtest
            config = BacktestConfig(
                start_date=window["test_start"],
                end_date=window["test_end"],
                initial_capital=10_000_000,
                position_size=0.10,
                rebalance_frequency=RebalanceFrequency.DAILY,
                commission_rate=0.0003,
                slippage_rate=0.0005,
                stop_loss=atr_stop_loss,
                take_profit=atr_take_profit,
            )

            engine = BacktestingEngine(config)
            price_data = df[["date", "symbol", "adjusted_close", "close"]].copy()
            results = engine.run_backtest(predictions=signals, price_data=price_data)

            all_results.append({
                "window": i + 1,
                "train_start": window["train_start"],
                "test_start": window["test_start"],
                "total_return": results.total_return,
                "annualized_return": results.annualized_return,
                "win_rate": results.win_rate,
                "sharpe_ratio": results.sharpe_ratio,
                "max_drawdown": results.max_drawdown,
                "total_trades": results.total_trades,
            })

            logger.info(f"Window {i+1} results: Return={results.total_return:.2%}, "
                       f"WinRate={results.win_rate:.2%}, Sharpe={results.sharpe_ratio:.2f}")

        except Exception as e:
            logger.error(f"Window {i+1} failed: {e}")
            continue

    # Aggregate results
    if not all_results:
        logger.error("No windows completed successfully")
        return

    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 70)
    print("WALK-FORWARD BACKTEST RESULTS")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("\n" + "=" * 70)
    print("AGGREGATED STATISTICS")
    print("=" * 70)

    print(f"Total Windows: {len(results_df)}")
    print(f"Win Rate (mean): {results_df['win_rate'].mean():.2%} ± {results_df['win_rate'].std():.2%}")
    print(f"Sharpe Ratio (mean): {results_df['sharpe_ratio'].mean():.2f} ± {results_df['sharpe_ratio'].std():.2f}")
    print(f"Total Return (mean): {results_df['total_return'].mean():.2%} ± {results_df['total_return'].std():.2%}")
    print(f"Max Drawdown (mean): {results_df['max_drawdown'].mean():.2%} ± {results_df['max_drawdown'].std():.2%}")
    print(f"Total Trades (mean): {results_df['total_trades'].mean():.0f}")

    # Consistency check: how many windows were profitable?
    profitable_windows = (results_df['total_return'] > 0).sum()
    print(f"\nProfitable Windows: {profitable_windows}/{len(results_df)} ({profitable_windows/len(results_df):.1%})")

    # Consistency check: how many windows had positive Sharpe?
    positive_sharpe_windows = (results_df['sharpe_ratio'] > 0).sum()
    print(f"Positive Sharpe Windows: {positive_sharpe_windows}/{len(results_df)} ({positive_sharpe_windows/len(results_df):.1%})")

    print("=" * 70)


if __name__ == "__main__":
    df = load_data()
    df = feature_engineering(df)

    run_walk_forward(df)
