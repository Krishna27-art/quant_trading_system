import os
import sys

import pandas as pd
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_platform.feature_store.macro import extract_historical_macro
from prediction_intelligence.base_logistic import build_features as canonical_build_features
from prediction_intelligence.base_logistic import build_label, SWING_FEATURES
from prediction_intelligence.meta_ensemble import MetaEnsemble
from prediction_intelligence.signal_adapter import SignalPrediction
from research_platform.backtesting.engine import (
    BacktestConfig,
    BacktestingEngine,
    RebalanceFrequency,
)
from utils.logger import get_logger

logger = get_logger("run_daytrading_backtest")


def load_data():
    logger.info("Loading Parquet data from data lake...")
    df = pd.read_parquet("data/bronze/equity_history/equity_history.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = df["close"].astype(float)
    df["adjusted_close"] = df["close"]  # required by backtest engine

    symbols = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY"]
    df = df[df["symbol"].isin(symbols)].copy()

    # Sort chronologically
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df


def feature_engineering(df):
    logger.info("Generating Machine Learning features...")
    # Generate canonical features
    timestamps = df["date"].unique()
    macro_df = extract_historical_macro(pd.DatetimeIndex(timestamps))
    df_merged = df.merge(macro_df, left_on="date", right_index=True, how="left")
    
    all_feats = []
    for sym, group in df_merged.groupby("symbol"):
        feats = canonical_build_features(group, timeframe="SWING")
        feats["symbol"] = sym
        feats["date"] = group["date"].values
        feats["close"] = group["close"].values
        feats["adjusted_close"] = group["adjusted_close"].values
        feats["high"] = group["high"].values
        feats["low"] = group["low"].values
        feats["open"] = group["open"].values
        feats["volume"] = group["volume"].values
        all_feats.append(feats)
        
    df_feats = pd.concat(all_feats, ignore_index=True)
    
    # Calculate ATR (Average True Range) for volatility-based stop/target sizing
    df_feats["tr"] = df_feats["high"] - df_feats["low"]
    df_feats["atr_14d"] = df_feats.groupby("symbol")["tr"].rolling(14).mean().reset_index(0, drop=True)
    df_feats["atr_pct"] = df_feats["atr_14d"] / df_feats["close"]
    
    return df_feats.dropna()


def run():
    df = load_data()
    df = feature_engineering(df)

    # Split into Train (First half of 2024) and Test (Rest)
    split_date = pd.to_datetime("2024-07-01")
    train_df = df[df["date"] < split_date].copy()
    test_df = df[df["date"] >= split_date].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        logger.error("Insufficient data for split.")
        return

    logger.info(f"Training set: {len(train_df)} rows. Test set: {len(test_df)} rows.")

    # Build labels for training
    all_train_with_labels = []
    for sym, group in train_df.groupby("symbol"):
        orig_group = df[(df["symbol"] == sym) & (df["date"] >= group["date"].min()) & (df["date"] <= group["date"].max())].sort_values("date")
        label_long = build_label(orig_group.set_index("date", drop=False), "SWING", side="long")
        
        group = group.copy()
        group["__label__"] = label_long.reindex(group["date"]).values
        all_train_with_labels.append(group)
        
    train_df_labeled = pd.concat(all_train_with_labels, ignore_index=True).dropna()

    # Train production MetaEnsemble model
    model = MetaEnsemble(timeframe="SWING", feature_cols=SWING_FEATURES)
    X_train = train_df_labeled[SWING_FEATURES]
    y_train = train_df_labeled["__label__"].astype(int)

    logger.info("Training MetaEnsemble model...")
    model.fit(X_train, y_train)

    # Predict on test set
    logger.info("Generating out-of-sample predictions...")
    X_test = test_df[SWING_FEATURES]
    proba_long = model.predict_proba(X_test)[:, 1]

    # Re-attach columns
    test_df = test_df.reset_index(drop=True)
    test_df["win_probability_long"] = proba_long

    # Generate Trading Signals for the Engine
    # Track currently held positions to emit explicit sell signals when they drop from top-N
    logger.info("Generating executable signals for backtesting engine...")
    signals = []
    current_positions = set()  # Symbols currently held

    # Sort by date to ensure chronological signal generation
    test_df = test_df.sort_values("date")

    for _date, group in test_df.groupby("date"):
        top_picks = group.nlargest(2, "win_probability_long")
        top_symbols = set(top_picks["symbol"].values)

        # Emit sell signals (prediction=0) for symbols that dropped from top-N
        for symbol in current_positions:
            if symbol not in top_symbols:
                # Find the row for this symbol on this date
                symbol_row = group[group["symbol"] == symbol]
                if not symbol_row.empty:
                    row = symbol_row.iloc[0]
                    signals.append(SignalPrediction(
                        date=row["date"],
                        symbol=row["symbol"],
                        prediction=0,
                        confidence=float(row["win_probability_long"]),
                        win_probability=float(row["win_probability_long"]),
                    ))

        # Emit buy signals (prediction=2) for new top picks
        for _, row in top_picks.iterrows():
            if row["win_probability_long"] > 0.50:  # Only take trades with positive expected value
                signals.append(SignalPrediction(
                    date=row["date"],
                    symbol=row["symbol"],
                    prediction=2,
                    confidence=float(row["win_probability_long"]),
                    win_probability=float(row["win_probability_long"]),
                ))

        # Update current positions
        current_positions = top_symbols

    # Configure the Backtesting Engine with ATR-based stop/target sizing
    # Use median ATR as baseline for dynamic sizing
    median_atr_pct = test_df["atr_pct"].median()
    atr_stop_loss = -2.0 * median_atr_pct  # 2x ATR as stop loss
    atr_take_profit = 1.5 * median_atr_pct  # 1.5x ATR as take profit

    logger.info(f"ATR-based sizing: median ATR={median_atr_pct:.2%}, stop={atr_stop_loss:.2%}, target={atr_take_profit:.2%}")

    config = BacktestConfig(
        start_date=test_df["date"].min().to_pydatetime(),
        end_date=test_df["date"].max().to_pydatetime(),
        initial_capital=10_000_000,
        position_size=0.10,
        rebalance_frequency=RebalanceFrequency.DAILY,
        commission_rate=0.0003,
        slippage_rate=0.0005,
        stop_loss=atr_stop_loss,  # ATR-based stop loss
        take_profit=atr_take_profit,  # ATR-based take profit
    )

    logger.info("Starting Backtesting Engine...")
    engine = BacktestingEngine(config)

    # Price data formatted for engine
    price_data = df[["date", "symbol", "adjusted_close", "close"]].copy()

    results = engine.run_backtest(predictions=signals, price_data=price_data)

    print("\n" + "=" * 50)
    print("BACKTEST RESULTS (Out-of-Sample)")
    print("=" * 50)
    print(f"Total Return:       {results.total_return:.2%}")
    print(f"Annualized Return:  {results.annualized_return:.2%}")
    print(f"Win Rate:           {results.win_rate:.2%}")
    print(f"Sharpe Ratio:       {results.sharpe_ratio:.2f}")
    if results.deflated_sharpe is not None:
        print(f"Deflated Sharpe:    {results.deflated_sharpe:.2f}")
    print(f"Max Drawdown:       {results.max_drawdown:.2%}")
    print(f"Total Trades:       {results.total_trades}")
    print("=" * 50)


if __name__ == "__main__":
    run()
