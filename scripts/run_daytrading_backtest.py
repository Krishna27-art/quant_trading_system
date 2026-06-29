import os
import sys
from datetime import datetime

import pandas as pd

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pydantic import BaseModel

from prediction_intelligence.lightgbm_ranker import LightGBMRankerModel
from research_platform.backtesting.engine import (
    BacktestConfig,
    BacktestingEngine,
    RebalanceFrequency,
)
from utils.logger import get_logger

logger = get_logger("run_daytrading_backtest")


class SignalPrediction(BaseModel):
    date: datetime
    symbol: str
    prediction: int
    confidence: float
    win_probability: float


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
    df["returns_1d"] = df.groupby("symbol")["close"].pct_change(1)
    df["momentum_5d"] = df.groupby("symbol")["close"].pct_change(5)
    df["momentum_10d"] = df.groupby("symbol")["close"].pct_change(10)
    df["volatility_10d"] = (
        df.groupby("symbol")["returns_1d"].rolling(10).std().reset_index(0, drop=True)
    )

    # Target label: forward 1-day return
    df["fwd_return"] = df.groupby("symbol")["returns_1d"].shift(-1)

    # Rank labels for LGBM
    df["rank_label"] = df.groupby("date")["fwd_return"].rank(pct=True, ascending=True)
    df["rank_label"] = pd.qcut(df["rank_label"], q=5, labels=False, duplicates="drop")

    df = df.dropna()
    return df


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

    features = ["returns_1d", "momentum_5d", "momentum_10d", "volatility_10d"]

    model = LightGBMRankerModel()

    # Set up index for the ranker model train function
    # It expects multiindex (date, symbol)
    train_df.set_index(["date", "symbol"], inplace=True)

    # Train the model and calibrate Win %
    logger.info("Training and calibrating LightGBM Ranker...")
    model.train(
        features_df=train_df[features],
        labels_df=train_df["fwd_return"],  # Needed for binary wins
        feature_columns=features,
        params=None,
    )

    # Predict on test set
    test_df.set_index(["date", "symbol"], inplace=True)
    logger.info("Generating out-of-sample predictions...")
    preds_df = model.predict(test_df[features])

    # Re-attach columns
    test_df = test_df.reset_index()
    test_df["alpha_score"] = preds_df["alpha_score"].values
    test_df["win_probability"] = preds_df["win_probability"].values

    # Generate Trading Signals for the Engine
    # Prediction 2 implies "Buy"
    logger.info("Generating executable signals for backtesting engine...")
    signals = []

    # Select top symbols each day as buys
    for _date, group in test_df.groupby("date"):
        top_picks = group.nlargest(2, "alpha_score")
        for _, row in top_picks.iterrows():
            if row["win_probability"] > 0.50:  # Only take trades with positive expected value
                signals.append(
                    SignalPrediction(
                        date=row["date"],
                        symbol=row["symbol"],
                        prediction=2,
                        confidence=row["alpha_score"],
                        win_probability=row["win_probability"],
                    )
                )

    # Configure the Backtesting Engine
    config = BacktestConfig(
        start_date=test_df["date"].min().to_pydatetime(),
        end_date=test_df["date"].max().to_pydatetime(),
        initial_capital=10_000_000,
        position_size=0.10,
        rebalance_frequency=RebalanceFrequency.DAILY,
        commission_rate=0.0003,
        slippage_rate=0.0005,
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
