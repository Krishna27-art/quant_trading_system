import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from research_platform.backtesting.engine import (
    BacktestConfig,
    BacktestingEngine,
    RebalanceFrequency,
)
from prediction_intelligence.signal_adapter import SignalPrediction
from utils.logger import get_logger

logger = get_logger("run_orb_backtest")

def load_data():
    logger.info("Generating mock data...")
    dates = pd.date_range(start="2024-01-01", periods=200, freq="B")
    symbols = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY"]
    
    records = []
    for sym in symbols:
        price = 1000.0
        for date in dates:
            ret = np.random.normal(0.0005, 0.015)
            price *= (1 + ret)
            records.append({
                "date": date,
                "symbol": sym,
                "close": price,
                "adjusted_close": price
            })
    
    df = pd.DataFrame(records)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df

def generate_orb_signals(df):
    logger.info("Generating ORB signals...")
    
    # We only have daily data, so we'll use a momentum proxy for ORB breakout
    df["returns_1d"] = df.groupby("symbol")["close"].pct_change(1)
    df = df.dropna()
    
    signals = []
    for _, row in df.iterrows():
        if row["returns_1d"] > 0.01:
            sig = SignalPrediction(
                symbol=row["symbol"],
                date=row["date"],
                prediction=2, # Buy
                confidence=0.8,
                win_probability=0.55
            )
            signals.append(sig)
        elif row["returns_1d"] < -0.01:
             sig = SignalPrediction(
                symbol=row["symbol"],
                date=row["date"],
                prediction=0, # Sell
                confidence=0.8,
                win_probability=0.55
            )
             signals.append(sig)
             
    return signals, df

def run():
    df = load_data()
    signals, df = generate_orb_signals(df)
    
    config = BacktestConfig(
        start_date=df["date"].min().to_pydatetime(),
        end_date=df["date"].max().to_pydatetime(),
        initial_capital=10_000_000,
        position_size=0.10,
        rebalance_frequency=RebalanceFrequency.DAILY,
        commission_rate=0.0003,
        slippage_rate=0.0005,
        stop_loss=-0.02,
        take_profit=0.05,
    )
    
    logger.info("Starting Backtesting Engine for ORB strategy...")
    engine = BacktestingEngine(config)
    price_data = df[["date", "symbol", "adjusted_close", "close"]].copy()
    
    results = engine.run_backtest(predictions=signals, price_data=price_data)
    
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS (ORB Proxy)")
    print("=" * 50)
    print(f"Total Return:       {results.total_return:.2%}")
    print(f"Annualized Return:  {results.annualized_return:.2%}")
    print(f"Win Rate:           {results.win_rate:.2%}")
    print(f"Total Trades:       {results.total_trades}")
    print("-" * 50)
    print(f"Validation Passed:  {results.validation_is_valid}")
    if results.validation_p_value is not None:
        print(f"Validation p-value: {results.validation_p_value:.4f}")
    if results.validation_warning is not None:
        print(f"Validation Warning: {results.validation_warning}")
    print(f"Leakage Detected:   {results.leakage_detected}")
    if results.leakage_reason is not None:
        print(f"Leakage Reason:     {results.leakage_reason}")
    print("=" * 50)

if __name__ == "__main__":
    run()
