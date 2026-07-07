"""
Walk-Forward Backtest

Implements walk-forward validation (rolling window backtesting) using the active
production MetaEnsemble model class and canonical feature engineering pipeline.
"""

import os
import sys
from datetime import timedelta

import pandas as pd

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

logger = get_logger("walkforward_backtest")


def load_data():
    symbols = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY"]
    import yfinance as yf
    all_dfs = []
    for sym in symbols:
        logger.info(f"Downloading daily data for {sym}...")
        raw = yf.download(f"{sym}.NS", period="3y", progress=False, auto_adjust=True)
        if raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.reset_index().rename(columns={
            "Date": "date", "Open": "open",
            "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df["symbol"] = sym
        df["adjusted_close"] = df["close"]
        all_dfs.append(df)
    if not all_dfs:
        raise ValueError("No data downloaded from yfinance!")
    df = pd.concat(all_dfs, ignore_index=True)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df


def feature_engineering(df):
    logger.info("Generating canonical Machine Learning features...")
    # Calculate ATR for portfolio risk/stop sizing later
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = (df["high"] - df["close"]).abs()
    df["low_close"] = (df["low"] - df["close"]).abs()
    df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)
    df["atr_14d"] = df.groupby("symbol")["tr"].rolling(14).mean().reset_index(0, drop=True)
    df["atr_pct"] = df["atr_14d"] / df["close"]

    df["volume_sma20"] = df.groupby("symbol")["volume"].transform(
        lambda x: x.rolling(20, min_periods=1).mean()
    )

    all_feats = []
    for sym, group in df.groupby("symbol"):
        group = group.sort_values("date")
        timestamps = pd.DatetimeIndex(group["date"])
        macro_df = extract_historical_macro(timestamps)
        extra_data = {col: macro_df[col] for col in macro_df.columns}
        
        feats = canonical_build_features(group.set_index("date", drop=False), "SWING", extra=extra_data)
        feats["symbol"] = sym
        feats["date"] = group["date"].values
        feats["close"] = group["close"].values
        feats["adjusted_close"] = group["close"].values
        feats["atr_pct"] = group["atr_pct"].values
        feats["volume"] = group["volume"].values
        feats["volume_sma20"] = group["volume_sma20"].values
        all_feats.append(feats)
        
    out_df = pd.concat(all_feats, ignore_index=True)
    out_df = out_df.dropna().reset_index(drop=True)
    return out_df


def generate_signals(test_df: pd.DataFrame) -> list[SignalPrediction]:
    """Generate trading signals from test predictions."""
    signals = []
    current_positions = set()
    test_df = test_df.sort_values("date")

    for date_val, group in test_df.groupby("date"):
        # Select top long picks based on win probability
        top_picks = group.nlargest(2, "win_probability_long")
        top_symbols = set(top_picks["symbol"].values)

        # Emit sell signals for symbols that dropped from top-N
        for symbol in current_positions:
            if symbol not in top_symbols:
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

        # Emit buy signals for new top picks
        for _, row in top_picks.iterrows():
            signals.append(SignalPrediction(
                date=row["date"],
                symbol=row["symbol"],
                prediction=2,
                confidence=float(row["win_probability_long"]),
                win_probability=float(row["win_probability_long"]),
            ))

        current_positions = top_symbols

    return signals


def generate_buy_and_hold_signals(test_df: pd.DataFrame) -> list[SignalPrediction]:
    """Buy and hold all symbols at the start date."""
    signals = []
    test_df = test_df.sort_values("date")
    start_date = test_df["date"].min()
    
    start_group = test_df[test_df["date"] == start_date]
    for _, row in start_group.iterrows():
        signals.append(SignalPrediction(
            date=row["date"],
            symbol=row["symbol"],
            prediction=2,
            confidence=1.0,
            win_probability=1.0,
        ))
    return signals


def generate_rsi_reversal_signals(test_df: pd.DataFrame) -> list[SignalPrediction]:
    """Buy when RSI < 35, sell/exit when RSI > 65."""
    signals = []
    current_positions = set()
    test_df = test_df.sort_values("date")
    
    for date_val, group in test_df.groupby("date"):
        # Check current positions for exits
        for sym in list(current_positions):
            sym_row = group[group["symbol"] == sym]
            if not sym_row.empty:
                row = sym_row.iloc[0]
                # Exit if RSI > 65
                if row.get("rsi_14d", 50) > 65:
                    signals.append(SignalPrediction(
                        date=row["date"],
                        symbol=row["symbol"],
                        prediction=0,
                        confidence=1.0,
                        win_probability=1.0,
                    ))
                    current_positions.remove(sym)
        
        # Check for new entries
        for _, row in group.iterrows():
            sym = row["symbol"]
            if sym not in current_positions:
                # Buy if RSI < 35
                if row.get("rsi_14d", 50) < 35:
                    signals.append(SignalPrediction(
                        date=row["date"],
                        symbol=row["symbol"],
                        prediction=2,
                        confidence=1.0,
                        win_probability=1.0,
                    ))
                    current_positions.add(sym)
                    
    return signals


def run_walk_forward(
    df: pd.DataFrame,
    train_window_days: int = 365,  # 1 year training
    test_window_days: int = 180,  # 6 months testing
    step_days: int = 180,  # Roll forward by 6 months
):
    """
    Run walk-forward backtest.
    """
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

        # Build labels for training
        all_train_with_labels = []
        for sym, group in train_df.groupby("symbol"):
            orig_group = df[(df["symbol"] == sym) & (df["date"] >= group["date"].min()) & (df["date"] <= group["date"].max())].sort_values("date")
            label_long = build_label(orig_group.set_index("date", drop=False), "SWING", side="long")
            
            group = group.copy()
            group["__label__"] = label_long.reindex(group["date"]).values
            all_train_with_labels.append(group)
            
        train_df_labeled = pd.concat(all_train_with_labels, ignore_index=True).dropna()

        if len(train_df_labeled) < 100 or len(test_df) < 50:
            logger.warning(f"Insufficient data for window {i+1}, skipping")
            continue

        logger.info(f"Train samples: {len(train_df_labeled)}, Test samples: {len(test_df)}")

        # Train model
        model_long = MetaEnsemble(timeframe="SWING", feature_cols=SWING_FEATURES)
        X_train = train_df_labeled[SWING_FEATURES]
        y_train = train_df_labeled["__label__"].astype(int)

        try:
            model_long.fit(X_train, y_train)

            # Predict on test set
            X_test = test_df[SWING_FEATURES]
            proba_long = model_long.predict_proba(X_test)[:, 1]

            test_df = test_df.copy()
            test_df["win_probability_long"] = proba_long

            logger.info(f"Win Probability Stats: min={proba_long.min():.4f}, max={proba_long.max():.4f}, mean={proba_long.mean():.4f}")

            # Generate signals
            signals = generate_signals(test_df)
            logger.info(f"Generated {len(signals)} signals for test period")

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
                max_adv_participation_rate=0.01,
            )

            engine = BacktestingEngine(config)
            price_data = df[["date", "symbol", "adjusted_close", "close", "volume", "volume_sma20"]].copy()
            results = engine.run_backtest(predictions=signals, price_data=price_data)

            # Run benchmarks
            bh_signals = generate_buy_and_hold_signals(test_df)
            rsi_signals = generate_rsi_reversal_signals(test_df)

            engine_bh = BacktestingEngine(config)
            results_bh = engine_bh.run_backtest(predictions=bh_signals, price_data=price_data)

            engine_rsi = BacktestingEngine(config)
            results_rsi = engine_rsi.run_backtest(predictions=rsi_signals, price_data=price_data)

            all_results.append({
                "window": i + 1,
                "train_start": window["train_start"],
                "test_start": window["test_start"],
                "strategy": "MetaEnsemble",
                "net_return": results.total_return,
                "net_sharpe": results.sharpe_ratio,
                "win_rate": results.win_rate,
                "max_drawdown": results.max_drawdown,
                "max_drawdown_duration": results.max_drawdown_duration if results.max_drawdown_duration is not None else 0,
                "cagr": results.cagr if results.cagr is not None else results.annualized_return,
                "volatility": results.volatility if results.volatility is not None else 0.0,
                "calmar": results.calmar_ratio if results.calmar_ratio is not None else 0.0,
                "avg_drawdown": results.avg_drawdown if results.avg_drawdown is not None else 0.0,
                "worst_trade": results.worst_trade if results.worst_trade is not None else 0.0,
                "longest_losing_streak": results.longest_losing_streak if results.longest_losing_streak is not None else 0,
                "total_trades": results.total_trades,
            })

            all_results.append({
                "window": i + 1,
                "train_start": window["train_start"],
                "test_start": window["test_start"],
                "strategy": "BuyAndHold",
                "net_return": results_bh.total_return,
                "net_sharpe": results_bh.sharpe_ratio,
                "win_rate": results_bh.win_rate,
                "max_drawdown": results_bh.max_drawdown,
                "max_drawdown_duration": results_bh.max_drawdown_duration if results_bh.max_drawdown_duration is not None else 0,
                "cagr": results_bh.cagr if results_bh.cagr is not None else results_bh.annualized_return,
                "volatility": results_bh.volatility if results_bh.volatility is not None else 0.0,
                "calmar": results_bh.calmar_ratio if results_bh.calmar_ratio is not None else 0.0,
                "avg_drawdown": results_bh.avg_drawdown if results_bh.avg_drawdown is not None else 0.0,
                "worst_trade": results_bh.worst_trade if results_bh.worst_trade is not None else 0.0,
                "longest_losing_streak": results_bh.longest_losing_streak if results_bh.longest_losing_streak is not None else 0,
                "total_trades": results_bh.total_trades,
            })

            all_results.append({
                "window": i + 1,
                "train_start": window["train_start"],
                "test_start": window["test_start"],
                "strategy": "RSIReversal",
                "net_return": results_rsi.total_return,
                "net_sharpe": results_rsi.sharpe_ratio,
                "win_rate": results_rsi.win_rate,
                "max_drawdown": results_rsi.max_drawdown,
                "max_drawdown_duration": results_rsi.max_drawdown_duration if results_rsi.max_drawdown_duration is not None else 0,
                "cagr": results_rsi.cagr if results_rsi.cagr is not None else results_rsi.annualized_return,
                "volatility": results_rsi.volatility if results_rsi.volatility is not None else 0.0,
                "calmar": results_rsi.calmar_ratio if results_rsi.calmar_ratio is not None else 0.0,
                "avg_drawdown": results_rsi.avg_drawdown if results_rsi.avg_drawdown is not None else 0.0,
                "worst_trade": results_rsi.worst_trade if results_rsi.worst_trade is not None else 0.0,
                "longest_losing_streak": results_rsi.longest_losing_streak if results_rsi.longest_losing_streak is not None else 0,
                "total_trades": results_rsi.total_trades,
            })

            logger.info(f"Window {i+1} results: Strategy={results.total_return:.2%}, "
                        f"Buy&Hold={results_bh.total_return:.2%}, RSIReversal={results_rsi.total_return:.2%}")

        except Exception as e:
            logger.error(f"Window {i+1} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Aggregate results
    if not all_results:
        logger.error("No windows completed successfully")
        return

    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 120)
    print("WALK-FORWARD BACKTEST RESULTS — STRATEGY COMPARISON")
    print("=" * 120)
    print(results_df.to_string(index=False))
    print("\n" + "=" * 120)
    print("AGGREGATED STATISTICS (NET OF COMMISSIONS, STT, SLIPPAGE, AND ADV CAPPED)")
    print("=" * 120)

    for strat, strat_df in results_df.groupby("strategy"):
        print(f"\n>>> Strategy: {strat} <<<")
        print(f"  Total Windows: {len(strat_df)}")
        print(f"  Win Rate (mean): {strat_df['win_rate'].mean():.2%} ± {strat_df['win_rate'].std():.2%}")
        print(f"  Net Sharpe Ratio (mean): {strat_df['net_sharpe'].mean():.2f} ± {strat_df['net_sharpe'].std():.2f}")
        print(f"  Net Return (mean): {strat_df['net_return'].mean():.2%} ± {strat_df['net_return'].std():.2%}")
        print(f"  CAGR (mean): {strat_df['cagr'].mean():.2%} ± {strat_df['cagr'].std():.2%}")
        print(f"  Annualized Volatility (mean): {strat_df['volatility'].mean():.2%} ± {strat_df['volatility'].std():.2%}")
        print(f"  Calmar Ratio (mean): {strat_df['calmar'].mean():.2f} ± {strat_df['calmar'].std():.2f}")
        print(f"  Max Drawdown (mean): {strat_df['max_drawdown'].mean():.2%} ± {strat_df['max_drawdown'].std():.2%}")
        print(f"  Average Drawdown (mean): {strat_df['avg_drawdown'].mean():.2%} ± {strat_df['avg_drawdown'].std():.2%}")
        print(f"  Max Drawdown Duration (mean): {strat_df['max_drawdown_duration'].mean():.1f} days")
        print(f"  Worst Trade Return (mean): {strat_df['worst_trade'].mean():.2%} ± {strat_df['worst_trade'].std():.2%}")
        print(f"  Longest Losing Streak (mean): {strat_df['longest_losing_streak'].mean():.1f} trades")
        print(f"  Total Trades (mean): {strat_df['total_trades'].mean():.0f}")

        profitable_windows = (strat_df['net_return'] > 0).sum()
        print(f"  Profitable Windows: {profitable_windows}/{len(strat_df)} ({profitable_windows/len(strat_df):.1%})")

        positive_sharpe_windows = (strat_df['net_sharpe'] > 0).sum()
        print(f"  Positive Sharpe Windows: {positive_sharpe_windows}/{len(strat_df)} ({positive_sharpe_windows/len(strat_df):.1%})")

    print("\n" + "=" * 120)


if __name__ == "__main__":
    df = load_data()
    df = feature_engineering(df)
    run_walk_forward(df)
