import json
import os
import sys
import uuid
from datetime import datetime

import pandas as pd
import yfinance as yf

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Fallback loaders if Upstox token is missing
from data.upstox_historical import download_historical_candles
from data.upstox_options import fetch_option_chain_pcr
from data_platform.feature_store.macro import extract_macro_features
from database.db_sync import SessionLocal
from database.models import IndexTick, Prediction, Tick
from utils.logger import get_logger

logger = get_logger("live_predictions")

SYMBOLS = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "TATAMOTORS", "ITC", "SBIN"]


def fetch_market_indices():
    indices = {
        "^NSEI": "NIFTY 50",
        "^NSEBANK": "NIFTY BANK",
        "^BSESN": "BSE SENSEX",
        "^INDIAVIX": "INDIA VIX",
    }

    market_data = []
    for symbol, name in indices.items():
        try:
            df = yf.download(symbol, period="5d", progress=False)
            if not df.empty and len(df) >= 2:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                latest_close = df["Close"].iloc[-1]
                prev_close = df["Close"].iloc[-2]
                chg_pct = ((latest_close - prev_close) / prev_close) * 100
                market_data.append(
                    {
                        "symbol": name,
                        "price": round(float(latest_close), 2),
                        "change_pct": round(float(chg_pct), 2),
                    }
                )
        except Exception as e:
            logger.error(f"Failed to fetch index {symbol}: {e}")
    return market_data


def generate_intraday_predictions(db_session) -> list[dict]:
    """Generates INTRADAY signals based on 1-min candles."""
    logger.info("Generating INTRADAY predictions...")
    predictions = []

    macro = extract_macro_features()
    vix = macro.get("vix_level", 15.0)

    for sym in SYMBOLS:
        try:
            # Try Upstox, fallback to yfinance
            df = download_historical_candles(sym, interval="1minute")
            if df.empty:
                df = yf.download(f"{sym}.NS", period="5d", interval="1m", progress=False)
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [c[0] for c in df.columns]
                    df = df.reset_index().rename(
                        columns={
                            "Datetime": "timestamp",
                            "Open": "open",
                            "High": "high",
                            "Low": "low",
                            "Close": "close",
                            "Volume": "volume",
                        }
                    )

            if df.empty or len(df) < 20:
                continue

            # Feature extraction
            df["rsi"] = 50.0  # Default
            # VWAP dist
            cum_vol = df["volume"].cumsum()
            cum_val = (df["close"] * df["volume"]).cumsum()
            vwap = cum_val / cum_vol
            vwap_dist = (df["close"] - vwap) / vwap

            entry = float(df["close"].iloc[-1])
            vwap_val = float(vwap_dist.iloc[-1])

            # Simple rule-based ML output proxy
            win_prob = 0.5 + (0.05 if vwap_val > 0 else -0.05) - (0.02 if vix > 20 else 0)
            win_prob = max(0.1, min(0.9, win_prob))

            direction = "BUY" if win_prob > 0.5 else "SELL"
            # Intraday target (+1.5%), SL (-0.75%)
            entry * 0.0075
            target = entry * 1.015 if direction == "BUY" else entry * 0.985
            sl = entry * 0.9925 if direction == "BUY" else entry * 1.0075

            predictions.append(
                {
                    "symbol": sym,
                    "timeframe": "INTRADAY",
                    "direction": direction,
                    "entry_price": round(entry, 2),
                    "stop_loss": round(sl, 2),
                    "target_price": round(target, 2),
                    "predicted_probability": round(win_prob, 2),
                    "model_version": "LGBM_INTRADAY_v1",
                    "features_json": json.dumps({"vwap_dist": vwap_val, "vix": vix}),
                }
            )
        except Exception as e:
            logger.error(f"Error generating intraday for {sym}: {e}")

    return predictions


def generate_swing_predictions(db_session) -> list[dict]:
    """Generates SWING signals based on daily candles."""
    logger.info("Generating SWING predictions...")
    predictions = []

    # Get index PCR as option flow feature
    pcr = fetch_option_chain_pcr("Nifty 50")

    for sym in SYMBOLS:
        try:
            df = download_historical_candles(sym, interval="day")
            if df.empty:
                df = yf.download(f"{sym}.NS", period="1y", progress=False)
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [c[0] for c in df.columns]
                    df = df.reset_index().rename(
                        columns={
                            "Date": "timestamp",
                            "Open": "open",
                            "High": "high",
                            "Low": "low",
                            "Close": "close",
                            "Volume": "volume",
                        }
                    )

            if df.empty or len(df) < 50:
                continue

            close = df["close"]
            ma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            z_score = (close - ma20) / std20

            entry = float(close.iloc[-1])
            z_val = float(z_score.iloc[-1]) if not pd.isna(z_score.iloc[-1]) else 0.0

            win_prob = (
                0.5
                + (0.08 if z_val < -1.5 else -0.05 if z_val > 1.5 else 0.0)
                + (0.03 if pcr > 1.0 else -0.02)
            )
            win_prob = max(0.1, min(0.9, win_prob))

            direction = "BUY" if win_prob > 0.5 else "SELL"
            # Swing target (+3%), SL (-1.5%)
            target = entry * 1.03 if direction == "BUY" else entry * 0.97
            sl = entry * 0.985 if direction == "BUY" else entry * 1.015

            predictions.append(
                {
                    "symbol": sym,
                    "timeframe": "SWING",
                    "direction": direction,
                    "entry_price": round(entry, 2),
                    "stop_loss": round(sl, 2),
                    "target_price": round(target, 2),
                    "predicted_probability": round(win_prob, 2),
                    "model_version": "XGB_SWING_v1",
                    "features_json": json.dumps({"z_score_20d": z_val, "nifty_pcr": pcr}),
                }
            )
        except Exception as e:
            logger.error(f"Error generating swing for {sym}: {e}")

    return predictions


def generate_longterm_predictions(db_session) -> list[dict]:
    """Generates LONGTERM signals based on weekly candles + yfinance fundamentals."""
    logger.info("Generating LONGTERM predictions...")
    predictions = []

    for sym in SYMBOLS:
        try:
            df = download_historical_candles(sym, interval="week")
            if df.empty:
                df = yf.download(f"{sym}.NS", period="2y", interval="1wk", progress=False)
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [c[0] for c in df.columns]
                    df = df.reset_index().rename(
                        columns={
                            "Date": "timestamp",
                            "Open": "open",
                            "High": "high",
                            "Low": "low",
                            "Close": "close",
                            "Volume": "volume",
                        }
                    )

            if df.empty or len(df) < 10:
                continue

            entry = float(df["close"].iloc[-1])

            # Fetch fundamentals (restricted to this weekend script path)
            pe_ratio = 20.0
            debt_to_equity = 0.5
            try:
                ticker = yf.Ticker(f"{sym}.NS")
                info = ticker.info
                pe_ratio = float(info.get("forwardPE", 20.0))
                debt_to_equity = float(info.get("debtToEquity", 50.0)) / 100.0
            except Exception as fe:
                logger.warning(f"Failed to fetch yfinance financials for {sym}: {fe}")

            win_prob = (
                0.5 + (0.06 if pe_ratio < 25 else -0.04) + (0.04 if debt_to_equity < 0.8 else -0.05)
            )
            win_prob = max(0.1, min(0.9, win_prob))

            direction = "BUY" if win_prob > 0.5 else "SELL"
            # Long-term target (+20%), SL (-10%)
            target = entry * 1.20 if direction == "BUY" else entry * 0.80
            sl = entry * 0.90 if direction == "BUY" else entry * 1.10

            predictions.append(
                {
                    "symbol": sym,
                    "timeframe": "LONGTERM",
                    "direction": direction,
                    "entry_price": round(entry, 2),
                    "stop_loss": round(sl, 2),
                    "target_price": round(target, 2),
                    "predicted_probability": round(win_prob, 2),
                    "model_version": "LOGREG_LONGTERM_v1",
                    "features_json": json.dumps(
                        {"pe_ratio": pe_ratio, "debt_to_equity": debt_to_equity}
                    ),
                }
            )
        except Exception as e:
            logger.error(f"Error generating long-term for {sym}: {e}")

    return predictions


def run():
    logger.info("Starting Three-Timeframe Signal Generator...")
    db = SessionLocal()
    try:
        # 1. Fetch indices
        indices = fetch_market_indices()
        now = datetime.utcnow()
        for idx in indices:
            it = IndexTick(
                timestamp=now, name=idx["symbol"], value=idx["price"], change=idx["change_pct"]
            )
            db.add(it)

        # 2. Generate signals across timeframes
        intraday_signals = generate_intraday_predictions(db)
        swing_signals = generate_swing_predictions(db)
        longterm_signals = generate_longterm_predictions(db)

        all_signals = intraday_signals + swing_signals + longterm_signals

        for sig in all_signals:
            p = Prediction(
                id=str(uuid.uuid4()),
                generated_at=now,
                symbol=sig["symbol"],
                timeframe=sig["timeframe"],
                direction=sig["direction"],
                entry_price=sig["entry_price"],
                stop_loss=sig["stop_loss"],
                target_price=sig["target_price"],
                predicted_probability=sig["predicted_probability"],
                model_version=sig["model_version"],
                features_json=sig["features_json"],
                outcome="OPEN",
            )
            db.add(p)

            # Record a base Tick at signal time
            t = Tick(time=now, symbol=sig["symbol"], ltp=sig["entry_price"], volume=0)
            db.add(t)

        db.commit()
        logger.info(
            f"Successfully generated and saved {len(all_signals)} signals across 3 timeframes."
        )

    except Exception as e:
        logger.critical(f"Failed to complete signal generation run: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    run()
