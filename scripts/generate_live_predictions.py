"""
Live Prediction Generator

Loads trained, versioned model artifacts and runs real features through them.
Writes calibrated predictions with entry, stop-loss, and target to the database.

Timeframes:
  INTRADAY  - 1-minute candles, 1.5% target, 0.75% SL, ATR-scaled
  SWING     - Daily candles, 3% target, 1.5% SL, ATR-scaled
  LONGTERM  - Weekly candles, 20% target, 10% SL, ATR-scaled

Win probability is the model's calibrated output — not a hand-coded formula.
If no trained model is found for a timeframe, that timeframe is skipped entirely
and logged as a warning. Predictions are never generated from rules.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.db_sync import SessionLocal
from database.models import IndexTick, Prediction, Tick
from data_platform.feature_store.macro import extract_macro_features
from QuantResearchOS.ml.inference.calibration import calibrate_or_passthrough
from utils.logger import get_logger
from utils.time_utils import now_ist

logger = get_logger("live_predictions")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/saved"))

# NSE symbols to score. Expand this list as training data grows.
SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY",
    "TATAMOTORS", "ITC", "SBIN", "HINDUNILVR", "BAJFINANCE",
    "KOTAKBANK", "AXISBANK", "LT", "ASIANPAINT", "MARUTI",
]

# Timeframe config: yfinance period/interval, ATR window, target/SL multiples
TIMEFRAME_CONFIG = {
    "INTRADAY": {
        "period": "5d",
        "interval": "1m",
        "min_bars": 50,
        "atr_window": 14,
        "target_pct": 0.015,
        "sl_pct": 0.0075,
        "model_file": "intraday_lgbm.pkl",
        "model_version_key": "INTRADAY_MODEL_VERSION",
        "default_version": "LGBM_INTRADAY_v1",
    },
    "SWING": {
        "period": "1y",
        "interval": "1d",
        "min_bars": 60,
        "atr_window": 14,
        "target_pct": 0.03,
        "sl_pct": 0.015,
        "model_file": "swing_lgbm.pkl",
        "model_version_key": "SWING_MODEL_VERSION",
        "default_version": "LGBM_SWING_v1",
    },
    "LONGTERM": {
        "period": "2y",
        "interval": "1wk",
        "min_bars": 40,
        "atr_window": 10,
        "target_pct": 0.20,
        "sl_pct": 0.10,
        "model_file": "longterm_lgbm.pkl",
        "model_version_key": "LONGTERM_MODEL_VERSION",
        "default_version": "LGBM_LONGTERM_v1",
    },
}

# Minimum calibrated probability to emit a prediction.
# Below this threshold the signal is discarded — do not lower this without
# re-running walk-forward validation.
MIN_WIN_PROB = 0.55

# Indian round-trip transaction cost floor (STT + brokerage + GST + exchange + stamp)
# Real retail cost is ~0.35-0.60% round trip. We use 0.40% as conservative floor.
ROUNDTRIP_COST_PCT = 0.004


# ---------------------------------------------------------------------------
# Model loader — loads a serialized sklearn-compatible model
# ---------------------------------------------------------------------------

def load_model(model_file: str) -> Optional[object]:
    """
    Load a serialized model from MODEL_DIR.
    Returns None if the file does not exist — caller must handle this.
    Never raises — a missing model means skip the timeframe, not crash.
    """
    import pickle
    path = MODEL_DIR / model_file
    if not path.exists():
        logger.warning(
            f"Model file not found: {path}. "
            f"Run training/train_models.py to generate it. "
            f"Skipping this timeframe."
        )
        return None
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Loaded model: {path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {path}: {e}")
        return None


def get_model_version(version_key: str, default: str) -> str:
    return os.getenv(version_key, default)


# ---------------------------------------------------------------------------
# Data fetcher — yfinance with validation
# ---------------------------------------------------------------------------

def fetch_ohlcv(symbol: str, period: str, interval: str, min_bars: int) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV from yfinance. Returns None if data is insufficient or invalid.
    NSE suffix (.NS) is appended automatically.
    """
    ticker = f"{symbol}.NS"
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    except Exception as e:
        logger.error(f"yfinance download failed for {ticker}: {e}")
        return None

    if df is None or df.empty:
        logger.warning(f"No data returned for {ticker}")
        return None

    # Flatten MultiIndex columns (yfinance returns these for single ticker too)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df.dropna(subset=["close", "volume"])
    df = df[df["volume"] > 0]

    if len(df) < min_bars:
        logger.warning(f"{ticker}: only {len(df)} bars, need {min_bars}. Skipping.")
        return None

    return df


# ---------------------------------------------------------------------------
# Feature engineering — real computed features, no hardcoded values
# ---------------------------------------------------------------------------

def compute_atr(df: pd.DataFrame, window: int) -> pd.Series:
    """Average True Range."""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Wilder RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(df: pd.DataFrame, atr_window: int, macro: dict) -> Optional[pd.Series]:
    """
    Compute a feature vector from the last bar of df.
    Returns None if any critical feature is NaN.

    Features:
      - rsi_14           : RSI(14) of close
      - ma20_z           : (close - MA20) / std20  — mean reversion z-score
      - ma50_z           : (close - MA50) / std50
      - atr_pct          : ATR / close — normalized volatility
      - vwap_dist        : (close - VWAP) / VWAP — intraday momentum proxy
      - vol_ratio        : volume[-1] / volume[-20].mean() — volume surge
      - ret_1            : 1-bar return
      - ret_5            : 5-bar return
      - ret_20           : 20-bar return
      - vix_level        : India VIX
      - market_regime    : -1/0/1 from VIX thresholds
      - usd_inr_chg      : USD/INR 1-day change %
      - dow_chg          : Dow Jones overnight change %
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    features = {}

    # RSI
    rsi = compute_rsi(close, 14)
    features["rsi_14"] = rsi.iloc[-1]

    # Mean reversion z-scores
    for w in [20, 50]:
        if len(close) >= w:
            ma = close.rolling(w).mean()
            std = close.rolling(w).std()
            z = (close - ma) / std.replace(0, np.nan)
            features[f"ma{w}_z"] = z.iloc[-1]
        else:
            features[f"ma{w}_z"] = np.nan

    # ATR normalized
    atr = compute_atr(df, atr_window)
    current_close = close.iloc[-1]
    features["atr_pct"] = atr.iloc[-1] / current_close if current_close > 0 else np.nan

    # VWAP distance (uses full available history as proxy)
    cum_vol = volume.cumsum()
    cum_val = (close * volume).cumsum()
    vwap = cum_val / cum_vol.replace(0, np.nan)
    features["vwap_dist"] = (close.iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1] if vwap.iloc[-1] > 0 else 0.0

    # Volume ratio
    if len(volume) >= 20:
        features["vol_ratio"] = volume.iloc[-1] / volume.iloc[-20:].mean() if volume.iloc[-20:].mean() > 0 else 1.0
    else:
        features["vol_ratio"] = 1.0

    # Return features
    for lag, key in [(1, "ret_1"), (5, "ret_5"), (20, "ret_20")]:
        if len(close) > lag:
            features[key] = float((close.iloc[-1] / close.iloc[-lag - 1]) - 1)
        else:
            features[key] = np.nan

    # Macro
    features["vix_level"] = macro.get("vix_level", 15.0)
    features["market_regime"] = macro.get("market_regime", 1)
    features["usd_inr_chg"] = macro.get("usd_inr_chg", 0.0)
    features["dow_chg"] = macro.get("dow_chg", 0.0)

    s = pd.Series(features)

    # Drop if critical features are NaN
    critical = ["rsi_14", "atr_pct", "ret_1", "ret_5"]
    if s[critical].isna().any():
        logger.warning(f"Critical features NaN: {s[critical][s[critical].isna()].index.tolist()}")
        return None

    # Fill non-critical NaN with 0
    s = s.fillna(0.0)
    return s


def compute_sl_target(entry: float, direction: str, atr: float,
                       base_target_pct: float, base_sl_pct: float) -> tuple[float, float]:
    """
    ATR-scaled stop-loss and target.
    Uses ATR if available and reasonable, else falls back to fixed percentage.
    """
    atr_pct = atr / entry if entry > 0 and not np.isnan(atr) else 0.0

    # Use ATR if it's within a sensible range (0.1% - 5%)
    if 0.001 < atr_pct < 0.05:
        sl_distance = max(atr_pct * 1.0, base_sl_pct)
        target_distance = max(atr_pct * 2.0, base_target_pct)
    else:
        sl_distance = base_sl_pct
        target_distance = base_target_pct

    if direction == "BUY":
        sl = round(entry * (1 - sl_distance), 2)
        target = round(entry * (1 + target_distance), 2)
    else:
        sl = round(entry * (1 + sl_distance), 2)
        target = round(entry * (1 - target_distance), 2)

    return sl, target


# ---------------------------------------------------------------------------
# Prediction generator — one timeframe
# ---------------------------------------------------------------------------

def generate_predictions_for_timeframe(
    timeframe: str,
    model,
    model_version: str,
    macro: dict,
    config: dict,
) -> list[dict]:
    """
    Score all SYMBOLS for one timeframe using the loaded model.
    Returns a list of prediction dicts ready to insert into the database.
    """
    predictions = []
    atr_window = config["atr_window"]

    FEATURE_COLS = [
        "rsi_14", "ma20_z", "ma50_z", "atr_pct", "vwap_dist",
        "vol_ratio", "ret_1", "ret_5", "ret_20",
        "vix_level", "market_regime", "usd_inr_chg", "dow_chg",
    ]

    for sym in SYMBOLS:
        try:
            df = fetch_ohlcv(sym, config["period"], config["interval"], config["min_bars"])
            if df is None:
                continue

            features = build_features(df, atr_window, macro)
            if features is None:
                logger.warning(f"{sym} [{timeframe}]: feature build failed, skipping.")
                continue

            # Align features to expected model columns
            X = features[FEATURE_COLS].values.reshape(1, -1)

            # Get calibrated probability from model
            try:
                proba = model.predict_proba(X)[0]
                # proba is [p_loss, p_win] for binary classifiers
                win_prob = float(proba[1]) if len(proba) == 2 else float(proba[0])
            except AttributeError:
                # Regressor — treat output as a score, normalize to 0-1
                raw = float(model.predict(X)[0])
                win_prob = float(np.clip(raw, 0.0, 1.0))

            # Apply probability calibration if a calibrator artifact exists.
            # Falls back to raw win_prob if no calibrator has been fitted yet
            # (cold start — needs 100-500 resolved predictions first).
            raw_win_prob = win_prob
            win_prob = calibrate_or_passthrough(win_prob, timeframe=timeframe)
            if win_prob != raw_win_prob:
                logger.debug(f"{sym} [{timeframe}]: raw={raw_win_prob:.3f} calibrated={win_prob:.3f}")

            # Hard filter: only emit high-confidence predictions
            if win_prob < MIN_WIN_PROB:
                logger.debug(f"{sym} [{timeframe}]: win_prob={win_prob:.3f} below threshold, skipping.")
                continue

            # Net edge check: expected value after transaction costs
            # EV = win_prob * target_pct - (1 - win_prob) * sl_pct - roundtrip_cost
            ev = (win_prob * config["target_pct"]
                  - (1 - win_prob) * config["sl_pct"]
                  - ROUNDTRIP_COST_PCT)
            if ev <= 0:
                logger.debug(f"{sym} [{timeframe}]: EV={ev:.4f} negative after costs, skipping.")
                continue

            entry = round(float(df["close"].iloc[-1]), 2)
            atr_val = compute_atr(df, atr_window).iloc[-1]

            direction = "BUY" if win_prob >= MIN_WIN_PROB else "SELL"
            sl, target = compute_sl_target(
                entry, direction, atr_val,
                config["target_pct"], config["sl_pct"]
            )

            feat_summary = {
                "rsi_14": round(float(features["rsi_14"]), 2),
                "ma20_z": round(float(features["ma20_z"]), 3),
                "vwap_dist": round(float(features["vwap_dist"]), 4),
                "atr_pct": round(float(features["atr_pct"]), 4),
                "ret_5": round(float(features["ret_5"]), 4),
                "vix_level": macro.get("vix_level"),
                "market_regime": macro.get("market_regime"),
                "ev_estimate": round(ev, 4),
            }

            predictions.append({
                "symbol": sym,
                "timeframe": timeframe,
                "direction": direction,
                "entry_price": entry,
                "stop_loss": sl,
                "target_price": target,
                "predicted_probability": round(win_prob, 4),
                "model_version": model_version,
                "features_json": json.dumps(feat_summary),
            })

            logger.info(
                f"{sym} [{timeframe}] {direction} entry={entry} "
                f"SL={sl} T={target} prob={win_prob:.3f} EV={ev:.4f}"
            )

        except Exception as e:
            logger.error(f"Error generating {timeframe} prediction for {sym}: {e}", exc_info=True)

    return predictions


# ---------------------------------------------------------------------------
# Market index data for dashboard
# ---------------------------------------------------------------------------

def fetch_market_indices() -> list[dict]:
    indices = {
        "^NSEI": "NIFTY 50",
        "^NSEBANK": "NIFTY BANK",
        "^BSESN": "BSE SENSEX",
        "^INDIAVIX": "INDIA VIX",
    }
    results = []
    for ticker, name in indices.items():
        try:
            df = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
            if df is None or df.empty or len(df) < 2:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            latest = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2])
            chg = round(((latest - prev) / prev) * 100, 2) if prev > 0 else 0.0
            results.append({"symbol": name, "price": round(latest, 2), "change_pct": chg})
        except Exception as e:
            logger.warning(f"Failed to fetch index {ticker}: {e}")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    logger.info("Starting live prediction generation.")

    if SessionLocal is None:
        logger.critical("Database session unavailable. Check DATABASE_URL. Aborting.")
        sys.exit(1)

    # Fetch macro features once — shared across all timeframes
    macro = extract_macro_features()
    logger.info(f"Macro: VIX={macro.get('vix_level')} regime={macro.get('market_regime')}")

    now = now_ist()
    db = SessionLocal()

    try:
        # Store index ticks
        for idx in fetch_market_indices():
            db.add(IndexTick(
                timestamp=now,
                name=idx["symbol"],
                value=idx["price"],
                change=idx["change_pct"],
            ))

        all_predictions = []

        for timeframe, config in TIMEFRAME_CONFIG.items():
            model = load_model(config["model_file"])
            if model is None:
                # No trained model — skip this timeframe entirely
                # Run training/train_models.py to create the artifact
                continue

            model_version = get_model_version(
                config["model_version_key"],
                config["default_version"],
            )

            preds = generate_predictions_for_timeframe(
                timeframe=timeframe,
                model=model,
                model_version=model_version,
                macro=macro,
                config=config,
            )
            all_predictions.extend(preds)

        if not all_predictions:
            logger.warning(
                "No predictions generated. Either no trained models exist "
                "or no symbol passed the EV filter. "
                "Check MODEL_DIR and run training/train_models.py."
            )

        for sig in all_predictions:
            db.add(Prediction(
                id=str(uuid.uuid4()),
                symbol=sig["symbol"],
                horizon=sig["timeframe"],
                prediction=sig["direction"],
                entry_price=sig["entry_price"],
                stop_loss=sig["stop_loss"],
                target_price=sig["target_price"],
                confidence=sig["predicted_probability"],
                model_version=sig["model_version"],
                features_used=sig["features_json"],
                actual_outcome="OPEN",
                prediction_time=now,
                expiry_time=_expiry_time(sig["timeframe"], now),
            ))
            db.add(Tick(
                time=now,
                symbol=sig["symbol"],
                ltp=sig["entry_price"],
                volume=0,
            ))

        db.commit()
        logger.info(f"Committed {len(all_predictions)} predictions to database.")

    except Exception as e:
        logger.critical(f"Fatal error in prediction run: {e}", exc_info=True)
        db.rollback()
        sys.exit(1)
    finally:
        db.close()


def _expiry_time(timeframe: str, now: datetime) -> datetime:
    """Approximate expiry timestamp for outcome resolution scheduling."""
    if timeframe == "INTRADAY":
        # Same trading day at 15:30 IST
        return now.replace(hour=15, minute=30, second=0, microsecond=0)
    elif timeframe == "SWING":
        return now + timedelta(days=10)
    else:
        return now + timedelta(days=60)


if __name__ == "__main__":
    run()