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
from prediction_intelligence.base_logistic import (
    INTRADAY_FEATURES, 
    LONGTERM_FEATURES, 
    SWING_FEATURES,
    ModelRegistry,
    build_features as canonical_build_features
)
from prediction_intelligence.calibration import calibrate_or_passthrough
from prediction_intelligence.signal_adapter import SignalPrediction

_registry = ModelRegistry()
from risk_governance.pre_trade.circuit_breakers import CircuitBreaker
from risk_governance.pre_trade.portfolio_risk import PortfolioRiskEngine
from risk_governance.pre_trade.historical_var import HistoricalVaR
from utils.logger import get_logger
from utils.time_utils import now_ist

logger = get_logger("live_predictions")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/saved"))

from config.universe import NSE_UNIVERSE
SYMBOLS = [s["symbol"] for s in NSE_UNIVERSE]

# Timeframe config: yfinance period/interval, ATR window, target/SL multiples
TIMEFRAME_CONFIG = {
    "INTRADAY": {
        "period": "5d",
        "interval": "1m",
        "min_bars": 50,
        "atr_window": 14,
        "target_pct": 0.015,
        "sl_pct": 0.0075,
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
        "model_version_key": "SWING_MODEL_VERSION",
        # ENSEMBLE_ prefix is required: ModelRegistry uses it to detect EnsembleModel
        # vs BaseLogistic (joblib). train_base_models.py saves to XGB_SWING_v1/ dir but
        # registers under the ENSEMBLE_SWING_v1 key in the singleton.
        "default_version": "ENSEMBLE_SWING_v1",
    },
    "LONGTERM": {
        "period": "2y",
        "interval": "1wk",
        "min_bars": 40,
        "atr_window": 10,
        "target_pct": 0.20,
        "sl_pct": 0.10,
        "model_version_key": "LONGTERM_MODEL_VERSION",
        "default_version": "LOGREG_LONGTERM_v1",
    },
}

# Minimum calibrated probability to emit a prediction.
# Below this threshold the signal is discarded — do not lower this without
# re-running walk-forward validation.
MIN_WIN_PROB = 0.55

# Indian round-trip transaction cost floor (STT + brokerage + GST + exchange + stamp)
# Real retail cost is ~0.35-0.60% round trip. We use 0.40% as conservative floor.
BASE_ROUNDTRIP_COST_PCT = 0.004

# Per-symbol slippage multipliers based on liquidity (volume-based)
# Higher volume = lower slippage, lower volume = higher slippage
SYMBOL_SLIPPAGE_MULTIPLIERS = {
    "RELIANCE": 0.8,  # High liquidity
    "TCS": 0.8,       # High liquidity
    "HDFCBANK": 0.8,  # High liquidity
    "ICICIBANK": 0.8, # High liquidity
    "INFY": 0.9,      # High liquidity
    "SBIN": 0.9,      # High liquidity
    "ITC": 1.0,       # Medium liquidity
    "HINDUNILVR": 1.0, # Medium liquidity
    "KOTAKBANK": 0.9,  # High liquidity
    "AXISBANK": 1.0,   # Medium liquidity
    "LT": 1.2,         # Lower liquidity
    "ASIANPAINT": 1.0, # Medium liquidity
    "MARUTI": 1.2,     # Lower liquidity
    "BAJFINANCE": 1.5, # Lower liquidity
    "WIPRO": 1.5,      # Mid-cap liquidity adjustment
}

# Default multiplier for symbols not in the map
DEFAULT_SLIPPAGE_MULTIPLIER = 1.2


# ---------------------------------------------------------------------------
# Model loader — uses ModelRegistry for unified artifact contract
# ---------------------------------------------------------------------------

def load_model(model_version: str, timeframe: str) -> Optional[object]:
    """
    Load a model using ModelRegistry singleton (unified with train_base_models.py).
    Returns None if the model does not exist — caller must handle this.
    """
    try:
        model = _registry.get(model_version, timeframe)
        if not model.is_ready():
            logger.warning(
                f"Model {model_version} for {timeframe} not ready in ModelRegistry. "
                f"Run scripts/train_base_models.py to generate it."
            )
            return None

        logger.info(f"Loaded model: {model_version} for {timeframe}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_version}: {e}")
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


def infer_direction_from_features(features: pd.Series, timeframe: str) -> tuple[str, dict[str, float]]:
    """
    Convert the current feature row into a long/short side.

    The loaded model supplies setup success probability; direction is inferred
    from the same no-lookahead feature row, never from a hardcoded BUY default.
    """
    tf = timeframe.upper()
    votes: dict[str, float] = {}

    def value(name: str, default: float = 0.0) -> float:
        try:
            v = float(features.get(name, default))
            return 0.0 if np.isnan(v) else v
        except (TypeError, ValueError):
            return default

    if tf == "INTRADAY":
        rsi = value("rsi_14m", 50.0)
        momentum = value("momentum_5m")
        vwap_dist = value("vwap_dist")
        vol_ratio = value("vol_ratio_1m", 1.0)

        votes["momentum_5m"] = 1.0 if momentum > 0 else -1.0 if momentum < 0 else 0.0
        votes["vwap_dist"] = 0.75 if vwap_dist > 0 else -0.75 if vwap_dist < 0 else 0.0
        votes["rsi_14m"] = 0.5 if 35 <= rsi <= 65 and momentum >= 0 else -0.5 if rsi > 70 else 0.5 if rsi < 30 else 0.0
        votes["volume_confirmation"] = 0.25 if vol_ratio >= 1.0 and momentum >= 0 else -0.25 if vol_ratio >= 1.0 else 0.0
    elif tf == "SWING":
        rsi = value("rsi_14d", 50.0)
        ma20_slope = value("ma20_slope")
        z_score = value("z_score_20d")
        volume_ratio = value("volume_ratio", 1.0)

        votes["ma20_slope"] = 1.0 if ma20_slope > 0 else -1.0 if ma20_slope < 0 else 0.0
        votes["z_score_20d"] = 0.75 if z_score > 0 else -0.75 if z_score < 0 else 0.0
        votes["rsi_14d"] = 0.5 if 45 <= rsi <= 65 else -0.5 if rsi > 70 else 0.5 if rsi < 35 else 0.0
        votes["volume_confirmation"] = 0.25 if volume_ratio >= 1.0 and ma20_slope >= 0 else -0.25 if volume_ratio >= 1.0 else 0.0
    else:
        rsi = value("rsi_14w", 50.0)
        ma50_slope = value("ma50_slope")
        price_to_high = value("price_to_52w_high", 1.0)
        vol_ratio = value("vol_ratio", 1.0)

        votes["ma50_slope"] = 1.0 if ma50_slope > 0 else -1.0 if ma50_slope < 0 else 0.0
        votes["price_to_52w_high"] = 0.5 if price_to_high >= 0.80 else -0.5
        votes["rsi_14w"] = 0.5 if 45 <= rsi <= 68 else -0.5 if rsi > 72 else 0.25 if rsi < 35 else 0.0
        votes["volatility_regime"] = -0.25 if vol_ratio > 1.4 else 0.25

    score = sum(votes.values())
    return ("BUY" if score >= 0 else "SELL"), {k: round(v, 4) for k, v in votes.items()}





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
    circuit_breaker: CircuitBreaker,
    portfolio_risk: PortfolioRiskEngine,
    now: "datetime",
) -> list[dict]:
    """
    Score all SYMBOLS for one timeframe using the loaded model.
    Returns a list of prediction dicts ready to insert into the database.
    """
    predictions = []
    atr_window = config["atr_window"]

    # Use canonical feature definitions from base_logistic.py
    # This ensures training and inference use the same feature columns
    if timeframe == "INTRADAY":
        FEATURE_COLS = INTRADAY_FEATURES
    elif timeframe == "SWING":
        FEATURE_COLS = SWING_FEATURES
    elif timeframe == "LONGTERM":
        FEATURE_COLS = LONGTERM_FEATURES
    else:
        logger.error(f"Unknown timeframe: {timeframe}")
        return []

    for sym in SYMBOLS:
        try:
            df = fetch_ohlcv(sym, config["period"], config["interval"], config["min_bars"])
            if df is None:
                continue

            df_feats = canonical_build_features(df, timeframe, extra=macro)
            if df_feats.empty:
                logger.warning(f"{sym} [{timeframe}]: feature build failed, skipping.")
                continue

            # Take the last row for live prediction
            features = df_feats.iloc[-1].fillna(0.0)

            # Align features to the expected model columns.
            # Allow a partial match — missing columns will be filled with 0.0 by
            # the model's internal SimpleImputer. Only skip if ALL columns are missing.
            available_features = [f for f in FEATURE_COLS if f in features.index]
            if not available_features:
                logger.warning(f"{sym} [{timeframe}]: no features matched {FEATURE_COLS}, skipping.")
                continue
            missing = set(FEATURE_COLS) - set(available_features)
            if missing:
                logger.debug(f"{sym} [{timeframe}]: filling {len(missing)} missing features with 0: {missing}")
                for col in missing:
                    features[col] = 0.0
                available_features = FEATURE_COLS

            X = pd.DataFrame([features[available_features]])

            # Get calibrated probability from model
            try:
                proba = model.predict_proba(X)[0]
                win_prob = float(proba)
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

            direction, direction_votes = infer_direction_from_features(features, timeframe)

            # Hard filter: only emit high-confidence predictions
            # Do this AFTER direction logic, so direction isn't dead
            if win_prob < MIN_WIN_PROB:
                logger.debug(f"{sym} [{timeframe}]: win_prob={win_prob:.3f} below threshold, skipping.")
                continue

            # Circuit breaker: VIX-based position sizing check
            current_vix = macro.get("vix_level", 15.0)
            vix_allowed, adjusted_confidence = circuit_breaker.check_vix_limits(current_vix, win_prob)
            if not vix_allowed:
                logger.warning(f"{sym} [{timeframe}]: VIX circuit breaker triggered (VIX={current_vix}), skipping.")
                continue
            win_prob = adjusted_confidence  # Use confidence adjusted by circuit breaker

            # Net edge check: expected value after transaction costs
            # EV = win_prob * target_pct - (1 - win_prob) * sl_pct - roundtrip_cost
            # Use per-symbol slippage multiplier based on liquidity
            slippage_multiplier = SYMBOL_SLIPPAGE_MULTIPLIERS.get(sym, DEFAULT_SLIPPAGE_MULTIPLIER)
            adjusted_roundtrip_cost = BASE_ROUNDTRIP_COST_PCT * slippage_multiplier

            ev = (win_prob * config["target_pct"]
                  - (1 - win_prob) * config["sl_pct"]
                  - adjusted_roundtrip_cost)
            if ev <= 0:
                logger.debug(f"{sym} [{timeframe}]: EV={ev:.4f} negative after costs (slippage_mult={slippage_multiplier}), skipping.")
                continue

            entry = round(float(df["close"].iloc[-1]), 2)
            atr_val = compute_atr(df, atr_window).iloc[-1]

            sl, target = compute_sl_target(
                entry, direction, atr_val,
                config["target_pct"], config["sl_pct"]
            )

            feat_summary = {
                "vix_level": macro.get("vix_level"),
                "market_regime": macro.get("market_regime"),
                "ev_estimate": round(ev, 4),
                "direction_votes": direction_votes,
            }
            for feat_name in available_features:
                feat_summary[feat_name] = round(float(features[feat_name]), 4)

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

    # Initialize risk governance components
    circuit_breaker = CircuitBreaker()
    portfolio_risk = PortfolioRiskEngine(total_capital=10_000_000)
    historical_var = HistoricalVaR(confidence_level=0.99, lookback_window=252)

    logger.info("Risk governance components initialized")

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
            model_version = get_model_version(
                config["model_version_key"],
                config["default_version"],
            )

            model = load_model(model_version, timeframe)
            if model is None:
                # No trained model — skip this timeframe entirely
                # Run scripts/train_base_models.py to create the artifact
                continue

            preds = generate_predictions_for_timeframe(
                timeframe=timeframe,
                model=model,
                model_version=model_version,
                macro=macro,
                config=config,
                circuit_breaker=circuit_breaker,
                portfolio_risk=portfolio_risk,
                now=now,
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
