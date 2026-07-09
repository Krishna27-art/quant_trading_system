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
from database.models import IndexTick, Prediction
from data_platform.feature_store.macro import extract_macro_features
from prediction_intelligence.base_logistic import (
    INTRADAY_FEATURES, 
    LONGTERM_FEATURES, 
    SWING_FEATURES,
    ModelRegistry,
    build_features as canonical_build_features,
    FEATURE_SCHEMA_VERSION
)
from prediction_intelligence.calibration import calibrate_or_passthrough
from prediction_intelligence.signal_adapter import SignalPrediction

_registry = ModelRegistry()
from risk_governance.pre_trade.circuit_breakers import CircuitBreaker
from risk_governance.pre_trade.portfolio_risk import PortfolioRiskEngine
from utils.logger import get_logger
from utils.time_utils import now_ist

logger = get_logger("live_predictions")

from validation.prediction_record import PredictionRecord
from validation.prediction_store import PredictionStore
_store = PredictionStore()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/saved"))

from config.universe import NSE_UNIVERSE
SYMBOLS = [s["symbol"] for s in NSE_UNIVERSE]

# Cache for per-symbol fundamental data
_fundamentals_cache: dict[str, dict[str, float]] = {}

def get_symbol_fundamentals(symbol: str) -> dict[str, float]:
    if symbol in _fundamentals_cache:
        return _fundamentals_cache[symbol]
    try:
        ticker_symbol = f"{symbol}.NS"
        info = yf.Ticker(ticker_symbol).info
        fund = {
            "pe_ratio": float(info.get("forwardPE", 20.0)),
            "debt_to_equity": float(info.get("debtToEquity", 50.0)) / 100.0,
        }
        logger.info(f"Fetched fundamentals for {symbol}: PE={fund['pe_ratio']:.2f}, D/E={fund['debt_to_equity']:.2f}")
    except Exception as e:
        logger.warning(f"Failed to fetch fundamentals for {symbol}: {e}. Using fallback defaults (PE=20.0, D/E=0.5).")
        fund = {"pe_ratio": 20.0, "debt_to_equity": 0.5}
    _fundamentals_cache[symbol] = fund
    return fund

# Timeframe config: yfinance period/interval, ATR window, target/SL multipliers
# target_multiplier and sl_multiplier allow independent scaling of target and stop-loss
# from ATR, breaking the fixed 2:1 ratio constraint
TIMEFRAME_CONFIG = {
    "INTRADAY": {
        "period": "5d",
        "interval": "1m",
        "min_bars": 50,
        "atr_window": 14,
        "target_pct": 0.015,
        "sl_pct": 0.0075,
        "target_multiplier": 2.5,  # Target = ATR * 2.5 (was hardcoded 2.0)
        "sl_multiplier": 1.0,    # SL = ATR * 1.0
        "atr_min_pct": 0.0005,
        "atr_max_pct": 0.01,
        "model_version_key": "INTRADAY_MODEL_VERSION",
        "default_version": "META_INTRADAY_v1",
    },
    "SWING": {
        "period": "1y",
        "interval": "1d",
        "min_bars": 60,
        "atr_window": 14,
        "target_pct": 0.03,
        "sl_pct": 0.015,
        "target_multiplier": 3.0,  # Target = ATR * 3.0 (was hardcoded 2.0)
        "sl_multiplier": 1.0,    # SL = ATR * 1.0
        "atr_min_pct": 0.001,
        "atr_max_pct": 0.05,
        "model_version_key": "SWING_MODEL_VERSION",
        "default_version": "META_SWING_v1",
    },
    "LONGTERM": {
        "period": "2y",
        "interval": "1wk",
        "min_bars": 40,
        "atr_window": 10,
        "target_pct": 0.20,
        "sl_pct": 0.10,
        "target_multiplier": 2.5,  # Target = ATR * 2.5 (was hardcoded 2.0)
        "sl_multiplier": 1.0,    # SL = ATR * 1.0
        "atr_min_pct": 0.005,
        "atr_max_pct": 0.20,
        "model_version_key": "LONGTERM_MODEL_VERSION",
        "default_version": "META_LONGTERM_v1",
    },
}

# Minimum calibrated probability to emit a prediction.
# Lowered to 0.35 so INTRADAY/SWING/LONGTERM signals all flow through.
# The 50% threshold at the UI layer is user-controllable (slider defaults to 0%).
MIN_WIN_PROB = 0.35

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

# Data sources with proper fallback chain
from data_platform.upstox_client import get_candles, get_index_overview
from data_platform.feeds.bar_aggregator import get_cached_ohlcv
from data_platform.sources.ingestion.ingestion_engine import IngestionEngine

def fetch_ohlcv(symbol: str, period: str, interval: str, min_bars: int) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV with proper fallback chain:
    1. Local BarAggregator cache (Redis/in-memory)
    2. Upstox REST (with token validation)
    3. IngestionEngine (NSELib → scraper → cache)
    """
    try:
        # Step 1: Attempt local in-memory/Redis cache read (zero REST latency)
        cached_df = get_cached_ohlcv(symbol, interval, min_bars=min_bars)
        if len(cached_df) >= min_bars:
            return cached_df

        # Step 2: Fallback to Upstox REST on cold start
        logger.info(f"{symbol} [{interval}]: local cache cold ({len(cached_df)}/{min_bars} bars) — bootstrapping from Upstox REST")
        days = 180
        if "d" in period:
            days = int(period.replace("d", ""))
        elif "y" in period:
            days = int(period.replace("y", "")) * 365
        
        upstox_interval = "1day"
        if interval == "1m":
            upstox_interval = "1minute"
        elif interval == "1d":
            upstox_interval = "1day"
        elif interval == "1wk":
            upstox_interval = "1week"
            
        try:
            candles = get_candles(symbol, interval=upstox_interval, days=days)
            if candles:
                df = pd.DataFrame(candles)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
                df.sort_index(inplace=True)
                
                df = df[["open", "high", "low", "close", "volume"]].copy()
                df = df.astype({c: float for c in df.columns})
                df = df.dropna(subset=["close", "volume"])
                df = df[df["volume"] > 0]
                
                if len(df) >= min_bars:
                    logger.info(f"{symbol}: successfully fetched {len(df)} bars from Upstox")
                    return df
                else:
                    logger.warning(f"{symbol}: only {len(df)} bars from Upstox, need {min_bars}")
        except Exception as upstox_err:
            # Check if it's an auth/token error
            error_msg = str(upstox_err).lower()
            if "token" in error_msg or "auth" in error_msg or "unauthorized" in error_msg or "401" in error_msg:
                logger.error(
                    f"🚨 CRITICAL: Upstox token invalid or expired! "
                    f"Run the manual token refresh script immediately. "
                    f"Error: {upstox_err}"
                )
            else:
                logger.warning(f"Upstox fetch failed for {symbol}: {upstox_err}")
        
        # Step 3: Fallback to IngestionEngine (NSELib → scraper → cache)
        logger.warning(f"{symbol} [{interval}]: Upstox failed, falling back to IngestionEngine")
        try:
            ingestion = IngestionEngine(cache_enabled=True)
            
            # Calculate date range for IngestionEngine
            from datetime import datetime, timedelta
            to_date = datetime.now().strftime("%Y-%m-%d")
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            result = ingestion.fetch_equity_history(symbol, from_date, to_date, use_fallback=True)
            
            if result.success and result.data:
                # Convert IngestionEngine data to DataFrame
                data_list = result.data if isinstance(result.data, list) else [result.data]
                df = pd.DataFrame(data_list)
                
                # Standardize column names
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df.set_index("timestamp", inplace=True)
                elif "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df.set_index("date", inplace=True)
                
                df.sort_index(inplace=True)
                
                # Ensure required columns exist
                required_cols = ["open", "high", "low", "close", "volume"]
                if all(col in df.columns for col in required_cols):
                    df = df[required_cols].copy()
                    df = df.astype({c: float for c in df.columns})
                    df = df.dropna(subset=["close", "volume"])
                    df = df[df["volume"] > 0]
                    
                    if len(df) >= min_bars:
                        logger.info(f"{symbol}: successfully fetched {len(df)} bars from IngestionEngine (source: {result.source})")
                        return df
                    else:
                        logger.warning(f"{symbol}: only {len(df)} bars from IngestionEngine, need {min_bars}")
                else:
                    logger.error(f"{symbol}: IngestionEngine data missing required columns: {df.columns}")
            else:
                logger.warning(f"{symbol}: IngestionEngine returned no data (error: {result.error})")
        except Exception as ingestion_err:
            logger.error(f"{symbol}: IngestionEngine fallback failed: {ingestion_err}")
        
        # All sources failed
        logger.error(f"{symbol}: All data sources failed. Skipping prediction.")
        return None
            
    except Exception as e:
        logger.error(f"OHLCV fetch failed for {symbol}: {e}", exc_info=True)
        return None


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
        macd_hist = value("macd_hist", 0.0)

        # Momentum: primary direction signal (weight 1.0)
        votes["momentum_5m"] = 1.0 if momentum > 0 else -1.0 if momentum < 0 else 0.0
        # VWAP distance: secondary directional (weight 0.75)
        votes["vwap_dist"] = 0.75 if vwap_dist > 0 else -0.75 if vwap_dist < 0 else 0.0
        # RSI: symmetric — oversold favours BUY, overbought favours SELL, neutral is 0
        if rsi < 30:
            votes["rsi_14m"] = 0.75   # strongly oversold → BUY
        elif rsi < 45:
            votes["rsi_14m"] = 0.25   # mildly oversold → lean BUY
        elif rsi <= 55:
            votes["rsi_14m"] = 0.0    # neutral RSI → no vote
        elif rsi <= 70:
            votes["rsi_14m"] = -0.25  # mildly overbought → lean SELL
        else:
            votes["rsi_14m"] = -0.75  # strongly overbought → SELL
        # MACD histogram: confirmation (weight 0.5)
        votes["macd_hist"] = 0.5 if macd_hist > 0 else -0.5 if macd_hist < 0 else 0.0
        # Volume confirmation: amplify only (weight 0.25)
        votes["volume_confirmation"] = 0.25 if vol_ratio >= 1.2 else 0.0
    elif tf == "SWING":
        rsi = value("rsi_14d", 50.0)
        ma20_slope = value("ma20_slope")
        z_score = value("z_score_20d")
        volume_ratio = value("volume_ratio", 1.0)

        # ma20_slope > 0  → trend UP  → BUY vote (+1.0)
        votes["ma20_slope"] = 1.0 if ma20_slope > 0 else -1.0 if ma20_slope < 0 else 0.0
        # z_score > 0 → price is ABOVE 20d mean → bullish momentum → BUY (+0.75)
        # z_score < -1.5 → deeply oversold → strong BUY mean-reversion
        if z_score < -1.5:
            votes["z_score_20d"] = 1.0   # deep oversold → strong BUY
        elif z_score < 0:
            votes["z_score_20d"] = 0.5   # slightly below mean → lean BUY
        elif z_score < 1.0:
            votes["z_score_20d"] = 0.25  # near mean, slight bullish
        elif z_score < 2.0:
            votes["z_score_20d"] = -0.25 # extended above mean → mild SELL
        else:
            votes["z_score_20d"] = -0.75 # deeply overbought → SELL
        # Symmetric RSI for swing
        if rsi < 35:
            votes["rsi_14d"] = 0.75
        elif rsi < 45:
            votes["rsi_14d"] = 0.25
        elif rsi <= 55:
            votes["rsi_14d"] = 0.0
        elif rsi <= 70:
            votes["rsi_14d"] = -0.25
        else:
            votes["rsi_14d"] = -0.75
        votes["volume_confirmation"] = 0.25 if volume_ratio >= 1.2 else 0.0
    else:
        rsi = value("rsi_14w", 50.0)
        ma50_slope = value("ma50_slope")
        price_to_high = value("price_to_52w_high", 1.0)
        vol_ratio = value("vol_ratio", 1.0)

        votes["ma50_slope"] = 1.0 if ma50_slope > 0 else -1.0 if ma50_slope < 0 else 0.0
        votes["price_to_52w_high"] = 0.5 if price_to_high >= 0.80 else -0.5
        if rsi < 35:
            votes["rsi_14w"] = 0.75
        elif rsi < 45:
            votes["rsi_14w"] = 0.25
        elif rsi <= 55:
            votes["rsi_14w"] = 0.0
        elif rsi <= 68:
            votes["rsi_14w"] = -0.25
        else:
            votes["rsi_14w"] = -0.75
        votes["volatility_regime"] = -0.25 if vol_ratio > 1.4 else 0.25

    score = sum(votes.values())
    # Positive or neutral score → BUY; strictly negative → SELL
    return ("BUY" if score >= 0 else "SELL"), {k: round(v, 4) for k, v in votes.items()}


def generate_explanation(features: pd.Series, direction: str, timeframe: str) -> str:
    """Generate a clean, explainable list of reasons for the prediction."""
    reasons = []
    tf = timeframe.upper()
    if tf == "INTRADAY":
        rsi = float(features.get("rsi_14m", 50.0))
        momentum = float(features.get("momentum_5m", 0.0))
        vwap_dist = float(features.get("vwap_dist", 0.0))
        
        if direction == "BUY":
            if momentum > 0: reasons.append("Positive momentum shift")
            if vwap_dist > 0: reasons.append("Trading above daily VWAP")
            if rsi < 40: reasons.append("RSI recovery from oversold zone")
            elif rsi < 60: reasons.append("Neutral RSI support")
        else:
            if momentum < 0: reasons.append("Negative momentum breakdown")
            if vwap_dist < 0: reasons.append("Trading below daily VWAP")
            if rsi > 60: reasons.append("RSI pullback from overbought zone")
            elif rsi > 40: reasons.append("Neutral RSI resistance")
            
    elif tf == "SWING":
        rsi = float(features.get("rsi_14d", 50.0))
        ma20_slope = float(features.get("ma20_slope", 0.0))
        z_score = float(features.get("z_score_20d", 0.0))
        
        if direction == "BUY":
            if ma20_slope > 0: reasons.append("Moving averages pointing upwards")
            if z_score < -1.0: reasons.append("Mean reversion: oversold z-score")
            if rsi < 40: reasons.append("Daily RSI recovery")
        else:
            if ma20_slope < 0: reasons.append("Moving averages pointing downwards")
            if z_score > 1.0: reasons.append("Mean reversion: overbought z-score")
            if rsi > 60: reasons.append("Daily RSI pullback")
            
    else:
        rsi = float(features.get("rsi_14w", 50.0))
        ma50_slope = float(features.get("ma50_slope", 0.0))
        price_to_high = float(features.get("price_to_52w_high", 1.0))
        
        if direction == "BUY":
            if ma50_slope > 0: reasons.append("Long-term moving average trend is bullish")
            if price_to_high > 0.9: reasons.append("Consolidating near 52-week highs")
            if rsi < 45: reasons.append("Weekly RSI base formation")
        else:
            if ma50_slope < 0: reasons.append("Long-term moving average trend is bearish")
            if price_to_high < 0.8: reasons.append("Trading far below 52-week highs")
            if rsi > 65: reasons.append("Weekly RSI exhaustion")
            
    if not reasons:
        reasons.append(f"Model signal based on {direction.lower()} alignment")
        
    return " | ".join(reasons)


def compute_local_attributions(model, X_df: pd.DataFrame, feature_cols: list[str]) -> list[tuple[str, float]]:
    """Compute exact feature contributions to the logistic regression model's prediction."""
    try:
        if not hasattr(model, "logistic") or model.logistic is None or model.logistic.pipeline is None:
            return []
        pipeline = model.logistic.pipeline
        clf = pipeline.named_steps.get("classifier")
        scaler = pipeline.named_steps.get("scaler")
        imputer = pipeline.named_steps.get("imputer")
        
        if clf is None or scaler is None or imputer is None:
            return []
            
        raw_vals = X_df[feature_cols].values
        imputed_vals = imputer.transform(raw_vals)
        scaled_vals = scaler.transform(imputed_vals)[0]
        
        coefs = clf.coef_[0]
        contributions = coefs * scaled_vals
        
        attribs = list(zip(feature_cols, contributions))
        attribs.sort(key=lambda item: abs(item[1]), reverse=True)
        return attribs
    except Exception as exc:
        logger.error(f"Error computing local attributions: {exc}")
        return []







def compute_sl_target(entry: float, direction: str, atr: float,
                       base_target_pct: float, base_sl_pct: float,
                       target_multiplier: float = 2.0,
                       sl_multiplier: float = 1.0,
                       atr_min_pct: float = 0.001,
                       atr_max_pct: float = 0.05) -> tuple[float, float]:
    """
    ATR-scaled stop-loss and target.
    Uses ATR if available and reasonable, else falls back to fixed percentage.
    
    Args:
        entry: Entry price
        direction: "BUY" or "SELL"
        atr: Average True Range value
        base_target_pct: Fallback target percentage if ATR unavailable
        base_sl_pct: Fallback stop-loss percentage if ATR unavailable
        target_multiplier: Multiplier for ATR when computing target distance
        sl_multiplier: Multiplier for ATR when computing stop-loss distance
        atr_min_pct: Minimum sensible ATR percentage
        atr_max_pct: Maximum sensible ATR percentage
    
    Returns:
        (stop_loss, target_price) tuple
    """
    atr_pct = atr / entry if entry > 0 and not np.isnan(atr) else 0.0

    # Use ATR if it's within a sensible range
    if atr_min_pct < atr_pct < atr_max_pct:
        sl_distance = max(atr_pct * sl_multiplier, base_sl_pct)
        target_distance = max(atr_pct * target_multiplier, base_target_pct)
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
    model_long,
    model_short,
    model_version: str,
    macro: dict,
    config: dict,
    circuit_breaker: CircuitBreaker,
    portfolio_risk: PortfolioRiskEngine,
    now: "datetime",
    db,
) -> list[SignalPrediction]:
    """
    Score all SYMBOLS for one timeframe using the loaded long/short models.
    Returns a list of SignalPrediction objects ready to insert into the database.
    """
    predictions = []
    atr_window = config["atr_window"]

    # Load existing open predictions to track portfolio risk controls
    from database.models import Prediction
    open_preds = db.query(Prediction).filter(Prediction.actual_outcome == "OPEN").all()
    current_open_positions = [{"symbol": p.symbol, "horizon": p.horizon} for p in open_preds]

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

            extra_data = dict(macro)
            if timeframe == "LONGTERM":
                fund = get_symbol_fundamentals(sym)
                extra_data.update(fund)

            df_feats = canonical_build_features(df, timeframe, extra=extra_data)
            if df_feats.empty:
                logger.warning(f"{sym} [{timeframe}]: feature build failed, skipping.")
                continue

            # Fail closed on schema mismatch
            missing_cols = set(FEATURE_COLS) - set(df_feats.columns)
            if missing_cols:
                logger.error(f"{sym} [{timeframe}]: Fail-closed rejection! Missing mandatory columns: {missing_cols}")
                continue

            # Extract latest row
            features = df_feats.iloc[-1].copy()

            # Check if features have NaNs or need imputation
            if features[FEATURE_COLS].isna().any():
                imputer = ModelRegistry().get_imputer(timeframe)
                if imputer is not None:
                    try:
                        imputed_vals = imputer.transform(features[FEATURE_COLS].to_frame().T)[0]
                        for idx, col in enumerate(FEATURE_COLS):
                            features[col] = imputed_vals[idx]
                    except Exception as exc:
                        logger.error(f"{sym} [{timeframe}]: Imputer transform failed ({exc}), rejecting tick.")
                        continue
                else:
                    # Fallback to neutral defaults ONLY if imputer is not yet trained/available
                    NEUTRAL_DEFAULTS = {
                        "rsi_14m": 50.0,
                        "rsi_14d": 50.0,
                        "rsi_14w": 50.0,
                        "price_to_52w_high": 1.0,
                        "vol_ratio": 1.0,
                        "vol_ratio_1m": 1.0,
                        "atr_pct": 0.02,
                        "volume_ratio": 1.0,
                        "nifty_pcr": 1.0,
                        "pe_ratio": 20.0,
                        "debt_to_equity": 0.5,
                        "vix": 15.0,
                    }
                    for col in FEATURE_COLS:
                        if pd.isna(features[col]):
                            if col not in NEUTRAL_DEFAULTS:
                                logger.error(f"{sym} [{timeframe}]: Missing core feature {col!r} without imputer/default. Rejecting tick.")
                                break
                            features[col] = NEUTRAL_DEFAULTS[col]
                    else:
                        pass
                    if features[FEATURE_COLS].isna().any():
                        continue

            available_features = FEATURE_COLS

            X = pd.DataFrame([features[available_features]])

            direction, direction_votes = infer_direction_from_features(features, timeframe)

            # Route to correct side model
            model = model_long if direction == "BUY" else model_short
            
            if model is None or not model.is_ready():
                logger.error(f"No ready model available for {timeframe} ({direction}). Skipping prediction. (Model missing or untrained)")
                continue

            # Get calibrated probability from the selected model
            try:
                proba = model.predict_proba(X)[0]
                if isinstance(proba, (np.ndarray, list)) and len(proba) > 1:
                    win_prob = float(proba[1])
                else:
                    win_prob = float(proba)
            except AttributeError:
                # Regressor — treat output as a score, normalize to 0-1
                raw = float(model.predict(X)[0])
                win_prob = float(np.clip(raw, 0.0, 1.0))

            # Raw win prob from side-specific model directly reflects correct outcome side
            raw_win_prob = win_prob

            # Apply probability calibration if a calibrator artifact exists.
            # Falls back to raw win_prob if no calibrator has been fitted yet
            # (cold start — needs 100-500 resolved predictions first).
            win_prob = calibrate_or_passthrough(raw_win_prob, timeframe=timeframe, direction=direction)
            if win_prob != raw_win_prob:
                logger.debug(f"{sym} [{timeframe}] ({direction}): raw={raw_win_prob:.3f} calibrated={win_prob:.3f}")

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

            # Portfolio risk check
            is_allowed, risk_reason = portfolio_risk.check_position_limits(current_open_positions, sym)
            if not is_allowed:
                logger.warning(f"{sym} [{timeframe}]: Portfolio risk limit check failed: {risk_reason}. Skipping prediction.")
                continue

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
                config["target_pct"], config["sl_pct"],
                target_multiplier=config.get("target_multiplier", 2.0),
                sl_multiplier=config.get("sl_multiplier", 1.0),
                atr_min_pct=config.get("atr_min_pct", 0.001),
                atr_max_pct=config.get("atr_max_pct", 0.05),
            )

            feat_summary = {
                "vix_level": macro.get("vix_level"),
                "market_regime": macro.get("market_regime"),
                "ev_estimate": round(ev, 4),
                "direction_votes": direction_votes,
            }
            for feat_name in available_features:
                feat_summary[feat_name] = round(float(features[feat_name]), 4)

            attribs = compute_local_attributions(model, X, available_features)
            feat_summary["top_attributions"] = {name: round(val, 4) for name, val in attribs[:5]}

            explanation = generate_explanation(features, direction, timeframe)
            if attribs:
                drivers_str = ", ".join([f"{name} ({'+' if val >= 0 else ''}{round(val, 2)})" for name, val in attribs[:3]])
                explanation += f" | Model Drivers: {drivers_str}"

            # Convert direction to Pydantic prediction format: 2=long (BUY), 1=short (SELL), 0=hold
            pred_class = 2 if direction == "BUY" else 1

            # Compute risk_reward_ratio
            risk_amt = abs(entry - sl)
            reward_amt = abs(target - entry)
            rr_ratio = reward_amt / risk_amt if risk_amt > 0 else 0.0

            sig_pred = SignalPrediction(
                date=now,
                symbol=sym,
                prediction=pred_class,
                confidence=round(win_prob, 4),
                win_probability=round(win_prob, 4),
                target_price=target,
                stop_loss=sl,
                expected_return=round(ev, 6),
                risk_reward_ratio=round(rr_ratio, 2),
                model_version=model_version,
                metadata={
                    "timeframe": timeframe,
                    "feature_version": FEATURE_SCHEMA_VERSION,
                    "features_json": json.dumps(feat_summary),
                    "reason": explanation,
                    "entry_price": entry,
                }
            )
            predictions.append(sig_pred)
            current_open_positions.append({"symbol": sym, "horizon": timeframe})

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
    """Fetch indices quotes using Upstox get_index_overview."""
    results = []
    try:
        overview = get_index_overview()
        for name, data in overview.items():
            # Translate keys
            pretty_name = "NIFTY 50" if name == "NIFTY50" else "BSE SENSEX" if name == "SENSEX" else "NIFTY BANK" if name == "BANKNIFTY" else "INDIA VIX" if name == "INDIAVIX" else name
            results.append({
                "symbol": pretty_name,
                "price": round(data["last_price"], 2) if data.get("last_price") else 0.0,
                "change_pct": round(data["pct_change"], 2) if data.get("pct_change") else 0.0,
            })
    except Exception as e:
        logger.warning(f"Failed to fetch Upstox index overview: {e}")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    logger.info("Starting live prediction generation.")

    if SessionLocal is None:
        logger.critical("Database session unavailable. Check DATABASE_URL. Aborting.")
        sys.exit(1)

    # Initialize risk governance components (historical_var dropped as it is unused)
    circuit_breaker = CircuitBreaker()
    portfolio_risk = PortfolioRiskEngine(total_capital=10_000_000)

    logger.info("Risk governance components initialized")

    # Fetch macro features once — shared across all timeframes
    macro = extract_macro_features()
    logger.info(f"Macro: VIX={macro.get('vix_level')} regime={macro.get('market_regime')}")

    now = now_ist()
    db = SessionLocal()

    # 1. Store index ticks (isolated try/except to avoid blocking prediction generation)
    try:
        for idx in fetch_market_indices():
            db.add(IndexTick(
                timestamp=now,
                name=idx["symbol"],
                value=idx["price"],
                change=idx["change_pct"],
            ))
        db.commit()
        logger.info("Stored market index ticks.")
    except Exception as e:
        logger.error(f"Error storing market index ticks: {e}")
        db.rollback()

    # 2. Generate and store predictions timeframe by timeframe
    for timeframe, config in TIMEFRAME_CONFIG.items():
        try:
            model_version = get_model_version(
                config["model_version_key"],
                config["default_version"],
            )

            model_long = load_model(model_version + "_long", timeframe)
            model_short = load_model(model_version + "_short", timeframe)
            if model_long is None and model_short is None:
                # Try fallback to legacy unsuffixed version
                legacy_model = load_model(model_version, timeframe)
                if legacy_model is None:
                    continue
                model_long = legacy_model
                model_short = legacy_model

            preds = generate_predictions_for_timeframe(
                timeframe=timeframe,
                model_long=model_long,
                model_short=model_short,
                model_version=model_version,
                macro=macro,
                config=config,
                circuit_breaker=circuit_breaker,
                portfolio_risk=portfolio_risk,
                now=now,
                db=db,
            )

            if not preds:
                logger.info(f"No predictions generated for timeframe {timeframe}.")
                continue

            # Store predictions for this timeframe
            for sig in preds:
                # Idempotency guard: supersede existing OPEN predictions for (symbol, horizon)
                try:
                    existing_open = db.query(Prediction).filter(
                        Prediction.symbol == sig.symbol,
                        Prediction.horizon == sig.metadata["timeframe"],
                        Prediction.actual_outcome == "OPEN"
                    ).all()
                    for old_pred in existing_open:
                        old_pred.actual_outcome = "SUPERSEDED"
                except Exception as ex:
                    logger.error(f"Error running idempotency guard check: {ex}")

                # Safely parse features snapshot JSON
                try:
                    feat_snapshot = json.loads(sig.metadata["features_json"])
                except Exception:
                    feat_snapshot = {"error": "could not parse features_json"}

                try:
                    record = PredictionRecord(
                        symbol=sig.symbol,
                        prediction_time=now,
                        model_version=sig.model_version,
                        feature_schema_version=sig.metadata["feature_version"],
                        feature_snapshot=feat_snapshot,
                        direction="BUY" if sig.prediction == 2 else "SELL",
                        timeframe=sig.metadata["timeframe"],
                        win_probability=sig.win_probability,
                        confidence=sig.win_probability,
                        expected_return=sig.expected_return,
                        regime=feat_snapshot.get("market_regime"),
                        reason=sig.metadata["reason"],
                        entry_price=sig.metadata["entry_price"],
                        stop_loss=sig.stop_loss,
                        target_price=sig.target_price,
                        expiry_time=_expiry_time(sig.metadata["timeframe"], now),
                    )
                    _store.store(record, db)
                except Exception as ex:
                    logger.error(f"Prediction validation failed for {sig.symbol}: {ex}")


            db.commit()
            logger.info(f"Committed {len(preds)} predictions for timeframe {timeframe} to database.")

        except Exception as e:
            logger.error(f"Failed prediction run for timeframe {timeframe}: {e}")
            db.rollback()

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
