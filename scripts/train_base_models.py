"""
scripts/train_base_models.py

One-time training script for all three timeframe models.
Run this before starting the live prediction loop.

    python scripts/train_base_models.py [--symbols RELIANCE TCS ...] [--model-dir /path]

Models saved:
    <MODEL_PATH>/LOGREG_LONGTERM_v1.joblib
    <MODEL_PATH>/XGB_SWING_v1/          (EnsembleModel directory)
    <MODEL_PATH>/LGBM_INTRADAY_v1.joblib

After this script completes, ModelRegistry will load them automatically.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))

from prediction_intelligence.base_logistic import (
    BaseLogistic,
    EnsembleModel,
    ModelRegistry,
    build_features,
    build_label,
    INTRADAY_FEATURES,
    SWING_FEATURES,
    LONGTERM_FEATURES,
)
from utils.label_validator import LabelValidator
from utils.logger import get_logger

logger = get_logger("train_base_models")

DEFAULT_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK",
    "INFY", "TATAMOTORS", "ITC", "SBIN",
    "WIPRO", "AXISBANK", "KOTAKBANK", "LT",
]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_daily(symbol: str, period: str = "3y") -> pd.DataFrame:
    raw = yf.download(f"{symbol}.NS", period=period, progress=False, auto_adjust=True)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.reset_index().rename(columns={
        "Date": "timestamp", "Open": "open",
        "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    return df.sort_values("timestamp").reset_index(drop=True)


def _load_weekly(symbol: str, period: str = "5y") -> pd.DataFrame:
    raw = yf.download(f"{symbol}.NS", period=period, interval="1wk",
                      progress=False, auto_adjust=True)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.reset_index().rename(columns={
        "Date": "timestamp", "Open": "open",
        "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    return df.sort_values("timestamp").reset_index(drop=True)


def _load_intraday(symbol: str) -> pd.DataFrame:
    # yfinance caps 1m history at 8 days — use 60m as proxy for training
    raw = yf.download(f"{symbol}.NS", period="60d", interval="60m",
                      progress=False, auto_adjust=True)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.reset_index().rename(columns={
        "Datetime": "timestamp", "Open": "open",
        "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    return df.sort_values("timestamp").reset_index(drop=True)


def _get_fundamentals(symbol: str) -> dict[str, float]:
    try:
        info = yf.Ticker(f"{symbol}.NS").info
        return {
            "pe_ratio":       float(info.get("forwardPE", 20.0)),
            "debt_to_equity": float(info.get("debtToEquity", 50.0)) / 100.0,
        }
    except Exception:
        return {"pe_ratio": 20.0, "debt_to_equity": 0.5}


# ---------------------------------------------------------------------------
# Training runners
# ---------------------------------------------------------------------------

def train_longterm(symbols: list[str], model_dir: str) -> None:
    logger.info("=== Training LOGREG_LONGTERM_v1 ===")

    validator = LabelValidator()
    all_X: list[pd.DataFrame] = []
    all_y: list[pd.Series]    = []

    for sym in symbols:
        logger.info(f"  Fetching weekly data for {sym}...")
        df = _load_weekly(sym)
        if df.empty or len(df) < 60:
            logger.warning(f"  {sym}: insufficient weekly data — skipping")
            continue

        fundamentals = _get_fundamentals(sym)
        feats = build_features(df, "LONGTERM", extra=fundamentals)
        label = build_label(df, "LONGTERM")

        # Align and drop NaN rows
        combined = feats[LONGTERM_FEATURES].copy()
        combined["__label__"] = label.values
        combined = combined.dropna()

        if len(combined) < 20:
            logger.warning(f"  {sym}: too few clean rows ({len(combined)}) — skipping")
            continue

        # Validate label distribution before including in training set
        label_stats = validator.validate_label_distribution(combined["__label__"], min_samples=20)
        if not label_stats["valid"]:
            logger.warning(f"  {sym}: label validation failed — skipping")
            continue

        all_X.append(combined[LONGTERM_FEATURES])
        all_y.append(combined["__label__"].astype(int))
        logger.info(f"  {sym}: {len(combined)} training rows, "
                    f"win_rate={combined['__label__'].mean():.2%}")

    if not all_X:
        logger.error("No training data collected for LONGTERM — aborting")
        return

    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True)
    logger.info(f"Total LONGTERM training rows: {len(X)}, overall win_rate={y.mean():.2%}")

    # Final validation on combined dataset
    final_stats = validator.validate_label_distribution(y, min_samples=100)
    if not final_stats["valid"]:
        logger.error("Combined LONGTERM label validation failed — aborting")
        return

    model     = BaseLogistic(n_splits=5)
    save_path = os.path.join(model_dir, "LOGREG_LONGTERM_v1.joblib")
    metrics   = model.train(X, y, LONGTERM_FEATURES, save_path=save_path)
    logger.info(f"LONGTERM model saved. Metrics: {metrics}")

    # Register in singleton so downstream calls within same process pick it up
    ModelRegistry().register("LOGREG_LONGTERM_v1", "LONGTERM", model)


def train_swing(symbols: list[str], model_dir: str) -> None:
    logger.info("=== Training XGB_SWING_v1 (EnsembleModel) ===")

    validator = LabelValidator()
    all_X: list[pd.DataFrame] = []
    all_y: list[pd.Series]    = []

    for sym in symbols:
        logger.info(f"  Fetching daily data for {sym}...")
        df = _load_daily(sym, period="3y")
        if df.empty or len(df) < 80:
            logger.warning(f"  {sym}: insufficient daily data — skipping")
            continue

        feats = build_features(df, "SWING", extra={"vix": 15.0, "nifty_pcr": 1.0})
        label = build_label(df, "SWING")

        combined = feats[SWING_FEATURES].copy()
        combined["__label__"] = label.values
        combined = combined.dropna()

        if len(combined) < 30:
            continue

        # Validate label distribution before including in training set
        label_stats = validator.validate_label_distribution(combined["__label__"], min_samples=30)
        if not label_stats["valid"]:
            logger.warning(f"  {sym}: label validation failed — skipping")
            continue

        all_X.append(combined[SWING_FEATURES])
        all_y.append(combined["__label__"].astype(int))
        logger.info(f"  {sym}: {len(combined)} rows, win_rate={combined['__label__'].mean():.2%}")

    if not all_X:
        logger.error("No training data collected for SWING — aborting")
        return

    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True)
    logger.info(f"Total SWING training rows: {len(X)}, overall win_rate={y.mean():.2%}")

    # Final validation on combined dataset
    final_stats = validator.validate_label_distribution(y, min_samples=100)
    if not final_stats["valid"]:
        logger.error("Combined SWING label validation failed — aborting")
        return

    model    = EnsembleModel(feature_cols=SWING_FEATURES)
    save_dir = os.path.join(model_dir, "XGB_SWING_v1")
    model.train(X, y)
    model.save(save_dir)
    logger.info(f"SWING ensemble saved → {save_dir}")

    ModelRegistry().register("XGB_SWING_v1", "SWING", model)


def train_intraday(symbols: list[str], model_dir: str) -> None:
    logger.info("=== Training LGBM_INTRADAY_v1 ===")

    validator = LabelValidator()
    all_X: list[pd.DataFrame] = []
    all_y: list[pd.Series]    = []

    for sym in symbols:
        logger.info(f"  Fetching 60m data for {sym} (proxy for intraday)...")
        df = _load_intraday(sym)
        if df.empty or len(df) < 50:
            logger.warning(f"  {sym}: insufficient intraday data — skipping")
            continue

        feats = build_features(df, "INTRADAY", extra={"vix": 15.0})
        label = build_label(df, "INTRADAY")

        combined = feats[INTRADAY_FEATURES].copy()
        combined["__label__"] = label.values
        combined = combined.dropna()

        if len(combined) < 30:
            continue

        # Validate label distribution before including in training set
        label_stats = validator.validate_label_distribution(combined["__label__"], min_samples=30)
        if not label_stats["valid"]:
            logger.warning(f"  {sym}: label validation failed — skipping")
            continue

        all_X.append(combined[INTRADAY_FEATURES])
        all_y.append(combined["__label__"].astype(int))
        logger.info(f"  {sym}: {len(combined)} rows, win_rate={combined['__label__'].mean():.2%}")

    if not all_X:
        logger.error("No training data collected for INTRADAY — aborting")
        return

    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True)
    logger.info(f"Total INTRADAY training rows: {len(X)}, overall win_rate={y.mean():.2%}")

    # Final validation on combined dataset
    final_stats = validator.validate_label_distribution(y, min_samples=100)
    if not final_stats["valid"]:
        logger.error("Combined INTRADAY label validation failed — aborting")
        return

    model     = BaseLogistic(n_splits=5)
    save_path = os.path.join(model_dir, "LGBM_INTRADAY_v1.joblib")
    metrics   = model.train(X, y, INTRADAY_FEATURES, save_path=save_path)
    logger.info(f"INTRADAY model saved. Metrics: {metrics}")

    ModelRegistry().register("LGBM_INTRADAY_v1", "INTRADAY", model)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train all base prediction models")
    parser.add_argument("--symbols",   nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--model-dir", default=os.environ.get(
        "MODEL_PATH",
        str(Path(__file__).parent.parent / "data" / "production" / "models"),
    ))
    parser.add_argument("--timeframe", choices=["all", "intraday", "swing", "longterm"],
                        default="all")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    logger.info(f"Model directory: {args.model_dir}")
    logger.info(f"Symbols: {args.symbols}")

    if args.timeframe in ("all", "longterm"):
        train_longterm(args.symbols, args.model_dir)

    if args.timeframe in ("all", "swing"):
        train_swing(args.symbols, args.model_dir)

    if args.timeframe in ("all", "intraday"):
        train_intraday(args.symbols, args.model_dir)

    logger.info("Training complete. Run generate_live_predictions.py to start live signals.")


if __name__ == "__main__":
    main()