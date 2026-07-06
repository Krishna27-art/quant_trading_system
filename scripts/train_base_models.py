"""
scripts/train_base_models.py

One-time training script for all three timeframe models.
Run this before starting the live prediction loop.

    python scripts/train_base_models.py [--symbols RELIANCE TCS ...] [--model-dir /path]

Models saved:
    <MODEL_PATH>/meta_ensemble_longterm/
    <MODEL_PATH>/meta_ensemble_swing/
    <MODEL_PATH>/meta_ensemble_intraday/

After this script completes, ModelRegistry will load them automatically.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_platform.feature_store.macro import extract_historical_macro
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
from prediction_intelligence.meta_ensemble import MetaEnsemble
from utils.label_validator import LabelValidator
from utils.logger import get_logger

logger = get_logger("train_base_models")

from config.universe import NSE_UNIVERSE
DEFAULT_SYMBOLS = [s["symbol"] for s in NSE_UNIVERSE]


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
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    return df.set_index("timestamp", drop=False).sort_index()


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
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    return df.set_index("timestamp", drop=False).sort_index()


def _load_intraday(symbol: str) -> pd.DataFrame:
    # Use real 1-minute data (8-day limit) for true timescale parity
    raw = yf.download(f"{symbol}.NS", period="8d", interval="1m",
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
    return df.set_index("timestamp", drop=False).sort_index()


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
    logger.info("=== Training LOGREG_LONGTERM_v1 (Long & Short Models) ===")

    validator = LabelValidator()
    all_X_long: list[pd.DataFrame] = []
    all_X_short: list[pd.DataFrame] = []

    for sym in symbols:
        logger.info(f"  Fetching weekly data for {sym}...")
        df = _load_weekly(sym)
        if df.empty or len(df) < 60:
            logger.warning(f"  {sym}: insufficient weekly data — skipping")
            continue

        macro_df = extract_historical_macro(pd.DatetimeIndex(df["timestamp"]))
        extra_data = {col: macro_df[col] for col in macro_df.columns}
        feats = build_features(df, "LONGTERM", extra=extra_data)
        
        # Build long and short labels
        label_long = build_label(df, "LONGTERM", side="long")
        label_short = build_label(df, "LONGTERM", side="short")

        # Align long
        combined_long = feats[LONGTERM_FEATURES].copy()
        combined_long["__label__"] = label_long.values
        combined_long["__symbol__"] = sym
        combined_long["__date__"] = combined_long.index
        combined_long = combined_long.dropna()

        # Align short
        combined_short = feats[LONGTERM_FEATURES].copy()
        combined_short["__label__"] = label_short.values
        combined_short["__symbol__"] = sym
        combined_short["__date__"] = combined_short.index
        combined_short = combined_short.dropna()

        # Validate and append long
        if len(combined_long) >= 20:
            label_stats = validator.validate_label_distribution(combined_long["__label__"], min_samples=20)
            if label_stats["valid"]:
                all_X_long.append(combined_long)

        # Validate and append short
        if len(combined_short) >= 20:
            label_stats = validator.validate_label_distribution(combined_short["__label__"], min_samples=20)
            if label_stats["valid"]:
                all_X_short.append(combined_short)

    if not all_X_long or not all_X_short:
        logger.error("No training data collected for LONGTERM long/short — aborting")
        return

    # Train Long Model
    combined_all_long = pd.concat(all_X_long, ignore_index=True)
    combined_all_long = combined_all_long.sort_values("__date__").reset_index(drop=True)
    combined_all_long = combined_all_long.drop_duplicates(subset=["__symbol__", "__date__"]).reset_index(drop=True)

    X_long = combined_all_long[LONGTERM_FEATURES]
    y_long = combined_all_long["__label__"].astype(int)
    logger.info(f"Total LONGTERM LONG training rows: {len(X_long)}, win_rate={y_long.mean():.2%}")

    model_long = MetaEnsemble(timeframe="LONGTERM", model_dir=model_dir, feature_cols=LONGTERM_FEATURES)
    metrics_long = model_long.fit(X_long, y_long)
    model_long.save(os.path.join(model_dir, "meta_ensemble_longterm_long"))
    logger.info(f"LONGTERM Long MetaEnsemble saved. Metrics: {metrics_long}")

    # Train Short Model
    combined_all_short = pd.concat(all_X_short, ignore_index=True)
    combined_all_short = combined_all_short.sort_values("__date__").reset_index(drop=True)
    combined_all_short = combined_all_short.drop_duplicates(subset=["__symbol__", "__date__"]).reset_index(drop=True)

    X_short = combined_all_short[LONGTERM_FEATURES]
    y_short = combined_all_short["__label__"].astype(int)
    logger.info(f"Total LONGTERM SHORT training rows: {len(X_short)}, win_rate={y_short.mean():.2%}")

    model_short = MetaEnsemble(timeframe="LONGTERM", model_dir=model_dir, feature_cols=LONGTERM_FEATURES)
    metrics_short = model_short.fit(X_short, y_short)
    model_short.save(os.path.join(model_dir, "meta_ensemble_longterm_short"))
    logger.info(f"LONGTERM Short MetaEnsemble saved. Metrics: {metrics_short}")

    # Fit Imputer
    imputer = SimpleImputer(strategy="median").fit(X_long)
    ModelRegistry(model_dir=model_dir).save_imputer("LONGTERM", imputer)

    # Register in singleton
    ModelRegistry().register("META_LONGTERM_v1_long", "LONGTERM", model_long)
    ModelRegistry().register("META_LONGTERM_v1_short", "LONGTERM", model_short)


def train_swing(symbols: list[str], model_dir: str) -> None:
    logger.info("=== Training XGB_SWING_v1 (Long & Short Models) ===")

    validator = LabelValidator()
    all_X_long: list[pd.DataFrame] = []
    all_X_short: list[pd.DataFrame] = []

    for sym in symbols:
        logger.info(f"  Fetching daily data for {sym}...")
        df = _load_daily(sym, period="3y")
        if df.empty or len(df) < 80:
            logger.warning(f"  {sym}: insufficient daily data — skipping")
            continue

        macro_df = extract_historical_macro(pd.DatetimeIndex(df["timestamp"]))
        extra_data = {col: macro_df[col] for col in macro_df.columns}
        feats = build_features(df, "SWING", extra=extra_data)
        
        # Build long and short labels
        label_long = build_label(df, "SWING", side="long")
        label_short = build_label(df, "SWING", side="short")

        # Align long
        combined_long = feats[SWING_FEATURES].copy()
        combined_long["__label__"] = label_long.values
        combined_long["__symbol__"] = sym
        combined_long["__date__"] = combined_long.index
        combined_long = combined_long.dropna()

        # Align short
        combined_short = feats[SWING_FEATURES].copy()
        combined_short["__label__"] = label_short.values
        combined_short["__symbol__"] = sym
        combined_short["__date__"] = combined_short.index
        combined_short = combined_short.dropna()

        # Validate and append long
        if len(combined_long) >= 30:
            label_stats = validator.validate_label_distribution(combined_long["__label__"], min_samples=30)
            if label_stats["valid"]:
                all_X_long.append(combined_long)

        # Validate and append short
        if len(combined_short) >= 30:
            label_stats = validator.validate_label_distribution(combined_short["__label__"], min_samples=30)
            if label_stats["valid"]:
                all_X_short.append(combined_short)

    if not all_X_long or not all_X_short:
        logger.error("No training data collected for SWING long/short — aborting")
        return

    # Train Long Model
    combined_all_long = pd.concat(all_X_long, ignore_index=True)
    combined_all_long = combined_all_long.sort_values("__date__").reset_index(drop=True)
    combined_all_long = combined_all_long.drop_duplicates(subset=["__symbol__", "__date__"]).reset_index(drop=True)

    X_long = combined_all_long[SWING_FEATURES]
    y_long = combined_all_long["__label__"].astype(int)
    logger.info(f"Total SWING LONG training rows: {len(X_long)}, win_rate={y_long.mean():.2%}")

    model_long = MetaEnsemble(timeframe="SWING", model_dir=model_dir, feature_cols=SWING_FEATURES)
    metrics_long = model_long.fit(X_long, y_long)
    model_long.save(os.path.join(model_dir, "meta_ensemble_swing_long"))
    logger.info(f"SWING Long MetaEnsemble saved. Metrics: {metrics_long}")

    # Train Short Model
    combined_all_short = pd.concat(all_X_short, ignore_index=True)
    combined_all_short = combined_all_short.sort_values("__date__").reset_index(drop=True)
    combined_all_short = combined_all_short.drop_duplicates(subset=["__symbol__", "__date__"]).reset_index(drop=True)

    X_short = combined_all_short[SWING_FEATURES]
    y_short = combined_all_short["__label__"].astype(int)
    logger.info(f"Total SWING SHORT training rows: {len(X_short)}, win_rate={y_short.mean():.2%}")

    model_short = MetaEnsemble(timeframe="SWING", model_dir=model_dir, feature_cols=SWING_FEATURES)
    metrics_short = model_short.fit(X_short, y_short)
    model_short.save(os.path.join(model_dir, "meta_ensemble_swing_short"))
    logger.info(f"SWING Short MetaEnsemble saved. Metrics: {metrics_short}")

    # Fit Imputer
    imputer = SimpleImputer(strategy="median").fit(X_long)
    ModelRegistry(model_dir=model_dir).save_imputer("SWING", imputer)

    # Register in singleton
    ModelRegistry().register("META_SWING_v1_long", "SWING", model_long)
    ModelRegistry().register("META_SWING_v1_short", "SWING", model_short)


def train_intraday(symbols: list[str], model_dir: str) -> None:
    logger.info("=== Training LGBM_INTRADAY_v1 (Long & Short Models) ===")

    validator = LabelValidator()
    all_X_long: list[pd.DataFrame] = []
    all_X_short: list[pd.DataFrame] = []

    for sym in symbols:
        logger.info(f"  Fetching 1m data for {sym}...")
        df = _load_intraday(sym)
        if df.empty or len(df) < 50:
            logger.warning(f"  {sym}: insufficient intraday data — skipping")
            continue

        macro_df = extract_historical_macro(pd.DatetimeIndex(df["timestamp"]))
        extra_data = {col: macro_df[col] for col in macro_df.columns}
        feats = build_features(df, "INTRADAY", extra=extra_data)
        
        # Build long and short labels
        label_long = build_label(df, "INTRADAY", side="long")
        label_short = build_label(df, "INTRADAY", side="short")

        # Align long
        combined_long = feats[INTRADAY_FEATURES].copy()
        combined_long["__label__"] = label_long.values
        combined_long["__symbol__"] = sym
        combined_long["__date__"] = combined_long.index
        combined_long = combined_long.dropna()

        # Align short
        combined_short = feats[INTRADAY_FEATURES].copy()
        combined_short["__label__"] = label_short.values
        combined_short["__symbol__"] = sym
        combined_short["__date__"] = combined_short.index
        combined_short = combined_short.dropna()

        # Validate and append long
        if len(combined_long) >= 30:
            label_stats = validator.validate_label_distribution(combined_long["__label__"], min_samples=30)
            if label_stats["valid"]:
                all_X_long.append(combined_long)

        # Validate and append short
        if len(combined_short) >= 30:
            label_stats = validator.validate_label_distribution(combined_short["__label__"], min_samples=30)
            if label_stats["valid"]:
                all_X_short.append(combined_short)

    if not all_X_long or not all_X_short:
        logger.error("No training data collected for INTRADAY long/short — aborting")
        return

    # Train Long Model
    combined_all_long = pd.concat(all_X_long, ignore_index=True)
    combined_all_long = combined_all_long.sort_values("__date__").reset_index(drop=True)
    combined_all_long = combined_all_long.drop_duplicates(subset=["__symbol__", "__date__"]).reset_index(drop=True)

    X_long = combined_all_long[INTRADAY_FEATURES]
    y_long = combined_all_long["__label__"].astype(int)
    logger.info(f"Total INTRADAY LONG training rows: {len(X_long)}, win_rate={y_long.mean():.2%}")

    model_long = MetaEnsemble(timeframe="INTRADAY", model_dir=model_dir, feature_cols=INTRADAY_FEATURES)
    metrics_long = model_long.fit(X_long, y_long)
    model_long.save(os.path.join(model_dir, "meta_ensemble_intraday_long"))
    logger.info(f"INTRADAY Long MetaEnsemble saved. Metrics: {metrics_long}")

    # Train Short Model
    combined_all_short = pd.concat(all_X_short, ignore_index=True)
    combined_all_short = combined_all_short.sort_values("__date__").reset_index(drop=True)
    combined_all_short = combined_all_short.drop_duplicates(subset=["__symbol__", "__date__"]).reset_index(drop=True)

    X_short = combined_all_short[INTRADAY_FEATURES]
    y_short = combined_all_short["__label__"].astype(int)
    logger.info(f"Total INTRADAY SHORT training rows: {len(X_short)}, win_rate={y_short.mean():.2%}")

    model_short = MetaEnsemble(timeframe="INTRADAY", model_dir=model_dir, feature_cols=INTRADAY_FEATURES)
    metrics_short = model_short.fit(X_short, y_short)
    model_short.save(os.path.join(model_dir, "meta_ensemble_intraday_short"))
    logger.info(f"INTRADAY Short MetaEnsemble saved. Metrics: {metrics_short}")

    # Fit Imputer
    imputer = SimpleImputer(strategy="median").fit(X_long)
    ModelRegistry(model_dir=model_dir).save_imputer("INTRADAY", imputer)

    # Register in singleton
    ModelRegistry().register("META_INTRADAY_v1_long", "INTRADAY", model_long)
    ModelRegistry().register("META_INTRADAY_v1_short", "INTRADAY", model_short)


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