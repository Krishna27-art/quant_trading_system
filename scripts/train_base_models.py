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

def train_timeframe(symbols: list[str], timeframe: str, model_dir: str, version: str = "v1") -> None:
    tf = timeframe.upper()
    logger.info(f"=== Training {tf} (Long & Short Models - Version {version}) ===")

    validator = LabelValidator()
    all_X_long: list[pd.DataFrame] = []
    all_X_short: list[pd.DataFrame] = []

    # Map parameters per timeframe
    if tf == "LONGTERM":
        feature_cols = LONGTERM_FEATURES
        min_samples = 20
        load_fn = lambda sym: _load_weekly(sym)
    elif tf == "SWING":
        feature_cols = SWING_FEATURES
        min_samples = 30
        load_fn = lambda sym: _load_daily(sym, period="3y")
    elif tf == "INTRADAY":
        feature_cols = INTRADAY_FEATURES
        min_samples = 30
        load_fn = lambda sym: _load_intraday(sym)
    else:
        raise ValueError(f"Unknown timeframe: {timeframe}")

    for sym in symbols:
        logger.info(f"  Fetching {timeframe.lower()} data for {sym}...")
        df = load_fn(sym)
        if df.empty or len(df) < (60 if tf == "LONGTERM" else 80 if tf == "SWING" else 50):
            logger.warning(f"  {sym}: insufficient data — skipping")
            continue

        macro_df = extract_historical_macro(pd.DatetimeIndex(df["timestamp"]))
        extra_data = {col: macro_df[col] for col in macro_df.columns}

        # Merging real symbol-specific fundamentals for LONGTERM timeframe
        if tf == "LONGTERM":
            fund = _get_fundamentals(sym)
            extra_data.update(fund)

        feats = build_features(df, tf, extra=extra_data)
        
        # Build long and short labels
        label_long = build_label(df, tf, side="long")
        label_short = build_label(df, tf, side="short")

        # Align long
        combined_long = feats[feature_cols].copy()
        combined_long["__label__"] = label_long.values
        combined_long["__symbol__"] = sym
        combined_long["__date__"] = combined_long.index
        combined_long = combined_long.dropna()

        # Align short
        combined_short = feats[feature_cols].copy()
        combined_short["__label__"] = label_short.values
        combined_short["__symbol__"] = sym
        combined_short["__date__"] = combined_short.index
        combined_short = combined_short.dropna()

        # Validate and append long
        if len(combined_long) >= min_samples:
            label_stats = validator.validate_label_distribution(combined_long["__label__"], min_samples=min_samples)
            if label_stats["valid"]:
                all_X_long.append(combined_long)

        # Validate and append short
        if len(combined_short) >= min_samples:
            label_stats = validator.validate_label_distribution(combined_short["__label__"], min_samples=min_samples)
            if label_stats["valid"]:
                all_X_short.append(combined_short)

    if not all_X_long or not all_X_short:
        logger.error(f"No training data collected for {tf} long/short — aborting")
        return

    # Train Long Model
    combined_all_long = pd.concat(all_X_long, ignore_index=True)
    combined_all_long = combined_all_long.sort_values("__date__").reset_index(drop=True)
    combined_all_long = combined_all_long.drop_duplicates(subset=["__symbol__", "__date__"]).reset_index(drop=True)

    X_long = combined_all_long[feature_cols]
    y_long = combined_all_long["__label__"].astype(int)
    logger.info(f"Total {tf} LONG training rows: {len(X_long)}, win_rate={y_long.mean():.2%}")

    model_long = MetaEnsemble(timeframe=tf, model_dir=model_dir, feature_cols=feature_cols)
    metrics_long = model_long.fit(X_long, y_long)
    
    # Save both to the versioned and copy/symlink to current default directory
    versioned_long_path = os.path.join(model_dir, f"meta_ensemble_{tf.lower()}_long_{version}")
    model_long.save(versioned_long_path)
    # Also save to un-versioned directory for default pathing fallback
    model_long.save(os.path.join(model_dir, f"meta_ensemble_{tf.lower()}_long"))
    logger.info(f"{tf} Long MetaEnsemble saved. Metrics: {metrics_long}")

    # Train Short Model
    combined_all_short = pd.concat(all_X_short, ignore_index=True)
    combined_all_short = combined_all_short.sort_values("__date__").reset_index(drop=True)
    combined_all_short = combined_all_short.drop_duplicates(subset=["__symbol__", "__date__"]).reset_index(drop=True)

    X_short = combined_all_short[feature_cols]
    y_short = combined_all_short["__label__"].astype(int)
    logger.info(f"Total {tf} SHORT training rows: {len(X_short)}, win_rate={y_short.mean():.2%}")

    model_short = MetaEnsemble(timeframe=tf, model_dir=model_dir, feature_cols=feature_cols)
    metrics_short = model_short.fit(X_short, y_short)
    
    versioned_short_path = os.path.join(model_dir, f"meta_ensemble_{tf.lower()}_short_{version}")
    model_short.save(versioned_short_path)
    model_short.save(os.path.join(model_dir, f"meta_ensemble_{tf.lower()}_short"))
    logger.info(f"{tf} Short MetaEnsemble saved. Metrics: {metrics_short}")

    # Fit Imputer
    imputer = SimpleImputer(strategy="median").fit(X_long)
    ModelRegistry(model_dir=model_dir).save_imputer(tf, imputer)

    # Register in singleton
    ModelRegistry().register(f"META_{tf}_{version}_long", tf, model_long)
    ModelRegistry().register(f"META_{tf}_{version}_short", tf, model_short)

    # Log the experiment details
    try:
        from research_platform.experiments.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker()
        tracker.log_experiment(
            hypothesis_title=f"Stacking Ensemble for {tf} timeframe",
            params={
                "timeframe": tf,
                "symbols": symbols,
                "feature_cols": feature_cols,
                "min_samples": min_samples,
            },
            metrics={
                "long_metrics": metrics_long,
                "short_metrics": metrics_short,
            }
        )
        logger.info(f"Logged experiment successfully for {tf} timeframe")
    except Exception as te:
        logger.warning(f"Failed to log experiment: {te}")


def train_longterm(symbols: list[str], model_dir: str, version: str) -> None:
    train_timeframe(symbols, "LONGTERM", model_dir, version)


def train_swing(symbols: list[str], model_dir: str, version: str) -> None:
    train_timeframe(symbols, "SWING", model_dir, version)


def train_intraday(symbols: list[str], model_dir: str, version: str) -> None:
    train_timeframe(symbols, "INTRADAY", model_dir, version)


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
    parser.add_argument("--version", default="v1", help="Model version identifier (e.g. v1, v2)")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    logger.info(f"Model directory: {args.model_dir}")
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Version: {args.version}")

    if args.timeframe in ("all", "longterm"):
        train_longterm(args.symbols, args.model_dir, args.version)

    if args.timeframe in ("all", "swing"):
        train_swing(args.symbols, args.model_dir, args.version)

    if args.timeframe in ("all", "intraday"):
        train_intraday(args.symbols, args.model_dir, args.version)

    logger.info("Training complete. Run generate_live_predictions.py to start live signals.")


if __name__ == "__main__":
    main()