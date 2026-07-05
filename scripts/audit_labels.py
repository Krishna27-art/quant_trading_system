"""
scripts/audit_labels.py

Standalone label validation audit script.
Runs label validation on training data to detect lookahead bias, distribution skew,
or timeline inconsistencies, without breaking the training pipeline.
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import yfinance as yf

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prediction_intelligence.triple_barrier import TripleBarrierLabeler
from utils.label_validator import LabelValidator
from utils.logger import get_logger

logger = get_logger("label_audit")


def fetch_symbol_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """Fetch matching timeframe data from yfinance."""
    tf = timeframe.upper()
    try:
        if tf == "INTRADAY":
            raw = yf.download(f"{symbol}.NS", period="8d", interval="1m", progress=False, auto_adjust=True)
            if raw.empty:
                return pd.DataFrame()
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            df = raw.reset_index().rename(columns={"Datetime": "timestamp", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
            return df.sort_values("timestamp").reset_index(drop=True)
        elif tf == "SWING":
            raw = yf.download(f"{symbol}.NS", period="1y", progress=False, auto_adjust=True)
            if raw.empty:
                return pd.DataFrame()
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            df = raw.reset_index().rename(columns={"Date": "timestamp", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
            return df.sort_values("timestamp").reset_index(drop=True)
        else:
            raw = yf.download(f"{symbol}.NS", period="3y", interval="1wk", progress=False, auto_adjust=True)
            if raw.empty:
                return pd.DataFrame()
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            df = raw.reset_index().rename(columns={"Date": "timestamp", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
            return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        logger.error(f"Failed to fetch data for {symbol} ({tf}): {e}")
        return pd.DataFrame()


def audit_label_integrity(symbols: list[str], timeframes: list[str]):
    """Run non-blocking label validation audits."""
    validator = LabelValidator()

    _defaults = {
        "INTRADAY": (0.015, -0.0075, 10),
        "SWING":    (0.030, -0.015, 15),
        "LONGTERM": (0.200, -0.100, 12),
    }

    logger.info(f"Starting Label Validation Audit for {len(symbols)} symbols...")
    total_audits = 0
    total_failed_symbols = 0

    for tf in timeframes:
        tp, sl, horizon = _defaults.get(tf.upper(), (0.05, -0.025, 10))
        labeler = TripleBarrierLabeler(
            upper_barrier_pct=tp,
            lower_barrier_pct=sl,
            vertical_barrier_days=horizon,
            validate_labels=False,  # Set False so compute_labels returns labels for us to audit manually
        )

        for sym in symbols:
            df = fetch_symbol_data(sym, tf)
            if df.empty or len(df) < horizon + 5:
                logger.warning(f"[{tf}] {sym}: insufficient data for audit")
                continue

            timestamps = pd.to_datetime(df["timestamp"])
            labels = labeler.compute_labels(
                prices=df["close"],
                timestamps=timestamps,
                symbol=sym,
            )

            if not labels:
                logger.warning(f"[{tf}] {sym}: generated 0 labels")
                continue

            total_audits += 1
            valid_labels, report = validator.validate_and_filter(labels)

            # Check skewness
            label_vals = [l.label_value for l in valid_labels]
            pos_ratio = sum(1 for v in label_vals if v == 1) / len(label_vals) if label_vals else 0.0

            if report["invalid"] > 0:
                logger.error(
                    f"[{tf}] {sym}: FAILED label audit! "
                    f"{report['invalid']}/{report['total']} invalid labels. "
                    f"Errors: {report['errors'][:3]}"
                )
                total_failed_symbols += 1
            else:
                logger.info(
                    f"[{tf}] {sym}: PASSED label audit. "
                    f"{report['valid']} valid labels. Pos/Neg Ratio: {pos_ratio:.2%}/{1 - pos_ratio:.2%}"
                )

    logger.info(f"Audit complete. Processed {total_audits} audits. Failed symbols: {total_failed_symbols}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit labels for quantitative research")
    parser.add_argument("--symbols", nargs="+", default=["RELIANCE", "TCS", "HDFCBANK"], help="Symbols to audit")
    parser.add_argument("--timeframes", nargs="+", default=["INTRADAY", "SWING", "LONGTERM"], help="Timeframes to audit")
    args = parser.parse_args()

    audit_label_integrity(args.symbols, args.timeframes)
