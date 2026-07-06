"""
Canonical Feature Builder Service

Single source of truth for all feature engineering mathematics across training,
backtesting, and live inference. Guarantees identical feature transformations
and prevents lookahead bias.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("canonical_builder")

LONGTERM_FEATURES = [
    "ma50_slope",
    "rsi_14w",
    "vol_ratio",
    "price_to_52w_high",
    "pe_ratio",
    "debt_to_equity",
    "vix",
]

SWING_FEATURES = [
    "z_score_20d",
    "rsi_14d",
    "ma20_slope",
    "atr_pct",
    "volume_ratio",
    "vix",
    "nifty_pcr",
]

INTRADAY_FEATURES = [
    "vwap_dist",
    "rsi_14m",
    "vol_ratio_1m",
    "range_pct",
    "momentum_5m",
    "vix",
]

TIMEFRAME_FEATURES = {
    "INTRADAY": INTRADAY_FEATURES,
    "SWING": SWING_FEATURES,
    "LONGTERM": LONGTERM_FEATURES,
}


class CanonicalFeatureBuilder:
    """
    Immutable service class providing standardized feature transformations.
    """

    @classmethod
    def get_feature_cols(cls, timeframe: str) -> list[str]:
        tf = timeframe.upper()
        if tf not in TIMEFRAME_FEATURES:
            raise ValueError(f"Unknown timeframe: {timeframe!r}. Expected INTRADAY, SWING, or LONGTERM.")
        return TIMEFRAME_FEATURES[tf]

    @classmethod
    def _rsi(cls, series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/window, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/window, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @classmethod
    def build_features(
        cls,
        df: pd.DataFrame,
        timeframe: str,
        extra: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Build a feature DataFrame from an OHLCV candle DataFrame.
        Supports both scalar dictionaries and historical pandas Series/DataFrames for `extra`.
        """
        extra = extra or {}
        tf = timeframe.upper()
        out = pd.DataFrame(index=df.index)

        if df.empty:
            return out

        close = df["close"]
        volume = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
        high = df["high"]
        low = df["low"]
        open_ = df["open"]

        # Helper to align extra variables (scalar or Series) to output index
        def _get_extra_col(key: str, default_val: float, alt_keys: list[str] | None = None) -> pd.Series | float:
            keys = [key] + (alt_keys or [])
            for k in keys:
                if k in extra:
                    val = extra[k]
                    if isinstance(val, (pd.Series, pd.DataFrame)):
                        if isinstance(val, pd.DataFrame):
                            val = val[val.columns[0]]
                        # Align series to candle index via reindex / ffill
                        aligned = val.reindex(df.index, method="ffill")
                        return aligned.fillna(default_val)
                    elif isinstance(val, dict):
                        return float(val.get(k, default_val))
                    else:
                        return float(val)
            return default_val

        # Resolve dates for session-based grouping (handles both DatetimeIndex and RangeIndex)
        if isinstance(df.index, pd.DatetimeIndex):
            dates = df.index.date
        elif "timestamp" in df.columns:
            dates = pd.to_datetime(df["timestamp"]).dt.date.values
        else:
            dates = np.zeros(len(df))

        if tf == "INTRADAY":
            cum_vol = volume.groupby(dates).cumsum()
            cum_val = (close * volume).groupby(dates).cumsum()
            vwap = cum_val / cum_vol.replace(0, np.nan)
            out["vwap_dist"] = (close - vwap) / vwap.replace(0, np.nan)
            out["rsi_14m"] = cls._rsi(close, 14)
            vol_avg = volume.rolling(20).mean()
            out["vol_ratio_1m"] = volume / vol_avg.replace(0, np.nan)
            out["range_pct"] = (high - low) / open_.replace(0, np.nan)
            out["momentum_5m"] = close.pct_change(5)
            out["vix"] = _get_extra_col("vix", 15.0, alt_keys=["vix_level", "^INDIAVIX"])

        elif tf == "SWING":
            ma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            out["z_score_20d"] = (close - ma20) / std20.replace(0, np.nan)
            out["rsi_14d"] = cls._rsi(close, 14)
            out["ma20_slope"] = ma20.diff(3) / ma20.shift(3).replace(0, np.nan)
            prev_close = close.shift(1)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            out["atr_pct"] = atr / close.replace(0, np.nan)
            vol_avg = volume.rolling(20).mean()
            out["volume_ratio"] = volume / vol_avg.replace(0, np.nan)
            out["vix"] = _get_extra_col("vix", 15.0, alt_keys=["vix_level", "^INDIAVIX"])
            out["nifty_pcr"] = _get_extra_col("nifty_pcr", 1.0, alt_keys=["pcr"])

        elif tf == "LONGTERM":
            ma50 = close.rolling(50).mean()
            out["ma50_slope"] = ma50.diff(5) / ma50.shift(5).replace(0, np.nan)
            out["rsi_14w"] = cls._rsi(close, 14)
            short_vol = close.pct_change().rolling(20).std()
            long_vol = close.pct_change().rolling(100).std()
            out["vol_ratio"] = short_vol / long_vol.replace(0, np.nan)
            rolling_max = close.rolling(52).max()
            out["price_to_52w_high"] = close / rolling_max.replace(0, np.nan)
            out["pe_ratio"] = _get_extra_col("pe_ratio", 20.0)
            out["debt_to_equity"] = _get_extra_col("debt_to_equity", 0.5)
            out["vix"] = _get_extra_col("vix", 15.0, alt_keys=["vix_level", "^INDIAVIX"])

        else:
            raise ValueError(f"Unknown timeframe: {timeframe!r}. Expected INTRADAY, SWING, or LONGTERM.")

        return out

    @classmethod
    def validate_schema(cls, df: pd.DataFrame, timeframe: str) -> bool:
        """
        Validate that the built DataFrame contains all required feature columns
        for the given timeframe and is not empty.
        """
        if df.empty:
            logger.error("Validation failed: empty feature DataFrame")
            return False
        required = cls.get_feature_cols(timeframe)
        missing = set(required) - set(df.columns)
        if missing:
            logger.error(f"Validation failed for {timeframe}: missing required feature columns: {missing}")
            return False
        return True
