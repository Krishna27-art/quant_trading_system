"""
Triple-Barrier Labeling

Institutional-grade triple-barrier labeling for ML training.
Follows the methodology from Lopez de Prado's "Advances in Financial Machine Learning".

Labels are generated based on which barrier is hit first:
- Upper barrier (take-profit): label = 1 (long win)
- Lower barrier (stop-loss): label = -1 (long loss)
- Time expiration (vertical barrier): label = 0 (no clear direction)

This implementation prevents lookahead bias by only using information
available at the time of label generation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from prediction_intelligence.label_models import Label, LabelType
from utils.label_validator import LabelValidator
from utils.logger import get_logger

logger = get_logger("triple_barrier")


class TripleBarrierLabeler:
    """
    Triple-barrier labeling for financial time series.

    Generates labels based on which barrier is hit first:
    - Upper barrier (take-profit)
    - Lower barrier (stop-loss)
    - Vertical barrier (time horizon)
    """

    def __init__(
        self,
        upper_barrier_pct: float = 0.015,  # 1.5% take-profit
        lower_barrier_pct: float = -0.02,  # -2% stop-loss
        vertical_barrier_days: int = 5,  # 5-day horizon
        validate_labels: bool = True,  # Hard-fail on validation errors
    ):
        """
        Initialize the triple-barrier labeler.

        Args:
            upper_barrier_pct: Take-profit threshold as positive percentage
            lower_barrier_pct: Stop-loss threshold as negative percentage
            vertical_barrier_days: Time horizon in days
            validate_labels: If True, validate all labels and raise on failure
        """
        self.upper_barrier_pct = upper_barrier_pct
        self.lower_barrier_pct = lower_barrier_pct
        self.vertical_barrier_days = vertical_barrier_days
        self.validate_labels = validate_labels
        self.validator = LabelValidator()

        logger.info(
            f"TripleBarrierLabeler initialized: upper={upper_barrier_pct:.2%}, "
            f"lower={lower_barrier_pct:.2%}, horizon={vertical_barrier_days}d, "
            f"validate={validate_labels}"
        )

    def compute_labels(
        self,
        prices: pd.Series,
        timestamps: pd.DatetimeIndex,
        symbol: str,
        source: str = "triple_barrier_v1",
        version: str = "tb_v1.0",
        ingestion_job: str = "unspecified_job",
    ) -> list[Label]:
        """
        Compute triple-barrier labels for a price series.

        Args:
            prices: Series of prices (typically close prices) or a DataFrame containing 'close', 'high', 'low'
            timestamps: DatetimeIndex corresponding to prices
            symbol: Instrument symbol
            source: Data source identifier
            version: Labeling logic version
            ingestion_job: Job identifier

        Returns:
            List of Label objects with triple-barrier direction labels
        """
        if len(prices) < self.vertical_barrier_days + 1:
            logger.warning(
                f"Insufficient data for {symbol}: {len(prices)} < "
                f"{self.vertical_barrier_days + 1} required"
            )
            return []

        # Determine if we have a DataFrame with high/low columns or a Series
        if isinstance(prices, pd.DataFrame):
            close_prices = prices["close"]
            high_prices = prices["high"] if "high" in prices.columns else prices["close"]
            low_prices = prices["low"] if "low" in prices.columns else prices["close"]
        else:
            close_prices = prices
            high_prices = prices
            low_prices = prices

        labels = []
        now = pd.Timestamp.now()

        for i in range(len(close_prices) - self.vertical_barrier_days):
            entry_price = close_prices.iloc[i]
            entry_time = timestamps[i]

            # Look ahead to see which barrier is hit first
            future_closes = close_prices.iloc[i + 1 : i + 1 + self.vertical_barrier_days]
            future_highs = high_prices.iloc[i + 1 : i + 1 + self.vertical_barrier_days]
            future_lows = low_prices.iloc[i + 1 : i + 1 + self.vertical_barrier_days]
            future_times = timestamps[i + 1 : i + 1 + self.vertical_barrier_days]

            if len(future_closes) == 0:
                continue

            # Calculate barriers
            upper_barrier = entry_price * (1 + self.upper_barrier_pct)
            lower_barrier = entry_price * (1 + self.lower_barrier_pct)

            # Check which barrier is hit first
            label_value = 0  # Default: vertical barrier (time expiration)
            exit_price = future_closes.iloc[-1]  # Price at vertical barrier
            exit_time = future_times.values[-1]
            exit_reason = "vertical"

            for j in range(len(future_closes)):
                bar_close = future_closes.iloc[j]
                bar_high = future_highs.iloc[j]
                bar_low = future_lows.iloc[j]
                bar_time = future_times.values[j]

                # Stop loss check (breaching lower barrier)
                if bar_low <= lower_barrier:
                    label_value = -1
                    exit_price = lower_barrier
                    exit_time = bar_time
                    exit_reason = "lower"
                    break
                # Take profit check (breaching upper barrier)
                elif bar_high >= upper_barrier:
                    label_value = 1
                    exit_price = upper_barrier
                    exit_time = bar_time
                    exit_reason = "upper"
                    break

            # Calculate actual return
            actual_return = (exit_price - entry_price) / entry_price

            # Calculate MFE (Maximum Favorable Excursion) and MAE (Maximum Adverse Excursion)
            if label_value == 1:  # Won
                actual_mfe = actual_return
                actual_mae = min(0, (future_lows.min() - entry_price) / entry_price)
            elif label_value == -1:  # Lost
                actual_mfe = max(0, (future_highs.max() - entry_price) / entry_price)
                actual_mae = actual_return
            else:  # Time expiration
                actual_mfe = max(0, (future_highs.max() - entry_price) / entry_price)
                actual_mae = min(0, (future_lows.min() - entry_price) / entry_price)

            # Create Label object with full PIT chain
            try:
                label = Label(
                    symbol=symbol,
                    label_type=LabelType.TRIPLE_BARRIER_DIRECTION,
                    label_value=float(label_value),
                    label_date=entry_time,
                    horizon_days=self.vertical_barrier_days,
                    event_time=entry_time,
                    publication_time=entry_time,  # Same bar for now
                    effective_time=entry_time,
                    ingestion_time=now,
                    source=source,
                    version=version,
                    ingestion_job=ingestion_job,
                    entry_price=float(entry_price),
                    target_price=float(upper_barrier),
                    stop_loss_price=float(lower_barrier),
                    actual_return=float(actual_return),
                    actual_mfe=float(actual_mfe),
                    actual_mae=float(actual_mae),
                    actual_duration_bars=j + 1 if exit_reason != "vertical" else self.vertical_barrier_days,
                )
                labels.append(label)
            except Exception as e:
                logger.error(f"Failed to create label for {symbol} @ {entry_time}: {e}")

        logger.info(f"Generated {len(labels)} triple-barrier labels for {symbol}")

        # Validate labels if enabled (hard failure on validation errors)
        if self.validate_labels and labels:
            valid_labels, report = self.validator.validate_and_filter(labels)
            if report["invalid"] > 0:
                error_msg = (
                    f"Label validation failed for {symbol}: "
                    f"{report['invalid']}/{report['total']} invalid labels. "
                    f"Errors: {report['errors'][:5]}"  # Show first 5 errors
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            logger.info(f"Label validation passed for {symbol}: {report['valid']}/{report['total']} valid")
            return valid_labels

        return labels

    def compute_labels_from_dataframe(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        date_col: str = "date",
        symbol_col: str = "symbol",
        **kwargs: Any,
    ) -> list[Label]:
        """
        Compute labels from a DataFrame with OHLCV data.

        Args:
            df: DataFrame with price data
            price_col: Column name for price (typically 'close')
            date_col: Column name for date
            symbol_col: Column name for symbol
            **kwargs: Additional arguments passed to compute_labels

        Returns:
            List of Label objects
        """
        if symbol_col not in df.columns:
            symbol = kwargs.get("symbol", "UNKNOWN")
        else:
            symbol = df[symbol_col].iloc[0]

        prices = df[price_col]
        timestamps = pd.to_datetime(df[date_col])

        return self.compute_labels(prices, timestamps, symbol, **kwargs)
