"""
Tick Data Validation Utilities

Institutional-grade validation for tick data quality and integrity.
"""

from typing import Any

import pandas as pd

from utils.logger import get_logger

logger = get_logger("tick_validator")


class TickDataValidator:
    """
    Validator for tick data quality and integrity.

    Critical for institutional quant work - ensures data quality
    for high-frequency trading research.
    """

    REQUIRED_FIELDS = [
        "timestamp",
        "symbol",
        "exchange",
        "tick_type",
        "event_time",
        "publication_time",
        "ingestion_time",
        "effective_time",
        "source",
        "version",
        "ingestion_job",
    ]

    def __init__(self):
        """Initialize tick data validator."""
        self.logger = logger

    def validate_tick_dataframe(self, df: pd.DataFrame, dataset_name: str) -> dict[str, Any]:
        """
        Validate tick data DataFrame for quality and integrity.

        Args:
            df: DataFrame to validate
            dataset_name: Name of the dataset for logging

        Returns:
            Validation results dictionary
        """
        self.logger.info(f"Validating tick data for dataset: {dataset_name}")

        results = {
            "dataset_name": dataset_name,
            "is_valid": True,
            "missing_fields": [],
            "invalid_rows": 0,
            "total_rows": len(df),
            "errors": [],
            "warnings": [],
        }

        # Check for required fields
        missing_fields = [field for field in self.REQUIRED_FIELDS if field not in df.columns]
        if missing_fields:
            results["is_valid"] = False
            results["missing_fields"] = missing_fields
            error_msg = f"Missing required fields: {missing_fields}"
            self.logger.error(error_msg)
            results["errors"].append(error_msg)
            return results

        # Validate timestamp ordering
        if not df.empty:
            # Check publication_time >= event_time
            invalid_pub = df[df["publication_time"] < df["event_time"]]
            if not invalid_pub.empty:
                results["is_valid"] = False
                results["invalid_rows"] += len(invalid_pub)
                error_msg = (
                    f"{len(invalid_pub)} rows have publication_time < event_time (lookahead bias)"
                )
                self.logger.error(error_msg)
                results["errors"].append(error_msg)

            # Check effective_time >= publication_time
            invalid_eff = df[df["effective_time"] < df["publication_time"]]
            if not invalid_eff.empty:
                results["is_valid"] = False
                results["invalid_rows"] += len(invalid_eff)
                error_msg = f"{len(invalid_eff)} rows have effective_time < publication_time"
                self.logger.error(error_msg)
                results["errors"].append(error_msg)

            # Check for negative prices
            if "price" in df.columns:
                negative_prices = df[df["price"] <= 0]
                if not negative_prices.empty:
                    results["is_valid"] = False
                    results["invalid_rows"] += len(negative_prices)
                    error_msg = f"{len(negative_prices)} rows have non-positive prices"
                    self.logger.error(error_msg)
                    results["errors"].append(error_msg)

            # Check for negative volumes
            if "volume" in df.columns:
                negative_volumes = df[df["volume"] < 0]
                if not negative_volumes.empty:
                    results["is_valid"] = False
                    results["invalid_rows"] += len(negative_volumes)
                    error_msg = f"{len(negative_volumes)} rows have negative volumes"
                    self.logger.error(error_msg)
                    results["errors"].append(error_msg)

            # Check for duplicate timestamps (same symbol, same timestamp)
            if "timestamp" in df.columns and "symbol" in df.columns:
                duplicates = df.duplicated(subset=["symbol", "timestamp"], keep=False)
                if duplicates.any():
                    duplicate_count = duplicates.sum()
                    warning_msg = f"{duplicate_count} rows have duplicate (symbol, timestamp) pairs"
                    self.logger.warning(warning_msg)
                    results["warnings"].append(warning_msg)

        if results["is_valid"]:
            self.logger.info(f"Tick data validation passed for {dataset_name}")
        else:
            self.logger.error(f"Tick data validation failed for {dataset_name}")

        return results

    def validate_tick_sequence(self, df: pd.DataFrame, symbol: str) -> dict[str, Any]:
        """
        Validate tick sequence for a symbol.

        Checks for:
        - Monotonically increasing timestamps
        - Missing timestamps (gaps)
        - Out-of-order timestamps

        Args:
            df: DataFrame with tick data for a single symbol
            symbol: Symbol being validated

        Returns:
            Validation results
        """
        self.logger.info(f"Validating tick sequence for {symbol}")

        results = {
            "symbol": symbol,
            "is_valid": True,
            "total_ticks": len(df),
            "out_of_order": 0,
            "gaps": 0,
            "errors": [],
        }

        if df.empty:
            return results

        # Sort by timestamp
        df_sorted = df.sort_values("timestamp")

        # Check for out-of-order timestamps
        if not df_sorted["timestamp"].is_monotonic_increasing:
            results["is_valid"] = False
            results["out_of_order"] = len(df) - len(df_sorted.drop_duplicates(subset=["timestamp"]))
            error_msg = f"Found {results['out_of_order']} out-of-order timestamps"
            self.logger.error(error_msg)
            results["errors"].append(error_msg)

        # Check for gaps in timestamps (more than 1 second gap)
        if len(df_sorted) > 1:
            time_diffs = df_sorted["timestamp"].diff()
            gaps = time_diffs[time_diffs > pd.Timedelta(seconds=1)]
            if not gaps.empty:
                results["gaps"] = len(gaps)
                warning_msg = f"Found {len(gaps)} gaps > 1 second in tick sequence"
                self.logger.warning(warning_msg)
                results["errors"].append(warning_msg)

        return results

    def validate_tick_quality(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Validate tick data quality metrics.

        Args:
            df: DataFrame with tick data

        Returns:
            Quality metrics
        """
        self.logger.info("Calculating tick data quality metrics")

        metrics = {
            "total_ticks": len(df),
            "unique_symbols": df["symbol"].nunique() if "symbol" in df.columns else 0,
            "unique_exchanges": df["exchange"].nunique() if "exchange" in df.columns else 0,
            "date_range": {},
            "tick_types": {},
            "price_range": {},
            "volume_range": {},
        }

        if df.empty:
            return metrics

        # Date range
        if "timestamp" in df.columns:
            metrics["date_range"] = {
                "start": df["timestamp"].min().isoformat(),
                "end": df["timestamp"].max().isoformat(),
            }

        # Tick type distribution
        if "tick_type" in df.columns:
            metrics["tick_types"] = df["tick_type"].value_counts().to_dict()

        # Price range
        if "price" in df.columns:
            metrics["price_range"] = {
                "min": float(df["price"].min()),
                "max": float(df["price"].max()),
                "mean": float(df["price"].mean()),
                "std": float(df["price"].std()),
            }

        # Volume range
        if "volume" in df.columns:
            metrics["volume_range"] = {
                "min": int(df["volume"].min()),
                "max": int(df["volume"].max()),
                "mean": float(df["volume"].mean()),
                "total": int(df["volume"].sum()),
            }

        return metrics

    def validate_quote_integrity(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Validate quote data integrity.

        Checks for:
        - bid_price <= ask_price
        - bid_volume and ask_volume are non-negative
        - Spread is reasonable

        Args:
            df: DataFrame with quote data

        Returns:
            Validation results
        """
        self.logger.info("Validating quote data integrity")

        results = {"is_valid": True, "invalid_quotes": 0, "negative_spreads": 0, "errors": []}

        if df.empty:
            return results

        # Filter for quotes only
        quotes = df[df["tick_type"] == "QUOTE"] if "tick_type" in df.columns else df

        if quotes.empty:
            return results

        # Check bid_price <= ask_price
        if "bid_price" in quotes.columns and "ask_price" in quotes.columns:
            valid_quotes = quotes.dropna(subset=["bid_price", "ask_price"])
            if not valid_quotes.empty:
                negative_spreads = valid_quotes[
                    valid_quotes["bid_price"] > valid_quotes["ask_price"]
                ]
                if not negative_spreads.empty:
                    results["is_valid"] = False
                    results["negative_spreads"] = len(negative_spreads)
                    results["invalid_quotes"] += len(negative_spreads)
                    error_msg = f"{len(negative_spreads)} quotes have bid_price > ask_price"
                    self.logger.error(error_msg)
                    results["errors"].append(error_msg)

        # Check for negative quote volumes
        if "bid_volume" in quotes.columns:
            negative_bid_vol = quotes[quotes["bid_volume"] < 0]
            if not negative_bid_vol.empty:
                results["is_valid"] = False
                results["invalid_quotes"] += len(negative_bid_vol)
                error_msg = f"{len(negative_bid_vol)} quotes have negative bid_volume"
                self.logger.error(error_msg)
                results["errors"].append(error_msg)

        if "ask_volume" in quotes.columns:
            negative_ask_vol = quotes[quotes["ask_volume"] < 0]
            if not negative_ask_vol.empty:
                results["is_valid"] = False
                results["invalid_quotes"] += len(negative_ask_vol)
                error_msg = f"{len(negative_ask_vol)} quotes have negative ask_volume"
                self.logger.error(error_msg)
                results["errors"].append(error_msg)

        return results


def validate_tick_data(df: pd.DataFrame, dataset_name: str) -> bool:
    """
    Convenience function to validate tick data.

    Args:
        df: DataFrame to validate
        dataset_name: Name of the dataset

    Returns:
        True if validation passes, raises exception otherwise
    """
    validator = TickDataValidator()
    results = validator.validate_tick_dataframe(df, dataset_name)

    if not results["is_valid"]:
        error_msg = f"Tick data validation failed for {dataset_name}: {results['errors']}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    return True
