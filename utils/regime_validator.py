"""
Market Regime Validation Utilities

Institutional-grade validation for market regime data quality and integrity.
"""

from typing import Any

import pandas as pd

from utils.logger import get_logger

logger = get_logger("regime_validator")


class MarketRegimeValidator:
    """
    Validator for market regime data quality and integrity.

    Critical for institutional quant work - ensures regime classifications
    are consistent and reliable for regime-aware strategies.
    """

    REQUIRED_FIELDS = [
        "date",
        "vol_regime",
        "trend_regime",
        "liquidity_regime",
        "macro_regime",
        "event_time",
        "publication_time",
        "ingestion_time",
        "effective_time",
        "source",
        "version",
        "ingestion_job",
    ]

    VALID_VOL_REGIMES = ["LOW", "NORMAL", "HIGH", "EXTREME"]
    VALID_TREND_REGIMES = ["BULL", "BEAR", "SIDEWAYS", "TRANSITION"]
    VALID_LIQUIDITY_REGIMES = ["HIGH", "NORMAL", "LOW", "DRY"]
    VALID_MACRO_REGIMES = ["EXPANSION", "PEAK", "CONTRACTION", "TROUGH", "RECOVERY"]

    def __init__(self):
        """Initialize market regime validator."""
        self.logger = logger

    def validate_regime_dataframe(self, df: pd.DataFrame, dataset_name: str) -> dict[str, Any]:
        """
        Validate market regime DataFrame for quality and integrity.

        Args:
            df: DataFrame to validate
            dataset_name: Name of the dataset for logging

        Returns:
            Validation results dictionary
        """
        self.logger.info(f"Validating market regime data for dataset: {dataset_name}")

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

        if df.empty:
            return results

        # Validate timestamp ordering
        invalid_pub = df[df["publication_time"] < df["event_time"]]
        if not invalid_pub.empty:
            results["is_valid"] = False
            results["invalid_rows"] += len(invalid_pub)
            error_msg = (
                f"{len(invalid_pub)} rows have publication_time < event_time (lookahead bias)"
            )
            self.logger.error(error_msg)
            results["errors"].append(error_msg)

        invalid_eff = df[df["effective_time"] < df["publication_time"]]
        if not invalid_eff.empty:
            results["is_valid"] = False
            results["invalid_rows"] += len(invalid_eff)
            error_msg = f"{len(invalid_eff)} rows have effective_time < publication_time"
            self.logger.error(error_msg)
            results["errors"].append(error_msg)

        # Validate regime values
        if "vol_regime" in df.columns:
            invalid_vol = df[~df["vol_regime"].isin(self.VALID_VOL_REGIMES)]
            if not invalid_vol.empty:
                results["is_valid"] = False
                results["invalid_rows"] += len(invalid_vol)
                error_msg = f"{len(invalid_vol)} rows have invalid vol_regime values"
                self.logger.error(error_msg)
                results["errors"].append(error_msg)

        if "trend_regime" in df.columns:
            invalid_trend = df[~df["trend_regime"].isin(self.VALID_TREND_REGIMES)]
            if not invalid_trend.empty:
                results["is_valid"] = False
                results["invalid_rows"] += len(invalid_trend)
                error_msg = f"{len(invalid_trend)} rows have invalid trend_regime values"
                self.logger.error(error_msg)
                results["errors"].append(error_msg)

        if "liquidity_regime" in df.columns:
            invalid_liq = df[~df["liquidity_regime"].isin(self.VALID_LIQUIDITY_REGIMES)]
            if not invalid_liq.empty:
                results["is_valid"] = False
                results["invalid_rows"] += len(invalid_liq)
                error_msg = f"{len(invalid_liq)} rows have invalid liquidity_regime values"
                self.logger.error(error_msg)
                results["errors"].append(error_msg)

        if "macro_regime" in df.columns:
            invalid_macro = df[~df["macro_regime"].isin(self.VALID_MACRO_REGIMES)]
            if not invalid_macro.empty:
                results["is_valid"] = False
                results["invalid_rows"] += len(invalid_macro)
                error_msg = f"{len(invalid_macro)} rows have invalid macro_regime values"
                self.logger.error(error_msg)
                results["errors"].append(error_msg)

        # Validate scores are non-negative
        for score_col in ["volatility_score", "trend_score", "liquidity_score", "macro_score"]:
            if score_col in df.columns:
                negative_scores = df[df[score_col] < 0]
                if not negative_scores.empty:
                    results["is_valid"] = False
                    results["invalid_rows"] += len(negative_scores)
                    error_msg = f"{len(negative_scores)} rows have negative {score_col}"
                    self.logger.error(error_msg)
                    results["errors"].append(error_msg)

        # Validate confidence is between 0 and 1
        if "confidence" in df.columns:
            invalid_conf = df[(df["confidence"] < 0) | (df["confidence"] > 1)]
            if not invalid_conf.empty:
                results["is_valid"] = False
                results["invalid_rows"] += len(invalid_conf)
                error_msg = f"{len(invalid_conf)} rows have invalid confidence values"
                self.logger.error(error_msg)
                results["errors"].append(error_msg)

        if results["is_valid"]:
            self.logger.info(f"Market regime validation passed for {dataset_name}")
        else:
            self.logger.error(f"Market regime validation failed for {dataset_name}")

        return results

    def validate_regime_consistency(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Validate regime consistency over time.

        Checks for:
        - Frequent regime transitions (instability)
        - Impossible regime combinations
        - Missing dates in time series

        Args:
            df: DataFrame with regime data

        Returns:
            Validation results
        """
        self.logger.info("Validating regime consistency")

        results = {
            "is_valid": True,
            "transition_count": 0,
            "impossible_combinations": 0,
            "missing_dates": 0,
            "errors": [],
            "warnings": [],
        }

        if df.empty:
            return results

        # Sort by date
        df_sorted = df.sort_values("date")

        # Check for frequent transitions (more than 3 transitions in 5 days)
        if len(df_sorted) > 1:
            transitions = 0
            for i in range(1, len(df_sorted)):
                # Count regime changes
                if (
                    df_sorted.iloc[i]["vol_regime"] != df_sorted.iloc[i - 1]["vol_regime"]
                    or df_sorted.iloc[i]["trend_regime"] != df_sorted.iloc[i - 1]["trend_regime"]
                    or df_sorted.iloc[i]["liquidity_regime"]
                    != df_sorted.iloc[i - 1]["liquidity_regime"]
                    or df_sorted.iloc[i]["macro_regime"] != df_sorted.iloc[i - 1]["macro_regime"]
                ):
                    transitions += 1

            results["transition_count"] = transitions

            if transitions > len(df_sorted) * 0.5:  # More than 50% transitions
                warning_msg = f"High regime transition rate: {transitions} transitions in {len(df_sorted)} days"
                self.logger.warning(warning_msg)
                results["warnings"].append(warning_msg)

        # Check for impossible regime combinations
        # Example: EXTREME volatility with HIGH liquidity is unlikely
        impossible_combinations = 0
        for _, row in df_sorted.iterrows():
            if row["vol_regime"] == "EXTREME" and row["liquidity_regime"] == "HIGH":
                impossible_combinations += 1
                warning_msg = f"Unlikely regime combination: EXTREME volatility with HIGH liquidity on {row['date']}"
                self.logger.warning(warning_msg)
                results["warnings"].append(warning_msg)

        results["impossible_combinations"] = impossible_combinations

        # Check for missing dates
        if len(df_sorted) > 1:
            date_range = (df_sorted["date"].max() - df_sorted["date"].min()).days
            expected_dates = date_range + 1
            actual_dates = len(df_sorted)
            missing_dates = expected_dates - actual_dates

            if missing_dates > 0:
                warning_msg = f"Missing {missing_dates} dates in regime time series"
                self.logger.warning(warning_msg)
                results["warnings"].append(warning_msg)

            results["missing_dates"] = missing_dates

        return results

    def validate_regime_distribution(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Analyze regime distribution for quality assessment.

        Args:
            df: DataFrame with regime data

        Returns:
            Distribution statistics
        """
        self.logger.info("Analyzing regime distribution")

        stats = {
            "total_regimes": len(df),
            "vol_regime_distribution": {},
            "trend_regime_distribution": {},
            "liquidity_regime_distribution": {},
            "macro_regime_distribution": {},
        }

        if df.empty:
            return stats

        # Calculate distributions
        if "vol_regime" in df.columns:
            stats["vol_regime_distribution"] = df["vol_regime"].value_counts().to_dict()

        if "trend_regime" in df.columns:
            stats["trend_regime_distribution"] = df["trend_regime"].value_counts().to_dict()

        if "liquidity_regime" in df.columns:
            stats["liquidity_regime_distribution"] = df["liquidity_regime"].value_counts().to_dict()

        if "macro_regime" in df.columns:
            stats["macro_regime_distribution"] = df["macro_regime"].value_counts().to_dict()

        return stats


def validate_market_regime_data(df: pd.DataFrame, dataset_name: str) -> bool:
    """
    Convenience function to validate market regime data.

    Args:
        df: DataFrame to validate
        dataset_name: Name of the dataset

    Returns:
        True if validation passes, raises exception otherwise
    """
    validator = MarketRegimeValidator()
    results = validator.validate_regime_dataframe(df, dataset_name)

    if not results["is_valid"]:
        error_msg = f"Market regime validation failed for {dataset_name}: {results['errors']}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    return True
