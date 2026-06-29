"""
Enhanced Data Quality Monitoring and Alerting

Institutional-grade monitoring for:
- Data gaps detection
- Ingestion delays
- Schema violations
- Quality score degradation
- Real-time alerting
"""

from pathlib import Path
from typing import Any

import pandas as pd

from data_platform.validation.base_validator import ValidationSeverity
from data_platform.validation.ingestion_validator import IngestionValidator
from utils.data_freshness_monitor import DataFreshnessMonitor
from utils.logger import get_logger
from utils.time_utils import now_ist

logger = get_logger("enhanced_monitoring", pipeline_id="data_monitoring")


class DataGapDetector:
    """
    Detects data gaps in time series data.

    Institutional systems must detect:
    - Missing trading days
    - Missing timestamps in tick data
    - Gaps in expected data ranges
    """

    def __init__(self):
        """Initialize data gap detector."""
        self.logger = logger

    def detect_trading_day_gaps(
        self, df: pd.DataFrame, date_column: str, expected_dates: list[str]
    ) -> dict[str, Any]:
        """
        Detect gaps in trading day coverage.

        Args:
            df: DataFrame with date column
            date_column: Name of date column
            expected_dates: List of expected dates (YYYY-MM-DD)

        Returns:
            Gap detection results
        """
        self.logger.info("Detecting trading day gaps")

        if date_column not in df.columns:
            return {
                "valid": False,
                "error": f"Date column {date_column} not found",
                "missing_dates": expected_dates,
            }

        actual_dates = set(pd.to_datetime(df[date_column]).dt.strftime("%Y-%m-%d"))
        expected_dates_set = set(expected_dates)

        missing_dates = expected_dates_set - actual_dates

        results = {
            "valid": len(missing_dates) == 0,
            "expected_count": len(expected_dates),
            "actual_count": len(actual_dates),
            "missing_count": len(missing_dates),
            "missing_dates": sorted(missing_dates),
            "gap_percentage": (
                (len(missing_dates) / len(expected_dates)) * 100 if expected_dates else 0
            ),
        }

        if missing_dates:
            self.logger.warning(f"Found {len(missing_dates)} missing trading days")

        return results

    def detect_timestamp_gaps(
        self,
        df: pd.DataFrame,
        timestamp_column: str,
        expected_frequency: str = "1min",
        max_gap_seconds: int = 60,
    ) -> dict[str, Any]:
        """
        Detect gaps in timestamp continuity for tick/intraday data.

        Args:
            df: DataFrame with timestamp column
            timestamp_column: Name of timestamp column
            expected_frequency: Expected frequency (e.g., '1min', '5min', '1s')
            max_gap_seconds: Maximum allowed gap in seconds

        Returns:
            Gap detection results
        """
        self.logger.info(f"Detecting timestamp gaps (expected: {expected_frequency})")

        if timestamp_column not in df.columns:
            return {"valid": False, "error": f"Timestamp column {timestamp_column} not found"}

        # Preserve arrival order: out-of-order ticks are themselves a gap signal.
        timestamps = pd.to_datetime(df[timestamp_column])

        if len(timestamps) < 2:
            return {"valid": True, "message": "Insufficient data for gap detection"}

        # Calculate time differences
        time_diffs = timestamps.diff().dt.total_seconds()

        # Detect gaps exceeding threshold
        gaps = time_diffs[time_diffs > max_gap_seconds]

        results = {
            "valid": len(gaps) == 0,
            "total_records": len(df),
            "gap_count": len(gaps),
            "max_gap_seconds": gaps.max() if not gaps.empty else 0,
            "avg_gap_seconds": gaps.mean() if not gaps.empty else 0,
            "gap_indices": gaps.index.tolist()[:10],  # First 10 gaps
            "expected_frequency": expected_frequency,
        }

        if not gaps.empty:
            self.logger.warning(f"Found {len(gaps)} timestamp gaps > {max_gap_seconds}s")

        return results

    def detect_symbol_coverage_gaps(
        self, df: pd.DataFrame, expected_symbols: list[str], symbol_column: str = "symbol"
    ) -> dict[str, Any]:
        """
        Detect gaps in symbol coverage.

        Args:
            df: DataFrame with symbol column
            expected_symbols: List of expected symbols
            symbol_column: Name of symbol column

        Returns:
            Gap detection results
        """
        self.logger.info("Detecting symbol coverage gaps")

        if symbol_column not in df.columns:
            return {
                "valid": False,
                "error": f"Symbol column {symbol_column} not found",
                "missing_symbols": expected_symbols,
            }

        actual_symbols = set(df[symbol_column].unique())
        expected_symbols_set = set(expected_symbols)

        missing_symbols = expected_symbols_set - actual_symbols

        results = {
            "valid": len(missing_symbols) == 0,
            "expected_count": len(expected_symbols),
            "actual_count": len(actual_symbols),
            "missing_count": len(missing_symbols),
            "missing_symbols": sorted(missing_symbols),
            "coverage_percentage": (
                (len(actual_symbols) / len(expected_symbols)) * 100 if expected_symbols else 0
            ),
        }

        if missing_symbols:
            self.logger.warning(f"Found {len(missing_symbols)} missing symbols")

        return results


class EnhancedDataQualityMonitor:
    """
    Enhanced monitoring system for data quality.

    Combines:
    - Data freshness monitoring
    - Gap detection
    - Schema validation
    - Quality score tracking
    - Alerting
    """

    def __init__(
        self, schema_registry_path: Path, quarantine_path: Path, alert_threshold_hours: int = 24
    ):
        """
        Initialize enhanced monitor.

        Args:
            schema_registry_path: Path to schema registry
            quarantine_path: Path to quarantine directory
            alert_threshold_hours: Hours after which to alert
        """
        self.freshness_monitor = DataFreshnessMonitor(alert_threshold_hours)
        self.gap_detector = DataGapDetector()
        self.ingestion_validator = IngestionValidator(
            schema_registry_path=schema_registry_path, quarantine_path=quarantine_path
        )
        self.logger = logger

        # Quality score history
        self.quality_history: dict[str, list[dict[str, Any]]] = {}

    def monitor_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        source: str,
        expected_dates: list[str] | None = None,
        expected_symbols: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Comprehensive monitoring of a dataset.

        Args:
            df: DataFrame to monitor
            dataset_name: Name of dataset
            source: Data source
            expected_dates: Expected dates (for gap detection)
            expected_symbols: Expected symbols (for coverage detection)

        Returns:
            Comprehensive monitoring results
        """
        self.logger.info(f"Running comprehensive monitoring for {dataset_name}")

        results = {
            "dataset_name": dataset_name,
            "source": source,
            "timestamp": now_ist().isoformat(),
            "overall_status": "unknown",
            "components": {},
        }

        # 1. Schema validation
        validated_df, validation_report, validation_metadata = (
            self.ingestion_validator.validate_at_ingestion(
                df=df, dataset_name=dataset_name, source=source
            )
        )
        results["components"]["validation"] = {
            "passed": validation_report.is_acceptable(),
            "score": validation_report.calculate_score(),
            "critical_failures": len(validation_report.critical_failures),
            "high_failures": len(
                [
                    r
                    for r in validation_report.results
                    if not r.passed and r.severity == ValidationSeverity.WARNING
                ]
            ),
            "metadata": validation_metadata,
        }

        # 2. Gap detection
        gap_results = {}

        if expected_dates and "date" in df.columns:
            gap_results["trading_days"] = self.gap_detector.detect_trading_day_gaps(
                df, "date", expected_dates
            )

        if "timestamp" in df.columns:
            gap_results["timestamps"] = self.gap_detector.detect_timestamp_gaps(df, "timestamp")

        if expected_symbols and "symbol" in df.columns:
            gap_results["symbols"] = self.gap_detector.detect_symbol_coverage_gaps(
                df, expected_symbols
            )

        results["components"]["gaps"] = gap_results

        # 3. Freshness check
        if "date" in df.columns:
            latest_date = pd.to_datetime(df["date"]).max()
            freshness_result = self.freshness_monitor.check_freshness(
                data_source=dataset_name, last_update_time=latest_date.isoformat()
            )
            results["components"]["freshness"] = freshness_result

        # 4. Calculate overall status
        validation_passed = results["components"]["validation"]["passed"]
        gaps_detected = any(
            not gap_result.get("valid", True) for gap_result in gap_results.values()
        )
        is_stale = results["components"].get("freshness", {}).get("stale", False)

        if validation_passed and not gaps_detected and not is_stale:
            results["overall_status"] = "healthy"
        elif not validation_passed:
            results["overall_status"] = "validation_failed"
        elif gaps_detected:
            results["overall_status"] = "gaps_detected"
        elif is_stale:
            results["overall_status"] = "stale"

        # 5. Track quality score history
        self._track_quality_score(dataset_name, results["components"]["validation"]["score"])

        # 6. Generate alerts if needed
        self._generate_alerts(results)

        return results

    def _track_quality_score(self, dataset_name: str, score: float):
        """Track quality score over time."""
        if dataset_name not in self.quality_history:
            self.quality_history[dataset_name] = []

        self.quality_history[dataset_name].append(
            {"timestamp": now_ist().isoformat(), "score": score}
        )

        # Keep only last 100 records
        if len(self.quality_history[dataset_name]) > 100:
            self.quality_history[dataset_name] = self.quality_history[dataset_name][-100:]

    def _generate_alerts(self, monitoring_results: dict[str, Any]):
        """Generate alerts based on monitoring results."""
        status = monitoring_results["overall_status"]
        dataset_name = monitoring_results["dataset_name"]

        if status == "validation_failed":
            self.freshness_monitor.send_alert(
                f"CRITICAL: Validation failed for {dataset_name}", severity="critical"
            )
        elif status == "gaps_detected":
            self.freshness_monitor.send_alert(
                f"WARNING: Data gaps detected for {dataset_name}", severity="warning"
            )
        elif status == "stale":
            self.freshness_monitor.send_alert(
                f"WARNING: Data is stale for {dataset_name}", severity="warning"
            )

    def get_quality_history(self, dataset_name: str) -> list[dict[str, Any]]:
        """Get quality score history for a dataset."""
        return self.quality_history.get(dataset_name, [])

    def detect_quality_degradation(self, dataset_name: str, threshold: float = 10.0) -> bool:
        """
        Detect if quality score has degraded significantly.

        Args:
            dataset_name: Name of dataset
            threshold: Threshold for degradation (percentage points)

        Returns:
            True if degradation detected
        """
        history = self.get_quality_history(dataset_name)

        if len(history) < 2:
            return False

        recent_score = history[-1]["score"]
        previous_score = history[-2]["score"]

        degradation = previous_score - recent_score

        if degradation > threshold:
            self.logger.warning(
                f"Quality degradation detected for {dataset_name}: "
                f"{previous_score:.2f} -> {recent_score:.2f} ({degradation:.2f} points)"
            )
            return True

        return False


def create_enhanced_monitor(
    schema_registry_path: Path | None = None,
    quarantine_path: Path | None = None,
    alert_threshold_hours: int = 24,
) -> EnhancedDataQualityMonitor:
    """
    Factory function to create an enhanced monitor.

    Args:
        schema_registry_path: Path to schema registry
        quarantine_path: Path to quarantine directory
        alert_threshold_hours: Hours after which to alert

    Returns:
        EnhancedDataQualityMonitor instance
    """
    from config.settings import BRONZE_DIR

    if schema_registry_path is None:
        schema_registry_path = BRONZE_DIR / "schema_registry"

    if quarantine_path is None:
        quarantine_path = BRONZE_DIR / "quarantine"

    return EnhancedDataQualityMonitor(
        schema_registry_path=schema_registry_path,
        quarantine_path=quarantine_path,
        alert_threshold_hours=alert_threshold_hours,
    )
