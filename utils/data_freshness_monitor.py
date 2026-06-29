"""
Data Freshness Monitor

Institutional-grade data freshness monitoring for data layer.
Part of the institutional data layer.
"""

from datetime import datetime
from typing import Any

from utils.logger import get_logger
from utils.time_utils import now_ist

logger = get_logger("data_freshness_monitor", pipeline_id="data_monitoring")


class DataFreshnessMonitor:
    """
    Monitors data freshness and sends alerts.

    Institutional firms monitor data 24/7:
    - NSE feed missing
    - PCR missing
    - OI missing
    - Any data delays
    """

    def __init__(self, alert_threshold_hours: int = 24):
        """
        Initialize data freshness monitor.

        Args:
            alert_threshold_hours: Hours after which to alert on stale data
        """
        self.alert_threshold_hours = alert_threshold_hours
        self.logger = logger

        # Data sources to monitor
        self.data_sources = {
            "nse_equity": {"expected_update": "15:30", "max_delay_hours": 2},
            "nse_options": {"expected_update": "15:30", "max_delay_hours": 2},
            "nse_bhavcopy": {"expected_update": "18:00", "max_delay_hours": 4},
            "fii_dii": {"expected_update": "18:00", "max_delay_hours": 24},
            "gift_nifty": {"expected_update": "09:00", "max_delay_hours": 1},
        }

    def check_freshness(self, data_source: str, last_update_time: str) -> dict[str, Any]:
        """
        Check freshness of a data source.

        Args:
            data_source: Name of data source
            last_update_time: Last update timestamp (ISO format)

        Returns:
            Freshness check results
        """
        self.logger.info(f"Checking freshness for {data_source}")

        if data_source not in self.data_sources:
            self.logger.warning(f"Unknown data source: {data_source}")
            return {"valid": False, "error": f"Unknown data source: {data_source}"}

        last_update = datetime.fromisoformat(last_update_time)
        now = now_ist()
        age_hours = (now - last_update).total_seconds() / 3600

        config = self.data_sources[data_source]
        max_delay = config["max_delay_hours"]

        results = {
            "data_source": data_source,
            "last_update": last_update_time,
            "current_time": now.isoformat(),
            "age_hours": age_hours,
            "max_delay_hours": max_delay,
            "stale": age_hours > max_delay,
            "alert": age_hours > self.alert_threshold_hours,
        }

        if results["stale"]:
            self.logger.warning(f"{data_source} is stale: {age_hours:.1f} hours old")

        if results["alert"]:
            self.logger.error(
                f"ALERT: {data_source} is critically stale: {age_hours:.1f} hours old"
            )

        return results

    def check_all_freshness(self, last_updates: dict[str, str]) -> dict[str, Any]:
        """
        Check freshness for all data sources.

        Args:
            last_updates: Dictionary mapping data source names to last update times

        Returns:
            Freshness check results for all sources
        """
        self.logger.info("Checking freshness for all data sources")

        results = {}

        for source, last_update in last_updates.items():
            results[source] = self.check_freshness(source, last_update)

        # Summary
        stale_count = sum(1 for r in results.values() if r["stale"])
        alert_count = sum(1 for r in results.values() if r["alert"])

        summary = {
            "total_sources": len(results),
            "stale_count": stale_count,
            "alert_count": alert_count,
            "sources": results,
        }

        self.logger.info(f"Freshness check: {stale_count} stale, {alert_count} alerts")

        return summary

    def check_missing_data(
        self, expected_data: dict[str, list[str]], actual_data: dict[str, list[str]]
    ) -> dict[str, Any]:
        """
        Check for missing data.

        Args:
            expected_data: Dictionary mapping source names to expected dates/symbols
            actual_data: Dictionary mapping source names to actual dates/symbols

        Returns:
            Missing data check results
        """
        self.logger.info("Checking for missing data")

        results = {}

        for source, expected in expected_data.items():
            actual = actual_data.get(source, [])
            missing = set(expected) - set(actual)

            results[source] = {
                "expected_count": len(expected),
                "actual_count": len(actual),
                "missing_count": len(missing),
                "missing_items": sorted(missing),
            }

            if missing:
                self.logger.warning(f"{source}: {len(missing)} items missing")

        return results

    def send_alert(self, message: str, severity: str = "warning") -> None:
        """
        Send an alert.

        Args:
            message: Alert message
            severity: Alert severity (info, warning, error, critical)
        """
        if severity == "critical":
            self.logger.error(f"CRITICAL ALERT: {message}")
        elif severity == "error":
            self.logger.error(f"ERROR ALERT: {message}")
        elif severity == "warning":
            self.logger.warning(f"WARNING ALERT: {message}")
        else:
            self.logger.info(f"INFO ALERT: {message}")

    def check_and_alert(
        self,
        last_updates: dict[str, str],
        expected_data: dict[str, list[str]] | None = None,
        actual_data: dict[str, list[str]] | None = None,
    ) -> None:
        """
        Check freshness and missing data, send alerts if needed.

        Args:
            last_updates: Dictionary mapping data source names to last update times
            expected_data: Dictionary mapping source names to expected dates/symbols
            actual_data: Dictionary mapping source names to actual dates/symbols
        """
        # Check freshness
        freshness_results = self.check_all_freshness(last_updates)

        # Send alerts for stale data
        for source, result in freshness_results["sources"].items():
            if result["alert"]:
                self.send_alert(
                    f"{source} is critically stale: {result['age_hours']:.1f} hours old",
                    severity="critical",
                )
            elif result["stale"]:
                self.send_alert(
                    f"{source} is stale: {result['age_hours']:.1f} hours old", severity="warning"
                )

        # Check missing data if provided
        if expected_data and actual_data:
            missing_results = self.check_missing_data(expected_data, actual_data)

            for source, result in missing_results.items():
                if result["missing_count"] > 0:
                    self.send_alert(
                        f"{source}: {result['missing_count']} items missing", severity="warning"
                    )
