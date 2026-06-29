"""
NSE Data Source Interface

Unified interface for all NSE data sources.
Provides abstraction for nselib, scraper, and future vendor feeds.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class IngestionResult:
    """
    Result of data ingestion operation.

    Contains the raw data, metadata, and lineage information.
    """

    success: bool
    data: Any | None
    source: str
    timestamp: datetime
    latency_ms: float
    error: str | None = None
    metadata: dict[str, Any] | None = None


class NSEDataSource(ABC):
    """
    Abstract base class for NSE data sources.

    All data sources (nselib, scraper, vendor) must implement this interface.
    """

    def __init__(self, name: str):
        """
        Initialize data source.

        Args:
            name: Source name (e.g., 'nselib', 'scraper', 'vendor')
        """
        self.name = name
        self._last_error: str | None = None
        self._success_count: int = 0
        self._failure_count: int = 0

    @abstractmethod
    def fetch_equity_history(self, symbol: str, from_date: str, to_date: str) -> IngestionResult:
        """
        Fetch equity history data.

        Args:
            symbol: Stock symbol
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)

        Returns:
            IngestionResult with equity history data
        """
        pass

    @abstractmethod
    def fetch_options_chain(self, symbol: str, expiry_date: str | None = None) -> IngestionResult:
        """
        Fetch options chain data.

        Args:
            symbol: Stock symbol
            expiry_date: Expiry date (YYYY-MM-DD), None for current

        Returns:
            IngestionResult with options chain data
        """
        pass

    @abstractmethod
    def fetch_fii_dii(self) -> IngestionResult:
        """
        Fetch FII/DII flow data.

        Returns:
            IngestionResult with FII/DII flow data
        """
        pass

    @abstractmethod
    def fetch_corporate_actions(
        self, from_date: str | None = None, to_date: str | None = None
    ) -> IngestionResult:
        """
        Fetch corporate actions data.

        Args:
            from_date: Start date (YYYY-MM-DD), None for all
            to_date: End date (YYYY-MM-DD), None for all

        Returns:
            IngestionResult with corporate actions data
        """
        pass

    @abstractmethod
    def fetch_trading_calendar(self) -> IngestionResult:
        """
        Fetch trading calendar data.

        Returns:
            IngestionResult with trading calendar data
        """
        pass

    @abstractmethod
    def fetch_security_master(self) -> IngestionResult:
        """
        Fetch security master data.

        Returns:
            IngestionResult with security master data
        """
        pass

    def get_health(self) -> dict[str, Any]:
        """
        Get source health metrics.

        Returns:
            Dictionary with health metrics
        """
        total = self._success_count + self._failure_count
        success_rate = self._success_count / total if total > 0 else 0.0

        return {
            "source": self.name,
            "success_count": self._success_count,
            "failure_count": self._failure_count,
            "success_rate": success_rate,
            "last_error": self._last_error,
        }

    def _record_success(self) -> None:
        """Record a successful operation."""
        self._success_count += 1
        self._last_error = None

    def _record_failure(self, error: str) -> None:
        """Record a failed operation."""
        self._failure_count += 1
        self._last_error = error
