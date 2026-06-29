"""
Ingestion Engine

Production-grade ingestion engine with automatic fallback logic.
Implements: Try nselib → fallback scraper → fallback cache → fail
"""

import json
from pathlib import Path

from data_platform.sources.ingestion.interface import IngestionResult
from data_platform.sources.ingestion.nselib_source import NSELibSource
from data_platform.sources.ingestion.scraper_source import ScraperSource
from utils.logger import get_logger
from utils.time_utils import now_ist

logger = get_logger("ingestion_engine", pipeline_id="ingestion_engine")


class IngestionEngine:
    """
    Production-grade ingestion engine with automatic fallback.

    Implements resilient ingestion chain:
    1. Try nselib (primary)
    2. Fallback to scraper (secondary)
    3. Fallback to cache (tertiary)
    4. Fail with error
    """

    def __init__(self, cache_enabled: bool = True):
        """
        Initialize ingestion engine.

        Args:
            cache_enabled: Enable cache fallback
        """
        self.primary_source = NSELibSource()
        self.secondary_source = ScraperSource()
        self.cache_enabled = cache_enabled
        self.cache_dir = Path("data/bronze/cache")
        self.logger = logger

        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_equity_history(
        self, symbol: str, from_date: str, to_date: str, use_fallback: bool = True
    ) -> IngestionResult:
        """
        Fetch equity history with automatic fallback.

        Args:
            symbol: Stock symbol
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            use_fallback: Enable fallback to secondary source

        Returns:
            IngestionResult with equity history data
        """
        self.logger.info(f"Fetching equity history for {symbol} with fallback enabled")

        # Try primary source (nselib)
        result = self.primary_source.fetch_equity_history(symbol, from_date, to_date)

        if result.success:
            self.logger.info("Successfully fetched from primary source (nselib)")
            return result

        self.logger.warning(f"Primary source failed: {result.error}")

        if not use_fallback:
            return result

        # Fallback to secondary source (scraper)
        self.logger.info("Falling back to secondary source (scraper)")
        result = self.secondary_source.fetch_equity_history(symbol, from_date, to_date)

        if result.success:
            self.logger.info("Successfully fetched from secondary source (scraper)")
            return result

        self.logger.warning(f"Secondary source failed: {result.error}")

        # Fallback to cache
        if self.cache_enabled:
            self.logger.info("Falling back to cache")
            result = self._fetch_from_cache(f"equity_history_{symbol}_{from_date}_{to_date}")

            if result.success:
                self.logger.info("Successfully fetched from cache")
                return result

        # All sources failed
        self.logger.error("All ingestion sources failed for equity history")
        return result

    def fetch_options_chain(
        self, symbol: str, expiry_date: str | None = None, use_fallback: bool = True
    ) -> IngestionResult:
        """
        Fetch options chain with automatic fallback.

        Args:
            symbol: Stock symbol
            expiry_date: Expiry date (YYYY-MM-DD), None for current
            use_fallback: Enable fallback to secondary source

        Returns:
            IngestionResult with options chain data
        """
        self.logger.info(f"Fetching options chain for {symbol} with fallback enabled")

        # Try primary source
        result = self.primary_source.fetch_options_chain(symbol, expiry_date)

        if result.success:
            self.logger.info("Successfully fetched from primary source (nselib)")
            return result

        self.logger.warning(f"Primary source failed: {result.error}")

        if not use_fallback:
            return result

        # Fallback to secondary source
        self.logger.info("Falling back to secondary source (scraper)")
        result = self.secondary_source.fetch_options_chain(symbol, expiry_date)

        if result.success:
            self.logger.info("Successfully fetched from secondary source (scraper)")
            return result

        self.logger.warning(f"Secondary source failed: {result.error}")

        # Fallback to cache
        if self.cache_enabled:
            cache_key = (
                f"options_chain_{symbol}_{expiry_date}"
                if expiry_date
                else f"options_chain_{symbol}"
            )
            result = self._fetch_from_cache(cache_key)

            if result.success:
                self.logger.info("Successfully fetched from cache")
                return result

        # All sources failed
        self.logger.error("All ingestion sources failed for options chain")
        return result

    def fetch_fii_dii(self, use_fallback: bool = True) -> IngestionResult:
        """
        Fetch FII/DII data with automatic fallback.

        Args:
            use_fallback: Enable fallback to secondary source

        Returns:
            IngestionResult with FII/DII data
        """
        self.logger.info("Fetching FII/DII data with fallback enabled")

        # Try primary source
        result = self.primary_source.fetch_fii_dii()

        if result.success:
            self.logger.info("Successfully fetched from primary source (nselib)")
            return result

        self.logger.warning(f"Primary source failed: {result.error}")

        if not use_fallback:
            return result

        # Fallback to secondary source
        self.logger.info("Falling back to secondary source (scraper)")
        result = self.secondary_source.fetch_fii_dii()

        if result.success:
            self.logger.info("Successfully fetched from secondary source (scraper)")
            return result

        self.logger.warning(f"Secondary source failed: {result.error}")

        # Fallback to cache
        if self.cache_enabled:
            result = self._fetch_from_cache("fii_dii")

            if result.success:
                self.logger.info("Successfully fetched from cache")
                return result

        # All sources failed
        self.logger.error("All ingestion sources failed for FII/DII")
        return result

    def fetch_corporate_actions(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
        use_fallback: bool = True,
    ) -> IngestionResult:
        """
        Fetch corporate actions with automatic fallback.

        Args:
            from_date: Start date (YYYY-MM-DD), None for all
            to_date: End date (YYYY-MM-DD), None for all
            use_fallback: Enable fallback to secondary source

        Returns:
            IngestionResult with corporate actions data
        """
        self.logger.info("Fetching corporate actions with fallback enabled")

        # Try primary source
        result = self.primary_source.fetch_corporate_actions(from_date, to_date)

        if result.success:
            self.logger.info("Successfully fetched from primary source (nselib)")
            return result

        self.logger.warning(f"Primary source failed: {result.error}")

        if not use_fallback:
            return result

        # Fallback to secondary source
        self.logger.info("Falling back to secondary source (scraper)")
        result = self.secondary_source.fetch_corporate_actions(from_date, to_date)

        if result.success:
            self.logger.info("Successfully fetched from secondary source (scraper)")
            return result

        self.logger.warning(f"Secondary source failed: {result.error}")

        # Fallback to cache
        if self.cache_enabled:
            cache_key = (
                f"corporate_actions_{from_date}_{to_date}"
                if from_date and to_date
                else "corporate_actions"
            )
            result = self._fetch_from_cache(cache_key)

            if result.success:
                self.logger.info("Successfully fetched from cache")
                return result

        # All sources failed
        self.logger.error("All ingestion sources failed for corporate actions")
        return result

    def fetch_trading_calendar(self, use_fallback: bool = True) -> IngestionResult:
        """
        Fetch trading calendar with automatic fallback.

        Args:
            use_fallback: Enable fallback to secondary source

        Returns:
            IngestionResult with trading calendar data
        """
        self.logger.info("Fetching trading calendar with fallback enabled")

        # Try primary source
        result = self.primary_source.fetch_trading_calendar()

        if result.success:
            self.logger.info("Successfully fetched from primary source (nselib)")
            return result

        self.logger.warning(f"Primary source failed: {result.error}")

        if not use_fallback:
            return result

        # Fallback to secondary source
        self.logger.info("Falling back to secondary source (scraper)")
        result = self.secondary_source.fetch_trading_calendar()

        if result.success:
            self.logger.info("Successfully fetched from secondary source (scraper)")
            return result

        self.logger.warning(f"Secondary source failed: {result.error}")

        # Fallback to cache
        if self.cache_enabled:
            result = self._fetch_from_cache("trading_calendar")

            if result.success:
                self.logger.info("Successfully fetched from cache")
                return result

        # All sources failed
        self.logger.error("All ingestion sources failed for trading calendar")
        return result

    def fetch_security_master(self, use_fallback: bool = True) -> IngestionResult:
        """
        Fetch security master with automatic fallback.

        Args:
            use_fallback: Enable fallback to secondary source

        Returns:
            IngestionResult with security master data
        """
        self.logger.info("Fetching security master with fallback enabled")

        # Try primary source
        result = self.primary_source.fetch_security_master()

        if result.success:
            self.logger.info("Successfully fetched from primary source (nselib)")
            return result

        self.logger.warning(f"Primary source failed: {result.error}")

        if not use_fallback:
            return result

        # Fallback to secondary source
        self.logger.info("Falling back to secondary source (scraper)")
        result = self.secondary_source.fetch_security_master()

        if result.success:
            self.logger.info("Successfully fetched from secondary source (scraper)")
            return result

        self.logger.warning(f"Secondary source failed: {result.error}")

        # Fallback to cache
        if self.cache_enabled:
            result = self._fetch_from_cache("security_master")

            if result.success:
                self.logger.info("Successfully fetched from cache")
                return result

        # All sources failed
        self.logger.error("All ingestion sources failed for security master")
        return result

    def _fetch_from_cache(self, cache_key: str) -> IngestionResult:
        """
        Fetch data from cache.

        Args:
            cache_key: Cache key

        Returns:
            IngestionResult with cached data
        """
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"

            if not cache_file.exists():
                return IngestionResult(
                    success=False,
                    data=None,
                    source="cache",
                    timestamp=now_ist(),
                    latency_ms=0,
                    error="Cache miss",
                )

            with open(cache_file) as f:
                data = json.load(f)

            return IngestionResult(
                success=True,
                data=data,
                source="cache",
                timestamp=now_ist(),
                latency_ms=0,
                metadata={"cached": True},
            )

        except Exception as e:
            return IngestionResult(
                success=False,
                data=None,
                source="cache",
                timestamp=now_ist(),
                latency_ms=0,
                error=f"Cache read failed: {str(e)}",
            )

    def save_to_cache(self, cache_key: str, data: dict) -> None:
        """
        Save data to cache.

        Args:
            cache_key: Cache key
            data: Data to cache
        """
        if not self.cache_enabled:
            return

        try:
            cache_file = self.cache_dir / f"{cache_key}.json"

            with open(cache_file, "w") as f:
                json.dump(data, f)

            self.logger.info(f"Saved to cache: {cache_key}")

        except Exception as e:
            self.logger.error(f"Failed to save to cache: {str(e)}")

    def get_source_health(self) -> dict:
        """
        Get health metrics for all sources.

        Returns:
            Dictionary with source health metrics
        """
        return {
            "primary": self.primary_source.get_health(),
            "secondary": self.secondary_source.get_health(),
        }
