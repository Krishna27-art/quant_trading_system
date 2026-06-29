"""
Scraper Source Implementation

Fallback data source using yfinance.
Secondary source when nselib fails or for missing endpoints.
"""

import time

import pandas as pd
import yfinance as yf

from data_platform.sources.ingestion.interface import IngestionResult, NSEDataSource
from utils.logger import get_logger
from utils.time_utils import now_ist

logger = get_logger("scraper_source", pipeline_id="ingestion_scraper")


class ScraperSource(NSEDataSource):
    """
    NSE data source using yfinance (live).
    """

    def __init__(self):
        """Initialize Scraper source."""
        super().__init__("scraper")
        self.logger = logger

    def fetch_equity_history(self, symbol: str, from_date: str, to_date: str) -> IngestionResult:
        """
        Fetch equity history data via yfinance.
        """
        start_time = time.time()

        try:
            self.logger.info(f"Fetching equity history for {symbol} via yfinance")
            yf_symbol = f"{symbol}.NS"
            # yfinance expects date in YYYY-MM-DD format
            df = yf.download(yf_symbol, start=from_date, end=to_date, progress=False)

            if df.empty:
                error_msg = f"No data returned from yfinance for {symbol}"
                self.logger.warning(error_msg)
                self._record_failure(error_msg)
                return IngestionResult(
                    success=False,
                    data=None,
                    source=self.name,
                    timestamp=now_ist(),
                    latency_ms=round((time.time() - start_time) * 1000, 2),
                    error=error_msg,
                )

            # Format dataframe correctly
            df = df.reset_index()
            # If yf returns MultiIndex columns (like in recent yf versions), flatten it
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]

            df.columns = [c.lower() for c in df.columns]
            df["symbol"] = symbol

            # Ensure required columns
            for col in ["open", "high", "low", "close", "volume"]:
                if col not in df.columns:
                    df[col] = 0.0

            return IngestionResult(
                success=True,
                data=df,
                source=self.name,
                timestamp=now_ist(),
                latency_ms=round((time.time() - start_time) * 1000, 2),
            )

        except Exception as e:
            error_msg = f"Failed to fetch equity history via scraper: {str(e)}"
            self.logger.error(error_msg)
            self._record_failure(error_msg)
            return IngestionResult(
                success=False,
                data=None,
                source=self.name,
                timestamp=now_ist(),
                latency_ms=round((time.time() - start_time) * 1000, 2),
                error=error_msg,
            )

    def fetch_options_chain(self, symbol: str, expiry_date: str | None = None) -> IngestionResult:
        # Not implemented for yfinance easily
        start_time = time.time()
        return IngestionResult(
            success=False,
            data=None,
            source=self.name,
            timestamp=now_ist(),
            latency_ms=round((time.time() - start_time) * 1000, 2),
            error="Not implemented in yfinance scraper",
        )

    def fetch_fii_dii(self) -> IngestionResult:
        start_time = time.time()
        return IngestionResult(
            success=False,
            data=None,
            source=self.name,
            timestamp=now_ist(),
            latency_ms=round((time.time() - start_time) * 1000, 2),
            error="Not implemented in yfinance scraper",
        )

    def fetch_corporate_actions(
        self, from_date: str | None = None, to_date: str | None = None
    ) -> IngestionResult:
        start_time = time.time()
        return IngestionResult(
            success=False,
            data=None,
            source=self.name,
            timestamp=now_ist(),
            latency_ms=round((time.time() - start_time) * 1000, 2),
            error="Not implemented in yfinance scraper",
        )

    def fetch_trading_calendar(self) -> IngestionResult:
        start_time = time.time()
        return IngestionResult(
            success=False,
            data=None,
            source=self.name,
            timestamp=now_ist(),
            latency_ms=round((time.time() - start_time) * 1000, 2),
            error="Not implemented in yfinance scraper",
        )

    def fetch_security_master(self) -> IngestionResult:
        start_time = time.time()
        return IngestionResult(
            success=False,
            data=None,
            source=self.name,
            timestamp=now_ist(),
            latency_ms=round((time.time() - start_time) * 1000, 2),
            error="Not implemented in yfinance scraper",
        )
