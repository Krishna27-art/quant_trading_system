"""
NSELib Source Implementation

Primary data source using official NSE nselib library.
"""

import time

import pandas as pd
from nselib import capital_market

from data_platform.sources.ingestion.interface import IngestionResult, NSEDataSource
from utils.logger import get_logger
from utils.time_utils import now_ist

logger = get_logger("nselib_source", pipeline_id="ingestion_nselib")


class NSELibSource(NSEDataSource):
    """
    NSE data source using official nselib library.

    Primary source for NSE data with official API access.
    """

    def __init__(self):
        """Initialize NSELib source."""
        super().__init__("nselib")
        self.logger = logger

    def fetch_equity_history(self, symbol: str, from_date: str, to_date: str) -> IngestionResult:
        """
        Fetch equity history data from nselib.

        Args:
            symbol: Stock symbol
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)

        Returns:
            IngestionResult with equity history data
        """
        start_time = time.time()

        try:
            self.logger.info(f"Fetching equity history for {symbol} from nselib")

            df = capital_market.equity_history(symbol=symbol, from_date=from_date, to_date=to_date)

            if df is None or df.empty:
                self._record_failure("No data returned from nselib")
                return IngestionResult(
                    success=False,
                    data=None,
                    source=self.name,
                    timestamp=now_ist(),
                    latency_ms=round((time.time() - start_time) * 1000, 2),
                    error="No data returned from nselib",
                )

            self._record_success()

            return IngestionResult(
                success=True,
                data=df,
                source=self.name,
                timestamp=now_ist(),
                latency_ms=round((time.time() - start_time) * 1000, 2),
                metadata={
                    "symbol": symbol,
                    "from_date": from_date,
                    "to_date": to_date,
                    "row_count": len(df),
                },
            )

        except Exception as e:
            error_msg = f"Failed to fetch equity history: {str(e)}"
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
        """
        Fetch options chain data from nselib.

        Args:
            symbol: Stock symbol
            expiry_date: Expiry date (YYYY-MM-DD), None for current

        Returns:
            IngestionResult with options chain data
        """
        start_time = time.time()

        try:
            self.logger.info(f"Fetching options chain for {symbol} from nselib")

            if expiry_date:
                df = capital_market.equity_option_chain(symbol=symbol, expiry_date=expiry_date)
            else:
                df = capital_market.equity_option_chain(symbol=symbol)

            if df is None or df.empty:
                self._record_failure("No data returned from nselib")
                return IngestionResult(
                    success=False,
                    data=None,
                    source=self.name,
                    timestamp=now_ist(),
                    latency_ms=round((time.time() - start_time) * 1000, 2),
                    error="No data returned from nselib",
                )

            self._record_success()

            return IngestionResult(
                success=True,
                data=df,
                source=self.name,
                timestamp=now_ist(),
                latency_ms=round((time.time() - start_time) * 1000, 2),
                metadata={"symbol": symbol, "expiry_date": expiry_date, "row_count": len(df)},
            )

        except Exception as e:
            error_msg = f"Failed to fetch options chain: {str(e)}"
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

    def fetch_fii_dii(self) -> IngestionResult:
        """
        Fetch FII/DII flow data from nselib.

        Returns:
            IngestionResult with FII/DII flow data
        """
        start_time = time.time()

        try:
            self.logger.info("Fetching FII/DII data from nselib")

            df = capital_market.fii_dii_market_activity()

            if df is None or df.empty:
                self._record_failure("No data returned from nselib")
                return IngestionResult(
                    success=False,
                    data=None,
                    source=self.name,
                    timestamp=now_ist(),
                    latency_ms=round((time.time() - start_time) * 1000, 2),
                    error="No data returned from nselib",
                )

            self._record_success()

            return IngestionResult(
                success=True,
                data=df,
                source=self.name,
                timestamp=now_ist(),
                latency_ms=round((time.time() - start_time) * 1000, 2),
                metadata={"row_count": len(df)},
            )

        except Exception as e:
            error_msg = f"Failed to fetch FII/DII data: {str(e)}"
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

    def fetch_corporate_actions(
        self, from_date: str | None = None, to_date: str | None = None
    ) -> IngestionResult:
        """
        Fetch corporate actions data from nselib.

        Args:
            from_date: Start date (YYYY-MM-DD), None for all
            to_date: End date (YYYY-MM-DD), None for all

        Returns:
            IngestionResult with corporate actions data
        """
        start_time = time.time()

        try:
            self.logger.info("Fetching corporate actions from nselib")

            df = capital_market.corporate_actions()

            if df is None or df.empty:
                self._record_failure("No data returned from nselib")
                return IngestionResult(
                    success=False,
                    data=None,
                    source=self.name,
                    timestamp=now_ist(),
                    latency_ms=round((time.time() - start_time) * 1000, 2),
                    error="No data returned from nselib",
                )

            # Filter by date range if provided
            if from_date or to_date:
                df["announcement_date"] = pd.to_datetime(df["announcement_date"], errors="coerce")

                if from_date:
                    df = df[df["announcement_date"] >= pd.to_datetime(from_date)]

                if to_date:
                    df = df[df["announcement_date"] <= pd.to_datetime(to_date)]

            self._record_success()

            return IngestionResult(
                success=True,
                data=df,
                source=self.name,
                timestamp=now_ist(),
                latency_ms=round((time.time() - start_time) * 1000, 2),
                metadata={"from_date": from_date, "to_date": to_date, "row_count": len(df)},
            )

        except Exception as e:
            error_msg = f"Failed to fetch corporate actions: {str(e)}"
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

    def fetch_trading_calendar(self) -> IngestionResult:
        """
        Fetch trading calendar data from nselib.

        Returns:
            IngestionResult with trading calendar data
        """
        start_time = time.time()

        try:
            self.logger.info("Fetching trading calendar from nselib")

            df = capital_market.trading_holiday_list()

            if df is None or df.empty:
                self._record_failure("No data returned from nselib")
                return IngestionResult(
                    success=False,
                    data=None,
                    source=self.name,
                    timestamp=now_ist(),
                    latency_ms=round((time.time() - start_time) * 1000, 2),
                    error="No data returned from nselib",
                )

            self._record_success()

            return IngestionResult(
                success=True,
                data=df,
                source=self.name,
                timestamp=now_ist(),
                latency_ms=round((time.time() - start_time) * 1000, 2),
                metadata={"row_count": len(df)},
            )

        except Exception as e:
            error_msg = f"Failed to fetch trading calendar: {str(e)}"
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

    def fetch_security_master(self) -> IngestionResult:
        """
        Fetch security master data from nselib.

        Returns:
            IngestionResult with security master data
        """
        start_time = time.time()

        try:
            self.logger.info("Fetching security master from nselib")

            df = capital_market.equity_list()

            if df is None or df.empty:
                self._record_failure("No data returned from nselib")
                return IngestionResult(
                    success=False,
                    data=None,
                    source=self.name,
                    timestamp=now_ist(),
                    latency_ms=round((time.time() - start_time) * 1000, 2),
                    error="No data returned from nselib",
                )

            self._record_success()

            return IngestionResult(
                success=True,
                data=df,
                source=self.name,
                timestamp=now_ist(),
                latency_ms=round((time.time() - start_time) * 1000, 2),
                metadata={"row_count": len(df)},
            )

        except Exception as e:
            error_msg = f"Failed to fetch security master: {str(e)}"
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
