import os
from datetime import datetime

import pandas as pd
import requests

from utils.logger import get_logger

logger = get_logger("macro_fred")


class FREDDataPipeline:
    """
    Pipeline to fetch macroeconomic data from the Federal Reserve Economic Data (FRED) API.
    Key series:
    - CPIAUCSL: CPI
    - GDP: Gross Domestic Product
    - UNRATE: Unemployment Rate
    - FEDFUNDS: Federal Funds Effective Rate
    - DGS10: 10-Year Treasury Constant Maturity Rate
    - T10Y2Y: 10-Year minus 2-Year Treasury Constant Maturity
    """

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        if not self.api_key:
            logger.warning("FRED_API_KEY not found. Pipeline will return mocked data or fail.")

    def fetch_series(self, series_id: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetches the latest data points for a specific FRED series.
        """
        if not self.api_key:
            return self._mock_data(series_id)

        try:
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": limit,
            }
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            observations = data.get("observations", [])
            if not observations:
                return pd.DataFrame()

            df = pd.DataFrame(observations)
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["value"])
            return df[["date", "value"]].sort_values("date")

        except Exception as e:
            logger.error(f"Error fetching FRED series {series_id}: {e}")
            return self._mock_data(series_id)

    def _mock_data(self, series_id: str) -> pd.DataFrame:
        """Fallback mock data if API key is missing or request fails."""
        logger.info(f"Returning mocked data for FRED series: {series_id}")
        dates = pd.date_range(end=datetime.now(), periods=10, freq="D")
        return pd.DataFrame({"date": dates, "value": [2.5] * 10})  # Dummy value
