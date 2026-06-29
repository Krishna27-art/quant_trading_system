from datetime import datetime, timedelta

import requests

from utils.logger import get_logger
from utils.secrets import get_secret

logger = get_logger("news_finnhub")


class FinnhubNewsPipeline:
    """
    Pipeline to fetch company news and sentiment data from the Finnhub API.
    """

    BASE_URL = "https://finnhub.io/api/v1/company-news"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or get_secret("FINNHUB_API_KEY")
        if not self.api_key:
            logger.warning("FINNHUB_API_KEY not found. Pipeline will fail.")

    def fetch_company_news(self, symbol: str, days_back: int = 7) -> list[dict]:
        """
        Fetches company news for a given symbol over the last `days_back` days.
        """
        if not self.api_key:
            raise RuntimeError(f"Cannot fetch news for {symbol}: FINNHUB_API_KEY is missing.")

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            params = {
                "symbol": symbol,
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
                "token": self.api_key,
            }
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data

        except Exception as e:
            logger.error(f"Error fetching Finnhub news for {symbol}: {e}", exc_info=True)
            raise RuntimeError(f"Error fetching Finnhub news for {symbol}") from e
