from typing import Any

import requests

from utils.logger import get_logger
from utils.upstox_helper import get_instrument_key, get_upstox_client_config

logger = get_logger("nse_options")


class NSEOptionsPipeline:
    """
    Fetches live F&O option chains from Upstox API v2.
    Extracts Open Interest (OI), Implied Volatility (IV), and LTP.
    """

    def __init__(self):
        pass

    def fetch_option_chain(self, symbol: str = "NIFTY") -> dict[str, Any]:
        """
        Fetches the option chain for the given symbol (e.g. NIFTY, BANKNIFTY, RELIANCE) from Upstox.
        """
        config = get_upstox_client_config()
        if not config:
            logger.warning("Upstox config not available, using mocked option chain")
            return self._mock_option_chain(symbol)

        underlying_key = get_instrument_key(symbol)
        if not underlying_key:
            # Fallback format for indices
            if symbol == "NIFTY":
                underlying_key = "NSE_INDEX|Nifty 50"
            elif symbol == "BANKNIFTY":
                underlying_key = "NSE_INDEX|Nifty Bank"
            else:
                return self._mock_option_chain(symbol)

        # Fetch expiries to find nearest expiry
        try:
            from data.upstox_options import get_nearest_expiry

            expiry = get_nearest_expiry(symbol)
            if not expiry:
                return self._mock_option_chain(symbol)

            url = "https://api.upstox.com/v2/option/chain"
            headers = {
                "accept": "application/json",
                "Authorization": f"Bearer {config.access_token}",
            }
            params = {"instrument_key": underlying_key, "expiry_date": expiry}

            resp = requests.get(url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()

            chain_data = resp.json().get("data", [])

            # Format the output into the structure expected by downstream services
            records_data = []
            for strike in chain_data:
                strike_price = float(strike.get("strike_price", 0))
                call_opt = strike.get("call_options", {})
                put_opt = strike.get("put_options", {})

                ce_data = {}
                pe_data = {}

                if call_opt:
                    call_market = call_opt.get("market_data", {})
                    ce_data = {
                        "openInterest": int(call_market.get("oi", 0)),
                        "impliedVolatility": float(
                            call_opt.get("option_greeks", {}).get("iv", 0.0)
                        ),
                        "lastPrice": float(call_market.get("ltp", 0.0)),
                    }

                if put_opt:
                    put_market = put_opt.get("market_data", {})
                    pe_data = {
                        "openInterest": int(put_market.get("oi", 0)),
                        "impliedVolatility": float(put_opt.get("option_greeks", {}).get("iv", 0.0)),
                        "lastPrice": float(put_market.get("ltp", 0.0)),
                    }

                records_data.append(
                    {
                        "strikePrice": strike_price,
                        "expiryDate": expiry,
                        "CE": ce_data,
                        "PE": pe_data,
                    }
                )

            logger.info(f"Successfully fetched option chain for {symbol} expiry {expiry}")
            return {"records": {"expiryDates": [expiry], "data": records_data}}

        except Exception as e:
            logger.error(f"Failed to fetch Upstox option chain for {symbol}: {e}")
            return self._mock_option_chain(symbol)

    def fetch_live_quote(self, symbol: str) -> dict[str, Any]:
        """
        Fetches live stock quotes (price, bid/ask spread, volume) from Upstox.
        """
        config = get_upstox_client_config()
        if not config:
            logger.warning("Upstox config not available, using mocked live quote")
            return {"priceInfo": {"lastPrice": 150.0}, "tradedVolume": 1000}

        key = get_instrument_key(symbol)
        if not key:
            return {"priceInfo": {"lastPrice": 150.0}, "tradedVolume": 1000}

        try:
            url = "https://api.upstox.com/v2/market-quote/quotes"
            headers = {
                "accept": "application/json",
                "Authorization": f"Bearer {config.access_token}",
            }
            params = {"instrument_key": key}

            resp = requests.get(url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()

            data = resp.json().get("data", {}).get(key, {})
            if not data:
                return {"priceInfo": {"lastPrice": 150.0}, "tradedVolume": 1000}

            last_price = float(data.get("last_price", 0.0))
            volume = int(data.get("volume", 0))

            logger.info(f"Successfully fetched live quote for {symbol} from Upstox")
            return {"priceInfo": {"lastPrice": last_price}, "tradedVolume": volume}
        except Exception as e:
            logger.error(f"Failed to fetch live quote for {symbol} from Upstox: {e}")
            return {"priceInfo": {"lastPrice": 150.0}, "tradedVolume": 1000}

    def _mock_option_chain(self, symbol: str) -> dict[str, Any]:
        """Mock payload for testing or if Upstox is unavailable."""
        return {
            "records": {
                "expiryDates": ["2026-06-30"],
                "data": [
                    {
                        "strikePrice": 22000,
                        "expiryDate": "2026-06-30",
                        "CE": {
                            "openInterest": 50000,
                            "impliedVolatility": 12.5,
                            "lastPrice": 150.0,
                        },
                        "PE": {
                            "openInterest": 30000,
                            "impliedVolatility": 14.0,
                            "lastPrice": 120.0,
                        },
                    }
                ],
            }
        }
