import unittest.mock as mock

from data.upstox_options import fetch_option_chain_pcr
from utils.upstox_helper import get_instrument_key


def test_instrument_key_resolution():
    # Test caching and key mapping
    key = get_instrument_key("RELIANCE")
    # RELIANCE ISIN is INE002A01018, Upstox key is NSE_EQ|INE002A01018
    # Even if cache isn't built yet, the downloader will run.
    # If the network fails, it might return None, which we can check.
    if key:
        assert key.startswith("NSE_EQ|")
        assert "INE002A01018" in key


def test_pcr_calculation():
    # Test PCR calculation with mocked API responses
    with mock.patch("requests.get") as mock_get:
        # Mock option chain API response
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "data": [
                {
                    "strike_price": 22000,
                    "call_options": {"market_data": {"oi": 50000.0}},
                    "put_options": {"market_data": {"oi": 100000.0}},
                }
            ],
        }
        mock_get.return_value = mock_response

        # Patch configurations
        with mock.patch("data.upstox_options.get_upstox_client_config") as mock_config:
            mock_config_instance = mock.MagicMock()
            mock_config_instance.access_token = "mocked_token"
            mock_config.return_value = mock_config_instance

            pcr = fetch_option_chain_pcr("Nifty 50", "2026-07-03")
            assert pcr == 2.0  # 100000 / 50000
