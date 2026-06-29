import unittest.mock as mock

from data.fred_macro import fetch_fred_observation, get_macro_indicators


def test_fred_observation_parsing():
    with mock.patch("requests.get") as mock_get:
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "observations": [
                {"date": "2026-06-01", "value": "5.21"},
                {"date": "2026-05-01", "value": "5.10"},
            ]
        }
        mock_get.return_value = mock_response

        val = fetch_fred_observation("INDCPIALLMINMEI", "mock_key")
        assert val == 5.21


def test_macro_indicators_consolidation():
    # If FRED_API_KEY is not set, it should fallback to defaults
    with mock.patch.dict("os.environ", {"FRED_API_KEY": ""}):
        indicators = get_macro_indicators()
        assert "india_cpi" in indicators
        assert "repo_rate" in indicators
        assert indicators["repo_rate"] == 6.50
        assert indicators["india_cpi"] == 5.0
