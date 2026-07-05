import unittest.mock as mock
import pandas as pd
from data_platform.pipelines.india_macro import get_macro_indicators
from data_platform.pipelines.macro_fred import FREDDataPipeline


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

        # Test FREDDataPipeline
        pipeline = FREDDataPipeline(api_key="mock_key")
        df = pipeline.fetch_series("INDCPIALLMINMEI")
        assert len(df) == 2
        assert df["value"].iloc[0] == 5.10  # sorted asc by date
        assert df["value"].iloc[1] == 5.21


def test_macro_indicators_consolidation():
    indicators = get_macro_indicators()
    assert "india_cpi" in indicators
    assert "repo_rate" in indicators
    assert indicators["repo_rate"] == 6.50
    assert indicators["india_cpi"] == 5.0
