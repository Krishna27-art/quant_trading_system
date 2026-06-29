"""
Unit Tests for Equity History Pipeline

Tests individual components and functions of the equity history pipeline.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data_platform.pipelines.equity_history import EquityHistoryConfig, EquityHistoryPipeline
from data_platform.validation.equity_rules import EquityValidator


class TestEquityHistoryConfig:
    """Test EquityHistoryConfig configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EquityHistoryConfig()
        assert config.symbol == "RELIANCE"
        assert config.from_date == "2024-01-01"
        assert config.to_date is not None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = EquityHistoryConfig(symbol="TCS", from_date="2023-01-01", to_date="2023-12-31")
        assert config.symbol == "TCS"
        assert config.from_date == "2023-01-01"
        assert config.to_date == "2023-12-31"


class TestEquityHistoryPipeline:
    """Test EquityHistoryPipeline methods."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return EquityHistoryConfig(symbol="RELIANCE", from_date="2024-01-01")

    @pytest.fixture
    def pipeline(self, mock_config):
        """Create pipeline instance."""
        # Mock ClickHouseClient, IngestionEngine, etc. to avoid real connections in init
        with (
            patch("data_platform.pipelines.equity_history.ClickHouseClient"),
            patch("data_platform.pipelines.equity_history.IngestionEngine"),
            patch("data_platform.pipelines.equity_history.RawBronzeLayer"),
            patch("data_platform.pipelines.equity_history.IngestionLineage"),
        ):
            pipe = EquityHistoryPipeline(mock_config)
            return pipe

    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.config.symbol == "RELIANCE"
        assert pipeline.config.from_date == "2024-01-01"

    def test_download_ohlcv_success(self, pipeline):
        """Test successful data download using IngestionEngine mock."""
        # Mock IngestionEngine result
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.source = "nselib"
        mock_result.latency_ms = 120.0
        mock_result.error = None
        mock_result.metadata = {}

        mock_df = pd.DataFrame(
            {
                "Symbol": ["RELIANCE"] * 5,
                "Series": ["EQ"] * 5,
                "Date": ["01-Jan-2024", "02-Jan-2024", "03-Jan-2024", "04-Jan-2024", "05-Jan-2024"],
                "PrevClose": [2400.0] * 5,
                "OpenPrice": [2400.0, 2410.0, 2420.0, 2430.0, 2440.0],
                "HighPrice": [2420.0, 2430.0, 2440.0, 2450.0, 2460.0],
                "LowPrice": [2390.0, 2400.0, 2410.0, 2420.0, 2430.0],
                "LastPrice": [2410.0, 2420.0, 2430.0, 2440.0, 2450.0],
                "ClosePrice": [2410.0, 2420.0, 2430.0, 2440.0, 2450.0],
                "AveragePrice": [2405.0] * 5,
                "TotalTradedQuantity": [1000000] * 5,
                "Turnover₹": [2405000000.0] * 5,
                "No.ofTrades": [50000] * 5,
            }
        )
        mock_result.data = mock_df
        pipeline.ingestion_engine.fetch_equity_history.return_value = mock_result

        # Mock validate_at_ingestion wrapper
        with patch("data_platform.pipelines.equity_history.validate_at_ingestion") as mock_val:
            mock_val.return_value = (
                mock_df.rename(
                    columns={
                        "Symbol": "symbol",
                        "Series": "series",
                        "Date": "date",
                        "PrevClose": "prev_close",
                        "OpenPrice": "open",
                        "HighPrice": "high",
                        "LowPrice": "low",
                        "LastPrice": "last_price",
                        "ClosePrice": "close",
                        "AveragePrice": "average_price",
                        "TotalTradedQuantity": "volume",
                        "Turnover₹": "turnover",
                        "No.ofTrades": "num_trades",
                    }
                ),
                {"validation_passed": True, "validation_score": 100.0},
            )

            result = pipeline.download_ohlcv()

            assert not result.empty
            assert len(result) == 5
            assert "close" in result.columns
            pipeline.ingestion_engine.fetch_equity_history.assert_called_once()

    def test_download_ohlcv_failure(self, pipeline):
        """Test data download failure."""
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "Connection timeout"
        pipeline.ingestion_engine.fetch_equity_history.return_value = mock_result

        with pytest.raises(Exception):
            pipeline.download_ohlcv()

    def test_validate_data_success(self, pipeline):
        """Test successful data validation."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5),
                "symbol": ["RELIANCE"] * 5,
                "open": [100, 101, 102, 103, 104],
                "high": [105, 106, 107, 108, 109],
                "low": [95, 96, 97, 98, 99],
                "close": [100, 101, 102, 103, 104],
                "volume": [1000000, 1100000, 1200000, 1300000, 1400000],
            }
        )

        validator = EquityValidator()
        result = validator.validate(df)
        assert result.is_acceptable()

    def test_validate_data_missing_columns(self, pipeline):
        """Test validation with missing columns."""
        df = pd.DataFrame(
            {"date": pd.date_range("2024-01-01", periods=5), "close": [100, 101, 102, 103, 104]}
        )

        validator = EquityValidator()
        result = validator.validate(df)

        assert not result.is_acceptable()

    def test_validate_data_negative_values(self, pipeline):
        """Test validation with negative values."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5),
                "symbol": ["RELIANCE"] * 5,
                "open": [100, 101, 102, 103, 104],
                "high": [105, 106, 107, 108, 109],
                "low": [95, 96, 97, 98, 99],
                "close": [100, 101, 102, 103, 104],
                "volume": [-1000000, 1100000, 1200000, 1300000, 1400000],
            }
        )

        validator = EquityValidator()
        result = validator.validate(df)
        assert result.failed_count > 0


class TestEquityValidator:
    """Test EquityValidator validation rules."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return EquityValidator()

    def test_validate_required_columns(self, validator):
        """Test required columns validation."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5),
                "symbol": ["RELIANCE"] * 5,
                "open": [100, 101, 102, 103, 104],
                "high": [105, 106, 107, 108, 109],
                "low": [95, 96, 97, 98, 99],
                "close": [100, 101, 102, 103, 104],
                "volume": [1000000, 1100000, 1200000, 1300000, 1400000],
            }
        )

        result = validator.validate(df)
        assert result.is_acceptable()

    def test_validate_no_negative_prices(self, validator):
        """Test negative price validation."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5),
                "symbol": ["RELIANCE"] * 5,
                "open": [100, 101, 102, 103, 104],
                "high": [105, 106, 107, 108, 109],
                "low": [95, 96, 97, 98, 99],
                "close": [100, 101, 102, 103, 104],
                "volume": [1000000, 1100000, 1200000, 1300000, 1400000],
            }
        )

        result = validator.validate(df)
        assert result.is_acceptable()

    def test_validate_no_negative_volume(self, validator):
        """Test negative volume validation."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5),
                "symbol": ["RELIANCE"] * 5,
                "open": [100, 101, 102, 103, 104],
                "high": [105, 106, 107, 108, 109],
                "low": [95, 96, 97, 98, 99],
                "close": [100, 101, 102, 103, 104],
                "volume": [1000000, 1100000, 1200000, 1300000, 1400000],
            }
        )

        result = validator.validate(df)
        assert result.is_acceptable()

    def test_validate_high_low_consistency(self, validator):
        """Test high-low price consistency."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5),
                "symbol": ["RELIANCE"] * 5,
                "open": [100, 101, 102, 103, 104],
                "high": [95, 106, 107, 108, 109],  # First High < Open
                "low": [95, 96, 97, 98, 99],
                "close": [100, 101, 102, 103, 104],
                "volume": [1000000, 1100000, 1200000, 1300000, 1400000],
            }
        )

        result = validator.validate(df)
        assert not result.is_acceptable()
