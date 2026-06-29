"""
Integration Tests for Equity History Pipeline

Tests end-to-end pipeline functionality with real data sources.
"""

import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from data_platform.pipelines.equity_history import EquityHistoryConfig, EquityHistoryPipeline


class TestEquityHistoryPipelineIntegration:
    """Integration tests for equity history pipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_config(self, temp_dir):
        """Create mock configuration with temporary directory."""
        return EquityHistoryConfig(
            symbol="RELIANCE", from_date="2024-01-01", to_date="2024-01-05", parquet_dir=temp_dir
        )

    @pytest.fixture
    def pipeline(self, mock_config):
        """Create pipeline instance."""
        return EquityHistoryPipeline(mock_config)

    def test_pipeline_end_to_end(self, pipeline):
        """Test complete pipeline execution."""
        # This test requires real NSE API access
        # Skip if API is unavailable
        pytest.skip("Requires NSE API access - skip in CI/CD")

        # Execute pipeline
        pipeline.run()

        # Verify output file exists
        assert pipeline.config.parquet_dir.exists()

        # Verify parquet file can be read
        output_file = pipeline.config.parquet_dir / f"{pipeline.config.symbol}.parquet"
        assert output_file.exists()

        # Verify data quality
        df = pd.read_parquet(output_file)
        assert not df.empty
        assert "close" in df.columns
        assert len(df) > 0

    def test_pipeline_with_validation(self, pipeline):
        """Test pipeline with data validation."""
        pytest.skip("Requires NSE API access - skip in CI/CD")

        # Execute pipeline
        pipeline.run()

        # Verify validation passed
        output_file = pipeline.config.parquet_dir / f"{pipeline.config.symbol}.parquet"
        df = pd.read_parquet(output_file)

        # Check for negative values
        assert (df["close"] >= 0).all()
        assert (df["volume"] >= 0).all()

        # Check for required columns
        required_cols = ["date", "open", "high", "low", "close", "volume"]
        for col in required_cols:
            assert col in df.columns

    def test_pipeline_error_handling(self, pipeline):
        """Test pipeline error handling."""
        # Test with invalid symbol
        pipeline.config.symbol = "INVALID_SYMBOL"

        # Should not crash, but should log error
        try:
            pipeline.run()
        except Exception as e:
            # Expected to fail with invalid symbol
            assert "INVALID_SYMBOL" in str(e) or "error" in str(e).lower()

    def test_pipeline_date_range_validation(self, pipeline):
        """Test pipeline date range validation."""
        # Test with invalid date range
        pipeline.config.from_date = "2024-12-31"
        pipeline.config.to_date = "2024-01-01"  # End before start

        # Should raise validation error
        with pytest.raises(ValueError):
            pipeline.run()


class TestOptionsChainPipelineIntegration:
    """Integration tests for options chain pipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_config(self, temp_dir):
        """Create mock configuration."""
        from data_platform.pipelines.options_chain import OptionsChainConfig

        return OptionsChainConfig(
            symbol="RELIANCE", from_date="2024-01-01", to_date="2024-01-05", data_dir=temp_dir
        )

    @pytest.fixture
    def pipeline(self, mock_config):
        """Create pipeline instance."""
        from data_platform.pipelines.options_chain import OptionsChainPipeline

        return OptionsChainPipeline(mock_config)

    def test_options_pipeline_end_to_end(self, pipeline):
        """Test complete options pipeline execution."""
        pytest.skip("Requires NSE API access - skip in CI/CD")

        # Execute pipeline
        pipeline.run()

        # Verify output
        assert pipeline.config.data_dir.exists()

        # Verify data quality
        output_file = pipeline.config.data_dir / f"{pipeline.config.symbol}_options_chain.parquet"
        if output_file.exists():
            df = pd.read_parquet(output_file)
            assert not df.empty


class TestCorporateActionsPipelineIntegration:
    """Integration tests for corporate actions pipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_config(self, temp_dir):
        """Create mock configuration."""
        from data_platform.pipelines.corporate_actions import CorporateActionsConfig

        return CorporateActionsConfig(
            from_date="2024-01-01", to_date="2024-12-31", data_dir=temp_dir
        )

    @pytest.fixture
    def pipeline(self, mock_config):
        """Create pipeline instance."""
        from data_platform.pipelines.corporate_actions import CorporateActionsPipeline

        return CorporateActionsPipeline(mock_config)

    def test_corporate_actions_pipeline_end_to_end(self, pipeline):
        """Test complete corporate actions pipeline execution."""
        pytest.skip("Requires NSE API access - skip in CI/CD")

        # Execute pipeline
        pipeline.run()

        # Verify output
        assert pipeline.config.data_dir.exists()

        # Verify data quality
        output_file = pipeline.config.data_dir / pipeline.config.parquet_file
        if output_file.exists():
            df = pd.read_parquet(output_file)
            assert not df.empty
            assert "symbol" in df.columns
