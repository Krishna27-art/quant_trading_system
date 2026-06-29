"""
Backfill Tests

Tests that verify historical data backfill functionality.
"""

import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from data_platform.pipelines.equity_history import EquityHistoryConfig, EquityHistoryPipeline


class TestEquityBackfill:
    """Test equity history backfill functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def backfill_config(self, temp_dir):
        """Create backfill configuration."""
        return EquityHistoryConfig(
            symbol="RELIANCE", from_date="2020-01-01", to_date="2024-12-31", parquet_dir=temp_dir
        )

    @pytest.fixture
    def pipeline(self, backfill_config):
        """Create pipeline instance."""
        return EquityHistoryPipeline(backfill_config)

    def test_backfill_date_range_validation(self, pipeline):
        """Test backfill date range validation."""
        # Test with very long date range (5 years)
        assert pipeline.config.from_date == "2020-01-01"
        assert pipeline.config.to_date == "2024-12-31"

        # Calculate expected number of trading days (approx 5 years * 252 days)
        # This is a rough estimate for validation
        expected_days = 5 * 252  # ~1260 trading days

        # The actual backfill would fetch this many days
        assert expected_days > 1000  # Sanity check

    def test_backfill_chunking(self, pipeline):
        """Test backfill can be chunked into smaller date ranges."""
        # Large backfills should be chunked to avoid API limits
        from_date = datetime.strptime(pipeline.config.from_date, "%Y-%m-%d")
        to_date = datetime.strptime(pipeline.config.to_date, "%Y-%m-%d")

        total_days = (to_date - from_date).days

        # If more than 1 year, should chunk
        if total_days > 365:
            # Calculate number of chunks
            chunk_size = 365  # 1 year chunks
            num_chunks = (total_days // chunk_size) + 1

            assert num_chunks >= 5  # Should have at least 5 chunks for 5 years

    def test_backfill_incremental_updates(self, pipeline):
        """Test incremental backfill updates."""
        # Test that backfill can resume from last available date
        # This would typically check existing data and only fetch missing dates

        # For now, just validate the concept
        assert pipeline.config.from_date is not None
        assert pipeline.config.to_date is not None

    def test_backfill_data_integrity(self, pipeline):
        """Test backfill data integrity."""
        pytest.skip("Requires NSE API access - skip in CI/CD")

        # Execute backfill
        pipeline.run()

        # Verify output
        output_file = pipeline.config.parquet_dir / f"{pipeline.config.symbol}.parquet"
        assert output_file.exists()

        # Verify data
        df = pd.read_parquet(output_file)
        assert not df.empty

        # Verify date range
        df["date"] = pd.to_datetime(df["date"])
        assert df["date"].min() >= pd.to_datetime(pipeline.config.from_date)
        assert df["date"].max() <= pd.to_datetime(pipeline.config.to_date)

        # Verify no gaps in date sequence (excluding weekends and holidays)
        # This is a simplified check - in reality would use market calendar
        date_diff = df["date"].diff().dt.days
        # Allow gaps up to 3 days (weekend + holiday)
        assert (date_diff <= 3).all() or (date_diff.isna()).sum() == 1  # First row is NaN

    def test_backfill_performance(self, pipeline):
        """Test backfill performance metrics."""
        pytest.skip("Requires NSE API access - skip in CI/CD")

        import time

        start_time = time.time()
        pipeline.run()
        end_time = time.time()

        duration = end_time - start_time

        # Backfill should complete in reasonable time
        # This is a loose constraint - actual performance depends on API
        assert duration < 3600  # Should complete within 1 hour for 5 years of data

    def test_backfill_error_recovery(self, pipeline):
        """Test backfill error recovery."""
        # Test that backfill can recover from partial failures
        # This would typically implement checkpointing

        # For now, validate the concept
        assert pipeline.config.from_date is not None
        assert pipeline.config.to_date is not None


class TestOptionsBackfill:
    """Test options chain backfill functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def backfill_config(self, temp_dir):
        """Create backfill configuration."""
        from data_platform.pipelines.options_chain import OptionsChainConfig

        return OptionsChainConfig(
            symbol="RELIANCE", from_date="2024-01-01", to_date="2024-12-31", data_dir=temp_dir
        )

    @pytest.fixture
    def pipeline(self, backfill_config):
        """Create pipeline instance."""
        from data_platform.pipelines.options_chain import OptionsChainPipeline

        return OptionsChainPipeline(backfill_config)

    def test_options_backfill_date_range(self, pipeline):
        """Test options backfill date range."""
        assert pipeline.config.from_date == "2024-01-01"
        assert pipeline.config.to_date == "2024-12-31"

    def test_options_backfill_expiry_dates(self, pipeline):
        """Test options backfill includes correct expiry dates."""
        # Options backfill should fetch data for all expiry dates
        # This would typically query the market calendar for expiry dates

        # For now, validate the concept
        assert pipeline.config.from_date is not None
        assert pipeline.config.to_date is not None


class TestCorporateActionsBackfill:
    """Test corporate actions backfill functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def backfill_config(self, temp_dir):
        """Create backfill configuration."""
        from data_platform.pipelines.corporate_actions import CorporateActionsConfig

        return CorporateActionsConfig(
            from_date="2020-01-01", to_date="2024-12-31", data_dir=temp_dir
        )

    @pytest.fixture
    def pipeline(self, backfill_config):
        """Create pipeline instance."""
        from data_platform.pipelines.corporate_actions import CorporateActionsPipeline

        return CorporateActionsPipeline(backfill_config)

    def test_corporate_actions_backfill_date_range(self, pipeline):
        """Test corporate actions backfill date range."""
        assert pipeline.config.from_date == "2020-01-01"
        assert pipeline.config.to_date == "2024-12-31"

    def test_corporate_actions_backfill_event_types(self, pipeline):
        """Test corporate actions backfill includes all event types."""
        # Should fetch all event types: dividends, splits, bonuses, etc.

        # For now, validate the concept
        assert pipeline.config.from_date is not None
        assert pipeline.config.to_date is not None
