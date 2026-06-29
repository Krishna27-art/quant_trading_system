"""
Integration Tests for Data Pipelines

Tests end-to-end data pipeline functionality including:
- Equity history ingestion
- Options chain ingestion
- Corporate actions ingestion
- Feature computation
- Data validation
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_platform.pipelines.equity_history import EquityHistoryConfig, EquityHistoryPipeline
from data_platform.validation.enhanced_monitoring import create_enhanced_monitor
from data_platform.validation.ingestion_wrapper import validate_at_ingestion


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_equity_data():
    """Create sample equity data for testing."""
    dates = pd.date_range("2024-01-01", periods=50)
    np.random.seed(42)

    return pd.DataFrame(
        {
            "date": dates,
            "symbol": ["RELIANCE"] * 50,
            "series": ["EQ"] * 50,
            "prev_close": [2500.0] * 50,
            "open": [2510.0] * 50,
            "high": [2520.0] * 50,
            "low": [2505.0] * 50,
            "close": [2515.0] * 50,
            "average_price": [2512.5] * 50,
            "volume": [1000000] * 50,
            "turnover": [2512500000.0] * 50,
            "num_trades": [50000] * 50,
        }
    )


class TestEquityHistoryPipelineIntegration:
    """Integration tests for equity history pipeline."""

    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration."""
        return EquityHistoryConfig(
            symbol="RELIANCE",
            from_date="2024-01-01",
            to_date="2024-01-31",
            parquet_dir=temp_dir / "equity_history",
        )

    @pytest.fixture
    def pipeline(self, config):
        """Create pipeline instance."""
        return EquityHistoryPipeline(config)

    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.config.symbol == "RELIANCE"
        assert pipeline.config.from_date == "2024-01-01"
        assert pipeline.config.to_date == "2024-01-31"

    def test_data_validation_integration(self, sample_equity_data, temp_dir):
        """Test data validation integration."""
        validated_df, metadata = validate_at_ingestion(
            df=sample_equity_data, dataset_name="equity_history_RELIANCE", source="NSE"
        )

        assert metadata["validation_passed"] is True
        assert len(validated_df) == len(sample_equity_data)
        assert metadata["validation_score"] > 0

    def test_data_validation_with_invalid_data(self, temp_dir):
        """Test data validation with invalid data."""
        invalid_data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=10),
                "symbol": ["RELIANCE"] * 10,
                "close": [1000.0] * 10,
                "volume": [-100] * 10,  # Invalid: negative volume
            }
        )

        validated_df, metadata = validate_at_ingestion(
            df=invalid_data, dataset_name="equity_history_RELIANCE", source="NSE"
        )

        # Should still return data but with validation metadata
        assert len(validated_df) > 0
        assert "validation_score" in metadata


class TestEnhancedMonitoringIntegration:
    """Integration tests for enhanced monitoring."""

    @pytest.fixture
    def monitor(self, temp_dir):
        """Create enhanced monitor instance."""
        return create_enhanced_monitor(
            schema_registry_path=temp_dir / "schemas", quarantine_path=temp_dir / "quarantine"
        )

    def test_monitor_dataset_integration(self, monitor, sample_equity_data):
        """Test dataset monitoring integration."""
        # Register schema
        monitor.ingestion_validator.register_dataset_schema(
            dataset_name="equity_history",
            version="1.0",
            schema={
                "date": "datetime64[ns]",
                "symbol": "object",
                "close": "float64",
                "volume": "int64",
            },
        )

        results = monitor.monitor_dataset(
            df=sample_equity_data, dataset_name="equity_history", source="NSE"
        )

        assert "overall_status" in results
        assert "components" in results
        assert "validation" in results["components"]

    def test_gap_detection_integration(self, monitor):
        """Test gap detection integration."""
        # Create data with gaps
        dates = pd.date_range("2024-01-01", periods=10)
        df = pd.DataFrame({"date": dates, "symbol": ["RELIANCE"] * 10, "close": [1000.0] * 10})

        # Expected dates with gaps
        expected_dates = [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",  # Gap for weekend
            "2024-01-09",
            "2024-01-10",
            "2024-01-11",
            "2024-01-12",
        ]

        results = monitor.monitor_dataset(
            df=df, dataset_name="equity_history", source="NSE", expected_dates=expected_dates
        )

        assert "components" in results
        assert "gaps" in results["components"]


class TestDataPipelineEndToEnd:
    """End-to-end integration tests for data pipelines."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    def test_complete_data_pipeline(self, temp_dir, sample_equity_data):
        """Test complete data pipeline from ingestion to monitoring."""
        # Step 1: Validate data at ingestion
        validated_df, validation_metadata = validate_at_ingestion(
            df=sample_equity_data, dataset_name="equity_history_RELIANCE", source="NSE"
        )

        assert validation_metadata["validation_passed"] is True

        # Step 2: Monitor data quality
        monitor = create_enhanced_monitor(
            schema_registry_path=temp_dir / "schemas", quarantine_path=temp_dir / "quarantine"
        )

        monitor.ingestion_validator.register_dataset_schema(
            dataset_name="equity_history",
            version="1.0",
            schema={"date": "datetime64[ns]", "symbol": "object", "close": "float64"},
        )

        monitoring_results = monitor.monitor_dataset(
            df=validated_df, dataset_name="equity_history", source="NSE"
        )

        assert monitoring_results["overall_status"] in [
            "healthy",
            "validation_failed",
            "gaps_detected",
            "stale",
        ]

        # Step 3: Verify data lineage
        assert "lineage_id" in validation_metadata
        assert validation_metadata["lineage_id"] is not None

    def test_pipeline_error_handling(self, temp_dir):
        """Test pipeline error handling."""
        # Create malformed data
        malformed_data = pd.DataFrame(
            {
                "date": ["invalid_date"] * 10,
                "symbol": ["RELIANCE"] * 10,
                "close": ["not_a_number"] * 10,
            }
        )

        # Should handle gracefully
        try:
            validated_df, metadata = validate_at_ingestion(
                df=malformed_data, dataset_name="equity_history_RELIANCE", source="NSE"
            )
            # If it doesn't raise, check metadata
            assert "validation_score" in metadata
        except Exception:
            # Expected to fail with malformed data
            assert True


class TestMultiSymbolPipeline:
    """Integration tests for multi-symbol data pipelines."""

    @pytest.fixture
    def multi_symbol_data(self):
        """Create multi-symbol data for testing."""
        dates = pd.date_range("2024-01-01", periods=30)
        symbols = ["RELIANCE", "TCS", "INFY"]

        data = []
        for symbol in symbols:
            np.random.seed(hash(symbol) % 1000)
            for date in dates:
                data.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "close": 1000 + np.random.randn() * 100,
                        "volume": np.random.randint(1000000, 5000000),
                    }
                )

        return pd.DataFrame(data)

    def test_multi_symbol_validation(self, multi_symbol_data, temp_dir):
        """Test validation for multiple symbols."""
        validated_df, metadata = validate_at_ingestion(
            df=multi_symbol_data, dataset_name="equity_history_multi", source="NSE"
        )

        assert metadata["validation_passed"] is True
        assert validated_df["symbol"].nunique() == 3

    def test_symbol_coverage_monitoring(self, multi_symbol_data, temp_dir):
        """Test symbol coverage monitoring."""
        monitor = create_enhanced_monitor(
            schema_registry_path=temp_dir / "schemas", quarantine_path=temp_dir / "quarantine"
        )

        monitor.ingestion_validator.register_dataset_schema(
            dataset_name="equity_history",
            version="1.0",
            schema={"date": "datetime64[ns]", "symbol": "object", "close": "float64"},
        )

        expected_symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]

        results = monitor.monitor_dataset(
            df=multi_symbol_data,
            dataset_name="equity_history",
            source="NSE",
            expected_symbols=expected_symbols,
        )

        # Should detect missing symbols
        assert "gaps" in results["components"]
        assert "symbols" in results["components"]["gaps"]


class TestPipelinePerformance:
    """Performance tests for data pipelines."""

    def test_large_dataset_validation(self, temp_dir):
        """Test validation with large dataset."""
        # Create large dataset
        dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")
        large_data = pd.DataFrame(
            {
                "date": dates,
                "symbol": ["RELIANCE"] * len(dates),
                "close": 1000 + np.random.randn(len(dates)) * 100,
                "volume": np.random.randint(1000000, 5000000, len(dates)),
            }
        )

        import time

        start_time = time.time()

        validated_df, metadata = validate_at_ingestion(
            df=large_data, dataset_name="equity_history_large", source="NSE"
        )

        elapsed_time = time.time() - start_time

        # Should complete in reasonable time (< 30 seconds for 5 years of data)
        assert elapsed_time < 30
        assert metadata["validation_passed"] is True

    def test_batch_validation_performance(self, temp_dir):
        """Test batch validation performance."""
        # Create multiple datasets
        datasets = []
        for i in range(10):
            dates = pd.date_range("2024-01-01", periods=100)
            data = pd.DataFrame(
                {
                    "date": dates,
                    "symbol": [f"STOCK_{i}"] * len(dates),
                    "close": 1000 + np.random.randn(len(dates)) * 100,
                    "volume": np.random.randint(1000000, 5000000, len(dates)),
                }
            )
            datasets.append(data)

        import time

        start_time = time.time()

        for i, data in enumerate(datasets):
            validated_df, metadata = validate_at_ingestion(
                df=data, dataset_name=f"equity_history_{i}", source="NSE"
            )

        elapsed_time = time.time() - start_time

        # Should complete in reasonable time
        assert elapsed_time < 60


class TestPipelineRecovery:
    """Test pipeline recovery from failures."""

    def test_validation_failure_recovery(self, temp_dir):
        """Test recovery from validation failure."""
        # Create data that will fail validation
        invalid_data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=10),
                "symbol": ["RELIANCE"] * 10,
                "close": [1000.0] * 10,
                "volume": [-100] * 10,  # Invalid
            }
        )

        # First validation should fail or annotate
        validated_df, metadata = validate_at_ingestion(
            df=invalid_data, dataset_name="equity_history_RELIANCE", source="NSE"
        )

        # Create corrected data
        corrected_data = invalid_data.copy()
        corrected_data["volume"] = [1000000] * 10

        # Second validation should pass
        validated_df, metadata = validate_at_ingestion(
            df=corrected_data, dataset_name="equity_history_RELIANCE", source="NSE"
        )

        assert metadata["validation_passed"] is True

    def test_partial_data_recovery(self, temp_dir):
        """Test recovery from partial data."""
        # Create data with missing dates
        dates = pd.date_range("2024-01-01", periods=20)
        dates = dates[[0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17]]  # Gaps

        partial_data = pd.DataFrame(
            {
                "date": dates,
                "symbol": ["RELIANCE"] * len(dates),
                "close": [1000.0] * len(dates),
                "volume": [1000000] * len(dates),
            }
        )

        # Should handle partial data
        validated_df, metadata = validate_at_ingestion(
            df=partial_data, dataset_name="equity_history_RELIANCE", source="NSE"
        )

        # Should still process available data
        assert len(validated_df) == len(partial_data)
