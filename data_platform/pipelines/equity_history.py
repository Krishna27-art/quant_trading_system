"""
NSE Equity History Pipeline

Downloads OHLCV data from NSE and saves to Parquet and DuckDB.
"""

from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
from pydantic import BaseModel, Field

from config.settings import BRONZE_EQUITY_HISTORY_DIR, CLICKHOUSE_DATABASE, DB_LOCK, DB_PATH
from data_platform.sources.ingestion.ingestion_engine import IngestionEngine
from data_platform.sources.ingestion.lineage import IngestionLineage
from data_platform.sources.ingestion.raw_bronze import RawBronzeLayer
from data_platform.validation.equity_rules import EquityValidator
from data_platform.validation.ingestion_wrapper import validate_at_ingestion
from utils.api_helpers import nse_api_call
from utils.clickhouse_client import ClickHouseClient
from utils.data_lineage import compute_checksum, write_lineage_record
from utils.init_directories import init_directories
from utils.logger import get_logger
from utils.metadata_catalog import MetadataCatalog
from utils.time_utils import now_ist
from utils.versioned_datasets import VersionedDataset

# Initialize directories
init_directories()

# Configure logging
logger = get_logger("equity_history")


class EquityHistoryConfig(BaseModel):
    """Configuration for equity history pipeline."""

    symbol: str = Field(default="RELIANCE", description="NSE stock symbol (e.g., RELIANCE)")
    from_date: str = Field(default="2024-01-01", description="Start date in YYYY-MM-DD format")
    to_date: str = Field(
        default_factory=lambda: now_ist().strftime("%Y-%m-%d"),
        description="End date in YYYY-MM-DD format",
    )
    parquet_dir: Path = Field(
        default=BRONZE_EQUITY_HISTORY_DIR, description="Directory to save parquet files"
    )
    duckdb_path: Path = Field(default=DB_PATH, description="Path to DuckDB database")


class EquityHistoryPipeline:
    """Pipeline for downloading and storing NSE equity history data."""

    def __init__(self, config: EquityHistoryConfig):
        self.config = config
        self.ch_client = ClickHouseClient()
        self.ingestion_engine = IngestionEngine()
        self.raw_bronze = RawBronzeLayer()
        self.lineage = IngestionLineage()

    @nse_api_call
    def download_ohlcv(self) -> pd.DataFrame:
        """
        Download OHLCV data from NSE.

        Returns:
            DataFrame with OHLCV data

        Raises:
            Exception: If download fails or data is invalid
        """
        logger.info(
            f"Downloading OHLCV data for {self.config.symbol} from {self.config.from_date} to {self.config.to_date}"
        )

        try:
            # Use institutional ingestion engine with fallback logic
            result = self.ingestion_engine.fetch_equity_history(
                symbol=self.config.symbol,
                from_date=self.config.from_date,
                to_date=self.config.to_date,
                use_fallback=True,
            )

            if not result.success:
                raise ValueError(f"Ingestion failed: {result.error}")

            df = result.data

            if df is None or df.empty:
                raise ValueError(f"No data returned for symbol {self.config.symbol}")

            logger.info(f"Downloaded {len(df)} records from {result.source}")
            logger.info(f"DataFrame shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")
            logger.info(f"Latency: {result.latency_ms}ms")

            # Store raw response in bronze layer
            dataset_name = f"equity_history_{self.config.symbol}"
            try:
                self.raw_bronze.store_raw_response(
                    dataset=dataset_name,
                    source=result.source,
                    raw_data=df,
                    metadata=result.metadata,
                )
                logger.info("Stored raw response to bronze layer")
            except Exception as e:
                logger.warning(f"Failed to store raw response: {str(e)}")

            # Record lineage
            try:
                self.lineage.record_ingestion(
                    dataset=dataset_name,
                    source=result.source,
                    success=result.success,
                    latency_ms=result.latency_ms,
                    metadata=result.metadata,
                    error=result.error,
                    fallback_used=(result.source != "nselib"),
                )
            except Exception as e:
                logger.warning(f"Failed to record lineage: {str(e)}")

            # Rename columns to standard format
            column_mapping = {
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
            df = df.rename(columns=column_mapping)

            # Convert numeric columns using institutional-grade vectorized operations
            numeric_columns = [
                "prev_close",
                "open",
                "high",
                "low",
                "last_price",
                "close",
                "average_price",
                "volume",
                "turnover",
                "num_trades",
            ]

            # Identify columns that actually exist and require string parsing
            cols_to_convert = [
                col for col in numeric_columns if col in df.columns and df[col].dtype == "object"
            ]

            if cols_to_convert:
                # Perform a single, vectorized regex replacement pass across all targeted columns
                # This prevents memory thrashing and intermediate object allocations
                df[cols_to_convert] = df[cols_to_convert].replace({",": "", "₹": ""}, regex=True)

                # Efficiently cast to numeric types, handling any corrupt data via coercion
                for col in cols_to_convert:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Set date as index
            df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y")
            df = df.set_index("date")

            logger.info(f"Date range: {df.index.min()} to {df.index.max()}")

            # Validate data with institutional-grade validation framework
            df, validation_metadata = validate_at_ingestion(
                df=df.reset_index(),  # Reset index for validation
                dataset_name=f"equity_history_{self.config.symbol}",
                source="NSE",
            )

            # Check if validation passed
            if not validation_metadata["validation_passed"]:
                error_msg = f"Validation failed for {self.config.symbol}: {validation_metadata.get('handling_metadata', {})}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Log validation score
            validation_score = validation_metadata["validation_score"]
            logger.info(f"Validation score: {validation_score:.2f}/100")

            # Set date as index again after validation
            df = df.set_index("date")

            return df

        except Exception as e:
            logger.error(f"Failed to download data for {self.config.symbol}: {str(e)}")
            raise

    def download_equity_data(self) -> pd.DataFrame:
        """Legacy nselib-backed download API retained for older callers/tests."""
        from pipelines import equity_history as legacy_equity_history

        return legacy_equity_history.capital_market.equity_history(
            symbol=self.config.symbol,
            from_date=self.config.from_date,
            to_date=self.config.to_date,
        )

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Validate DataFrame with institutional-grade rules.

        Args:
            df: DataFrame to validate

        Raises:
            ValueError: If validation fails with critical issues
        """
        logger.info("Running institutional-grade validation...")

        validator = EquityValidator(dataset_name=f"{self.config.symbol}_equity_history")
        report = validator.validate(df)

        logger.info(f"Validation score: {report.calculate_score()}/100")
        logger.info(f"Validation status: {'PASS' if report.is_acceptable() else 'FAIL'}")

        # Log validation summary
        for result in report.results:
            if not result.passed:
                logger.warning(f"Validation failed: {result.rule_name} - {result.message}")

        # Raise error if critical failures found
        if not report.is_acceptable():
            critical_failures = report.get_critical_failures()
            error_msg = f"Critical validation failures for {self.config.symbol}: "
            error_msg += ", ".join([f.rule_name for f in critical_failures])
            raise ValueError(error_msg)

        logger.info("Validation passed successfully")

    def save_parquet(self, df: pd.DataFrame) -> Path:
        """
        Save DataFrame to partitioned Parquet format.

        Data is partitioned by year and month for efficient querying:
        data/bronze/equity_history/year=2025/month=01/*.parquet

        Args:
            df: DataFrame to save

        Returns:
            Path to saved parquet directory
        """
        # Reset index to make date a column for partitioning
        df_reset = df.reset_index()

        # Add partition columns
        df_reset["year"] = df_reset["date"].dt.year
        df_reset["month"] = df_reset["date"].dt.month

        # Ensure symbol column exists
        if "symbol" not in df_reset.columns:
            df_reset["symbol"] = self.config.symbol

        logger.info(f"Saving to partitioned Parquet: {self.config.parquet_dir}")
        logger.info("Partition columns: year, month")
        logger.info(f"Date range: {df_reset['date'].min()} to {df_reset['date'].max()}")
        logger.info(f"Records: {len(df_reset)}")

        try:
            # Use versioned dataset manager
            versioned = VersionedDataset(self.config.parquet_dir)

            # Create metadata
            metadata = {
                "symbol": self.config.symbol,
                "from_date": self.config.from_date,
                "to_date": self.config.to_date,
                "source": "NSE",
                "partitioned": True,
            }

            # Save snapshot
            snapshot_path = versioned.save_snapshot(
                df=df_reset, dataset_name=f"{self.config.symbol}_equity_history", metadata=metadata
            )

            logger.info(f"Successfully saved versioned snapshot: {snapshot_path}")

            # Write lineage record
            checksum = compute_checksum(df)
            write_lineage_record(
                dataset=f"{self.config.symbol}_equity_history",
                source="NSE",
                row_count=len(df),
                checksum=checksum,
            )

            # Update metadata catalog
            catalog = MetadataCatalog()
            catalog.update_row_count("equity_history", len(df))
            quality_score = catalog.calculate_quality_score("equity_history")
            catalog.update_quality_score("equity_history", quality_score)

            # Save to ClickHouse (Production)
            self.save_to_clickhouse(df_reset)

            return snapshot_path
        except Exception as e:
            logger.error(f"Failed to save Parquet: {str(e)}")
            raise

    def save_to_clickhouse(self, df: pd.DataFrame) -> None:
        """
        Save DataFrame to ClickHouse (Production).

        Args:
            df: DataFrame to save
        """
        logger.info(f"Saving to ClickHouse: {CLICKHOUSE_DATABASE}")

        try:
            # Create database if not exists
            self.ch_client.create_database(CLICKHOUSE_DATABASE)

            # Transform data to match ClickHouse schema
            df_transformed = df.copy()

            # Ensure column names match ClickHouse schema
            column_mapping = {
                "date": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }

            # Select and rename columns
            available_columns = [col for col in column_mapping if col in df_transformed.columns]
            df_transformed = df_transformed[available_columns]
            df_transformed.columns = [column_mapping[col] for col in available_columns]

            # Add symbol column if not present
            if "symbol" not in df_transformed.columns:
                df_transformed["symbol"] = self.config.symbol

            # Convert date column
            df_transformed["date"] = pd.to_datetime(df_transformed["date"]).dt.date

            from datetime import timezone

            # Add created_at
            df_transformed["created_at"] = datetime.now(timezone.utc)

            # Insert into ClickHouse
            self.ch_client.insert_dataframe("equity_history", df_transformed)

            logger.info(f"Successfully saved {len(df_transformed)} rows to ClickHouse")

        except Exception as e:
            logger.error(f"Failed to save to ClickHouse: {str(e)}")
            # Don't raise - ClickHouse is production, DuckDB is research

    def save_duckdb(self, df: pd.DataFrame) -> None:
        """
        Save DataFrame to DuckDB.

        Args:
            df: DataFrame to save
        """
        logger.info(f"Saving to DuckDB: {self.config.duckdb_path}")

        try:
            with DB_LOCK:
                con = duckdb.connect(str(self.config.duckdb_path))

                # Initialize schema from schema.sql file
                schema_path = (
                    Path(__file__).parent.parent.parent / "storage" / "database" / "schema.sql"
                )
                if schema_path.exists():
                    with open(schema_path) as f:
                        schema_sql = f.read()
                    con.execute(schema_sql)
                    logger.info("Database schema initialized from schema.sql")

                # Insert data
                table_name = "equity_history"
                df_reset = df.reset_index()

                # Ensure symbol column exists
                if "symbol" not in df_reset.columns:
                    df_reset["symbol"] = self.config.symbol

                # Convert turnover column (remove special characters and convert to float)
                if "turnover" in df_reset.columns:
                    df_reset["turnover"] = (
                        df_reset["turnover"]
                        .astype(str)
                        .str.replace(",", "")
                        .str.replace("₹", "")
                        .astype(float)
                    )

                con.register("temp_df", df_reset)
                con.execute(f"INSERT INTO {table_name} SELECT * FROM temp_df")
                con.unregister("temp_df")

                # Verify insertion
                count = con.execute(
                    f"SELECT COUNT(*) FROM {table_name} WHERE symbol = '{self.config.symbol}'"
                ).fetchone()[0]
                logger.info(
                    f"Inserted {count} records into DuckDB table: {table_name} for symbol: {self.config.symbol}"
                )

                con.close()

            logger.info(f"Successfully saved to DuckDB: {self.config.duckdb_path}")

        except Exception as e:
            logger.error(f"Failed to save to DuckDB: {str(e)}")
            raise

    def run(self) -> None:
        """
        Execute the complete pipeline: Download → Save Parquet → Save DuckDB.
        """
        logger.info(f"Starting equity history pipeline for {self.config.symbol}")

        try:
            if pd.to_datetime(self.config.from_date) > pd.to_datetime(self.config.to_date):
                raise ValueError(
                    f"Invalid date range for {self.config.symbol}: "
                    f"{self.config.from_date} > {self.config.to_date}"
                )
            # Step 1: Download OHLCV
            df = self.download_ohlcv()

            # Step 2: Save to Parquet
            parquet_path = self.save_parquet(df)

            # Step 3: Save to DuckDB
            self.save_duckdb(df)

            logger.info(f"Pipeline completed successfully for {self.config.symbol}")
            logger.info(f"Data saved to: {parquet_path} and {self.config.duckdb_path}")

        except Exception as e:
            message = f"Pipeline failed for {self.config.symbol}: {str(e)}"
            logger.error(message)
            raise type(e)(message) from e


def main():
    """Main function to run the pipeline with example configuration."""
    config = EquityHistoryConfig(
        symbol="RELIANCE",
        from_date="2024-01-01",
        to_date="2024-12-31",
    )

    pipeline = EquityHistoryPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
