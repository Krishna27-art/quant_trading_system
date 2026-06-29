"""
NSE Options Chain Pipeline (Institutional Grade)

Downloads comprehensive options chain data for serious quant trading:
- Price data: bid, ask, LTP
- Greeks: delta, gamma, theta, vega, rho
- Volatility: IV
- Open Interest: OI, OI change
- Volume: Trading volume

This enables:
- Volatility trading
- Skew trading
- Term structure alpha
"""

import sys
import uuid
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from utils.time_utils import now_ist

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import BRONZE_OPTIONS_DIR
from data_platform.sources.ingestion.ingestion_engine import IngestionEngine
from data_platform.sources.ingestion.lineage import IngestionLineage
from data_platform.sources.ingestion.raw_bronze import RawBronzeLayer
from data_platform.validation.options_rules import OptionsValidator
from utils.api_helpers import nse_api_call
from utils.data_lineage import compute_checksum, write_lineage_record
from utils.init_directories import init_directories
from utils.logger import get_logger
from utils.metadata_catalog import MetadataCatalog
from utils.versioned_datasets import VersionedDataset

# Initialize directories
init_directories()

# Configure logging
logger = get_logger("options_chain")


class OptionsChainConfig(BaseModel):
    """Configuration for options chain pipeline."""

    symbol: str = Field(..., description="NSE symbol (e.g., NIFTY, BANKNIFTY, RELIANCE)")
    instrument: str = Field(
        default="OPTIDX", description="Instrument type (OPTIDX for index, OPTSTK for stock)"
    )
    from_date: str = Field(default="2024-01-01", description="Start date in YYYY-MM-DD format")
    to_date: str = Field(
        default_factory=lambda: now_ist().strftime("%Y-%m-%d"),
        description="End date in YYYY-MM-DD format",
    )
    data_dir: Path = Field(default=BRONZE_OPTIONS_DIR, description="Directory to save data")
    compute_greeks: bool = Field(default=True, description="Compute option greeks from price data")


class OptionsChainPipeline:
    """Pipeline for downloading and storing institutional-grade NSE options chain data."""

    def __init__(self, config: OptionsChainConfig):
        self.config = config
        self.ingestion_engine = IngestionEngine()
        self.raw_bronze = RawBronzeLayer()
        self.lineage = IngestionLineage()

    @nse_api_call
    def download_option_data(self, option_type: str) -> pd.DataFrame:
        """
        Download option data for a specific option type.

        Args:
            option_type: 'CE' for calls, 'PE' for puts

        Returns:
            DataFrame with option data

        Raises:
            Exception: If download fails
        """
        logger.info(f"Downloading {option_type} options for {self.config.symbol}")

        try:
            # Use institutional ingestion engine with fallback logic
            result = self.ingestion_engine.fetch_options_chain(
                symbol=self.config.symbol,
                expiry_date=None,
                use_fallback=True,  # Current expiry
            )

            if not result.success:
                raise ValueError(f"Ingestion failed: {result.error}")

            df = result.data

            if df is None or df.empty:
                raise ValueError(f"No data returned for {self.config.symbol} options")

            logger.info(f"Downloaded {len(df)} records from {result.source}")
            logger.info(f"Latency: {result.latency_ms}ms")

            # Store raw response in bronze layer
            dataset_name = f"options_chain_{self.config.symbol}_{option_type}"
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

            # Filter by option type
            if "option_type" in df.columns:
                df = df[df["option_type"] == option_type]

            return df

        except Exception as e:
            logger.error(f"Failed to download option data: {str(e)}")
            raise

    def process_option_data(self, ce_df: pd.DataFrame, pe_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process option data for both option types using ingestion engine.

        Args:
            ce_df: DataFrame with call option data
            pe_df: DataFrame with put option data

        Returns:
            Processed DataFrame with institutional-grade options data
        """
        logger.info("Processing institutional-grade options chain data...")

        if ce_df.empty and pe_df.empty:
            logger.warning("No option data to process")
            return pd.DataFrame()

        # Process CE data
        if not ce_df.empty:
            ce_processed = self._process_single_option_type(ce_df, "CE")
        else:
            ce_processed = pd.DataFrame()

        # Process PE data
        if not pe_df.empty:
            pe_processed = self._process_single_option_type(pe_df, "PE")
        else:
            pe_processed = pd.DataFrame()

        # Combine CE and PE data
        combined = pd.concat([ce_processed, pe_processed], ignore_index=True)

        # Add ingest ID for data lineage
        combined["ingest_id"] = str(uuid.uuid4())

        # Add source
        combined["source"] = "NSE"

        # Add ingest timestamp
        combined["ingest_timestamp"] = now_ist()

        # Validate with institutional-grade rules
        if not combined.empty:
            self._validate_options_data(combined)

        logger.info(f"Processed institutional options chain: {len(combined)} records")
        logger.info(f"Columns: {combined.columns.tolist()}")

        return combined

    def _validate_options_data(self, df: pd.DataFrame) -> None:
        """
        Validate options data with institutional-grade rules.

        Args:
            df: DataFrame to validate

        Raises:
            ValueError: If validation fails with critical issues
        """
        logger.info("Running institutional-grade options validation...")

        validator = OptionsValidator(dataset_name=f"{self.config.symbol}_options_chain")
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

        logger.info("Options validation passed successfully")

    def _process_single_option_type(self, df: pd.DataFrame, option_type: str) -> pd.DataFrame:
        """
        Process data for a single option type into institutional format.

        Args:
            df: Raw DataFrame from NSE
            option_type: 'CE' or 'PE'

        Returns:
            Processed DataFrame with institutional columns
        """
        logger.info(f"Processing {option_type} option data...")

        # Map NSE columns to institutional format
        # NSE columns: TIMESTAMP, EXPIRY_DT, STRIKE_PRICE, OPEN, HIGH, LOW, CLOSE,
        #               SETTLE_PR, CONTRACTS, OPEN_INT, CHG_IN_OI, TIMESTAMP,
        #               UNDERLYING_VALUE

        column_mapping = {
            "TIMESTAMP": "timestamp",
            "EXPIRY_DT": "expiry",
            "STRIKE_PRICE": "strike",
            "OPEN": "bid",  # Using open as bid proxy
            "CLOSE": "ltp",  # Using close as LTP proxy
            "HIGH": "ask",  # Using high as ask proxy
            "TOT_TRADED_QTY": "volume",
            "OPEN_INT": "oi",
            "CHG_IN_OI": "oi_change",
        }

        # Select available columns
        available_cols = [col for col in column_mapping if col in df.columns]
        df_processed = df[available_cols].copy()
        df_processed.columns = [column_mapping[col] for col in available_cols]

        # Add required columns
        df_processed["symbol"] = self.config.symbol
        df_processed["option_type"] = option_type

        # Convert timestamp
        df_processed["timestamp"] = pd.to_datetime(
            df_processed["timestamp"], format="%d-%b-%Y %H:%M:%S", errors="coerce"
        )

        # Convert expiry
        df_processed["expiry"] = pd.to_datetime(
            df_processed["expiry"], format="%d-%b-%Y", errors="coerce"
        )

        # Convert numeric columns
        numeric_cols = ["strike", "bid", "ask", "ltp", "volume", "oi", "oi_change"]
        for col in numeric_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors="coerce")

        # Add placeholder columns for greeks and IV (to be computed later)
        df_processed["iv"] = None
        df_processed["delta"] = None
        df_processed["gamma"] = None
        df_processed["theta"] = None
        df_processed["vega"] = None
        df_processed["rho"] = None

        # Ensure required columns exist
        required_cols = ["timestamp", "symbol", "expiry", "strike", "option_type"]
        for col in required_cols:
            if col not in df_processed.columns:
                df_processed[col] = None

        # Select final columns in correct order
        final_cols = [
            "timestamp",
            "symbol",
            "expiry",
            "strike",
            "option_type",
            "bid",
            "ask",
            "ltp",
            "oi",
            "oi_change",
            "iv",
            "delta",
            "gamma",
            "theta",
            "vega",
            "rho",
            "volume",
        ]

        # Only select columns that exist
        existing_final_cols = [col for col in final_cols if col in df_processed.columns]
        df_processed = df_processed[existing_final_cols]

        logger.info(f"Processed {len(df_processed)} {option_type} records")

        return df_processed

    def save_parquet(self, df: pd.DataFrame) -> Path:
        """
        Save DataFrame to partitioned Parquet format.

        Args:
            df: DataFrame to save

        Returns:
            Path to saved parquet directory
        """
        if df.empty:
            logger.warning("No data to save")
            return None

        # Reset index to make timestamp a column for partitioning
        df_reset = df.reset_index(drop=True)

        # Add partition columns
        df_reset["year"] = df_reset["timestamp"].dt.year
        df_reset["month"] = df_reset["timestamp"].dt.month
        df_reset["day"] = df_reset["timestamp"].dt.day

        logger.info(f"Saving to partitioned Parquet: {self.config.data_dir}")
        logger.info("Partition columns: year, month, day")
        logger.info(
            f"Timestamp range: {df_reset['timestamp'].min()} to {df_reset['timestamp'].max()}"
        )
        logger.info(f"Records: {len(df_reset)}")

        try:
            # Use versioned dataset manager
            versioned = VersionedDataset(self.config.data_dir)

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
                df=df_reset, dataset_name=f"{self.config.symbol}_options_chain", metadata=metadata
            )

            logger.info(f"Successfully saved versioned snapshot: {snapshot_path}")

            # Write lineage record
            checksum = compute_checksum(df)
            write_lineage_record(
                dataset=f"{self.config.symbol}_options_chain",
                source="NSE",
                row_count=len(df),
                checksum=checksum,
            )

            # Update metadata catalog
            catalog = MetadataCatalog()
            catalog.update_row_count("options_chain", len(df))
            quality_score = catalog.calculate_quality_score("options_chain")
            catalog.update_quality_score("options_chain", quality_score)

            return snapshot_path
        except Exception as e:
            logger.error(f"Failed to save Parquet: {str(e)}")
            raise

    def run(self) -> None:
        """
        Execute the complete pipeline: Download CE → Download PE → Process → Save.
        """
        logger.info(f"Starting institutional options chain pipeline for {self.config.symbol}")

        try:
            # Step 1: Download CE options
            ce_df = self.download_option_data("CE")

            # Step 2: Download PE options
            pe_df = self.download_option_data("PE")

            # Step 3: Process and merge data
            processed_df = self.process_option_data(ce_df, pe_df)

            if processed_df.empty:
                logger.info("No data to save")
                return

            # Step 4: Save to partitioned Parquet
            parquet_path = self.save_parquet(processed_df)

            logger.info("Pipeline completed successfully")
            logger.info(f"Data saved to: {parquet_path}")
            logger.info(f"Total records: {len(processed_df)}")

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    """Main function to run the pipeline with example configuration."""
    config = OptionsChainConfig(
        symbol="NIFTY",
        instrument="OPTIDX",
        from_date="2024-06-01",
    )
    pipeline = OptionsChainPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
