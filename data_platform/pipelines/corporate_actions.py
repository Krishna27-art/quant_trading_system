"""
NSE Corporate Actions Pipeline

Downloads corporate actions data (splits, bonuses, dividends) from NSE.
Critical for backtest adjustment.
"""

import re
from pathlib import Path

import duckdb
import pandas as pd
from pydantic import BaseModel, Field

from config.settings import DB_PATH, RAW_CORPORATE_ACTIONS_DIR
from data_platform.sources.ingestion.ingestion_engine import IngestionEngine
from data_platform.sources.ingestion.lineage import IngestionLineage
from data_platform.sources.ingestion.raw_bronze import RawBronzeLayer
from data_platform.validation.corporate_rules import CorporateValidator
from utils.api_helpers import nse_api_call
from utils.data_lineage import compute_checksum, write_lineage_record
from utils.init_directories import init_directories
from utils.logger import get_logger
from utils.metadata_catalog import MetadataCatalog
from utils.time_utils import now_ist
from utils.versioned_datasets import VersionedDataset

# Initialize directories
init_directories()

# Configure logging
logger = get_logger("corporate_actions")


class CorporateActionsConfig(BaseModel):
    """Configuration for corporate actions pipeline."""

    from_date: str = Field(default="2024-01-01", description="Start date in YYYY-MM-DD format")
    to_date: str = Field(
        default_factory=lambda: now_ist().strftime("%Y-%m-%d"),
        description="End date in YYYY-MM-DD format",
    )
    data_dir: Path = Field(default=RAW_CORPORATE_ACTIONS_DIR, description="Directory to save data")
    parquet_file: str = Field(
        default="corporate_actions.parquet", description="Output parquet filename"
    )
    duckdb_path: Path = Field(default=DB_PATH, description="Path to DuckDB database")


class CorporateActionsPipeline:
    """Pipeline for downloading and storing NSE corporate actions data."""

    def __init__(self, config: CorporateActionsConfig):
        self.config = config
        self.ingestion_engine = IngestionEngine()
        self.raw_bronze = RawBronzeLayer()
        self.lineage = IngestionLineage()

    @nse_api_call
    def download_corporate_actions(self) -> pd.DataFrame:
        """
        Download corporate actions from NSE.

        Returns:
            DataFrame with corporate actions data

        Raises:
            Exception: If download fails or data is invalid
        """
        logger.info(
            f"Downloading corporate actions from {self.config.from_date} to {self.config.to_date}"
        )

        try:
            # Use institutional ingestion engine with fallback logic
            result = self.ingestion_engine.fetch_corporate_actions(
                from_date=self.config.from_date, to_date=self.config.to_date, use_fallback=True
            )

            if not result.success:
                raise ValueError(f"Ingestion failed: {result.error}")

            df = result.data

            if df is None or df.empty:
                logger.warning("No corporate actions data returned")
                return pd.DataFrame()

            logger.info(f"Downloaded {len(df)} records from {result.source}")
            logger.info(f"Latency: {result.latency_ms}ms")

            # Store raw response in bronze layer
            dataset_name = "corporate_actions"
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

            logger.info(f"Downloaded {len(df)} corporate actions records")
            logger.info(f"Columns: {df.columns.tolist()}")

            # Process and standardize data
            df = self._process_data(df)

            return df

        except Exception as e:
            logger.error(f"Failed to download corporate actions: {str(e)}")
            raise

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and standardize corporate actions data.

        Args:
            df: Raw DataFrame from NSE

        Returns:
            Processed DataFrame with standard columns
        """
        logger.info("Processing corporate actions data...")

        # Select and rename columns
        df_processed = df[["symbol", "subject", "recDate", "exDate"]].copy()
        df_processed.columns = ["symbol", "subject", "record_date", "ex_date"]

        # Parse action type and ratio from subject
        df_processed["action_type"] = df_processed["subject"].apply(self._parse_action_type)
        df_processed["ratio"] = df_processed["subject"].apply(self._parse_ratio)

        # Convert date columns
        df_processed["record_date"] = pd.to_datetime(
            df_processed["record_date"], format="%d-%b-%Y", errors="coerce"
        )
        df_processed["ex_date"] = pd.to_datetime(
            df_processed["ex_date"], format="%d-%b-%Y", errors="coerce"
        )

        # Filter to EQ series only
        df_processed = df_processed[df_processed["symbol"].notna()]

        # Validate with institutional-grade rules
        if not df_processed.empty:
            self._validate_corporate_data(df_processed)

        # Select final columns
        df_final = df_processed[["symbol", "action_type", "record_date", "ex_date", "ratio"]].copy()

        logger.info(f"Processed {len(df_final)} corporate actions records")
        logger.info(
            "Action type distribution", extra={"data": str(df_final["action_type"].value_counts())}
        )

        return df_final

    def _validate_corporate_data(self, df: pd.DataFrame) -> None:
        """
        Validate corporate actions data with institutional-grade rules.

        Args:
            df: DataFrame to validate

        Raises:
            ValueError: If validation fails with critical issues
        """
        logger.info("Running institutional-grade corporate actions validation...")

        validator = CorporateValidator(dataset_name="corporate_actions")
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
            error_msg = "Critical validation failures for corporate actions: "
            error_msg += ", ".join([f.rule_name for f in critical_failures])
            raise ValueError(error_msg)

        logger.info("Corporate actions validation passed successfully")

    def _parse_action_type(self, subject: str) -> str:
        """
        Parse action type from subject string.

        Args:
            subject: Subject string from NSE

        Returns:
            Action type (DIVIDEND, BONUS, SPLIT, OTHER)
        """
        if pd.isna(subject):
            return "OTHER"

        subject_upper = subject.upper()

        if "DIVIDEND" in subject_upper or "DIV" in subject_upper:
            return "DIVIDEND"
        elif "BONUS" in subject_upper:
            return "BONUS"
        elif "SPLIT" in subject_upper:
            return "SPLIT"
        elif "RIGHTS" in subject_upper:
            return "RIGHTS"
        else:
            return "OTHER"

    def _parse_ratio(self, subject: str) -> float | None:
        """
        Parse ratio from subject string.

        Args:
            subject: Subject string from NSE

        Returns:
            Ratio as float, or None if not applicable
        """
        if pd.isna(subject):
            return None

        # Try to extract ratio patterns like "2:1", "3:2", etc.
        ratio_pattern = r"(\d+)\s*:\s*(\d+)"
        match = re.search(ratio_pattern, subject)
        if match:
            return float(match.group(1)) / float(match.group(2))

        # Try to extract dividend amount like "Rs 2 Per Share"
        dividend_pattern = r"Rs\s*(\d+(?:\.\d+)?)\s*Per\s*Share"
        match = re.search(dividend_pattern, subject, re.IGNORECASE)
        if match:
            return float(match.group(1))

        return None

    def save_parquet(self, df: pd.DataFrame) -> Path:
        """
        Save DataFrame to Parquet format.

        Args:
            df: DataFrame to save

        Returns:
            Path to saved parquet file
        """
        parquet_path = self.config.data_dir / self.config.parquet_file
        logger.info(f"Saving to Parquet: {parquet_path}")

        try:
            # Use versioned dataset manager
            versioned = VersionedDataset(self.config.data_dir)

            # Create metadata
            metadata = {
                "from_date": self.config.from_date,
                "to_date": self.config.to_date,
                "source": "NSE",
            }

            # Save snapshot
            snapshot_path = versioned.save_snapshot(
                df=df, dataset_name="corporate_actions", metadata=metadata
            )

            logger.info(f"Successfully saved versioned snapshot: {snapshot_path}")

            # Write lineage record
            checksum = compute_checksum(df)
            write_lineage_record(
                dataset="corporate_actions", source="NSE", row_count=len(df), checksum=checksum
            )

            # Update metadata catalog
            catalog = MetadataCatalog()
            catalog.update_row_count("corporate_actions", len(df))
            quality_score = catalog.calculate_quality_score("corporate_actions")
            catalog.update_quality_score("corporate_actions", quality_score)

            return snapshot_path
        except Exception as e:
            logger.error(f"Failed to save Parquet: {str(e)}")
            raise

    def save_duckdb(self, df: pd.DataFrame) -> None:
        """
        Save DataFrame to DuckDB.

        Args:
            df: DataFrame to save
        """
        logger.info(f"Saving to DuckDB: {self.config.duckdb_path}")

        try:
            con = duckdb.connect(str(self.config.duckdb_path))

            # Initialize schema from schema.sql file
            schema_path = Path(__file__).parent.parent / "database" / "schema.sql"
            if schema_path.exists():
                with open(schema_path) as f:
                    schema_sql = f.read()
                con.execute(schema_sql)
                logger.info("Database schema initialized from schema.sql")

            # Insert data
            table_name = "corporate_actions"

            # Insert data (replace existing data for the date range)
            if not df.empty:
                con.register("temp_df", df)

                # Delete existing data for the date range
                min_date = (
                    df["record_date"].min().strftime("%Y-%m-%d")
                    if not df["record_date"].isna().all()
                    else None
                )
                max_date = (
                    df["record_date"].max().strftime("%Y-%m-%d")
                    if not df["record_date"].isna().all()
                    else None
                )

                if min_date and max_date:
                    con.execute(
                        f"DELETE FROM {table_name} WHERE record_date BETWEEN '{min_date}' AND '{max_date}'"
                    )

                # Insert new data
                con.execute(f"INSERT INTO {table_name} SELECT * FROM temp_df")
                con.unregister("temp_df")

                # Verify insertion
                count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                logger.info(f"Total records in DuckDB table: {table_name} = {count}")

            con.close()
            logger.info(f"Successfully saved to DuckDB: {self.config.duckdb_path}")

        except Exception as e:
            logger.error(f"Failed to save to DuckDB: {str(e)}")
            raise

    def run(self) -> None:
        """
        Execute the complete pipeline: Download → Process → Save Parquet → Save DuckDB.
        """
        logger.info("Starting corporate actions pipeline")

        try:
            # Step 1: Download corporate actions
            df = self.download_corporate_actions()

            if df.empty:
                logger.info("No data to save")
                return

            # Step 2: Save to Parquet
            parquet_path = self.save_parquet(df)

            # Step 3: Save to DuckDB
            self.save_duckdb(df)

            logger.info("Pipeline completed successfully")
            logger.info(f"Data saved to: {parquet_path} and {self.config.duckdb_path}")

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    """Main function to run the pipeline with example configuration."""
    config = CorporateActionsConfig(
        from_date="2024-01-01",
    )
    pipeline = CorporateActionsPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
