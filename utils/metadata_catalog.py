"""
Metadata Catalog Utility

Institutional-grade metadata management for data governance and lineage.
Provides programmatic access to dataset metadata and quality tracking.
"""

from pathlib import Path

import duckdb
import pandas as pd
import yaml

from config.settings import DB_PATH
from utils.logger import get_logger
from utils.time_utils import now_ist

logger = get_logger("metadata_catalog")


class MetadataCatalog:
    """
    Metadata catalog for institutional data governance.

    Provides centralized access to dataset metadata, quality scores, and lineage information.
    """

    def __init__(self, catalog_path: Path | None = None):
        """
        Initialize the metadata catalog.

        Args:
            catalog_path: Path to datasets.yaml (defaults to data/catalog/datasets.yaml)
        """
        if catalog_path is None:
            catalog_path = Path(__file__).parent.parent / "data" / "catalog" / "datasets.yaml"

        self.catalog_path = catalog_path
        self.catalog = self._load_catalog()
        self.logger = logger

    def _load_catalog(self) -> dict:
        """
        Load the catalog from YAML file.

        Returns:
            Dictionary with catalog data
        """
        try:
            with open(self.catalog_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load catalog: {str(e)}")
            raise

    def get_dataset_metadata(self, dataset_name: str) -> dict | None:
        """
        Get metadata for a specific dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary with dataset metadata or None if not found
        """
        datasets = self.catalog.get("datasets", {})
        return datasets.get(dataset_name)

    def get_all_datasets(self) -> dict[str, dict]:
        """
        Get metadata for all datasets.

        Returns:
            Dictionary mapping dataset names to metadata
        """
        return self.catalog.get("datasets", {})

    def get_active_datasets(self) -> dict[str, dict]:
        """
        Get metadata for active datasets only.

        Returns:
            Dictionary mapping active dataset names to metadata
        """
        datasets = self.get_all_datasets()
        return {
            name: metadata
            for name, metadata in datasets.items()
            if metadata.get("status") == "active"
        }

    def get_datasets_by_owner(self, owner: str) -> dict[str, dict]:
        """
        Get datasets owned by a specific owner.

        Args:
            owner: Owner name

        Returns:
            Dictionary mapping dataset names to metadata
        """
        datasets = self.get_all_datasets()
        return {
            name: metadata for name, metadata in datasets.items() if metadata.get("owner") == owner
        }

    def get_datasets_by_source(self, source: str) -> dict[str, dict]:
        """
        Get datasets from a specific source.

        Args:
            source: Data source name

        Returns:
            Dictionary mapping dataset names to metadata
        """
        datasets = self.get_all_datasets()
        return {
            name: metadata
            for name, metadata in datasets.items()
            if metadata.get("source") == source
        }

    def get_datasets_by_layer(self, data_layer: str) -> dict[str, dict]:
        """
        Get datasets in a specific data layer.

        Args:
            data_layer: Data layer (bronze, silver, gold, master)

        Returns:
            Dictionary mapping dataset names to metadata
        """
        datasets = self.get_all_datasets()
        return {
            name: metadata
            for name, metadata in datasets.items()
            if metadata.get("data_layer") == data_layer
        }

    def update_row_count(self, dataset_name: str, row_count: int) -> None:
        """
        Update row count for a dataset.

        Args:
            dataset_name: Name of the dataset
            row_count: Current row count
        """
        datasets = self.catalog.get("datasets", {})
        if dataset_name in datasets:
            datasets[dataset_name]["row_count"] = row_count
            self._save_catalog()
            self.logger.info(f"Updated row count for {dataset_name}: {row_count}")
        else:
            self.logger.warning(f"Dataset {dataset_name} not found in catalog")

    def update_quality_score(self, dataset_name: str, quality_score: float) -> None:
        """
        Update quality score for a dataset.

        Args:
            dataset_name: Name of the dataset
            quality_score: Quality score (0-100)
        """
        datasets = self.catalog.get("datasets", {})
        if dataset_name in datasets:
            datasets[dataset_name]["quality_score"] = quality_score
            self._save_catalog()
            self.logger.info(f"Updated quality score for {dataset_name}: {quality_score}")
        else:
            self.logger.warning(f"Dataset {dataset_name} not found in catalog")

    def calculate_quality_score(self, dataset_name: str) -> float:
        """
        Calculate quality score for a dataset based on validation results.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Quality score (0-100)
        """
        try:
            conn = duckdb.connect(str(DB_PATH))

            # Get validation results from data_lineage table
            query = """
                SELECT
                    COUNT(*) as total_records,
                    COUNT(CASE WHEN checksum IS NOT NULL THEN 1 END) as checksum_records
                FROM data_lineage
                WHERE dataset = ?
            """

            result = conn.execute(query, [dataset_name]).fetchone()
            conn.close()

            if result and result[0] > 0:
                total_records = result[0]
                checksum_records = result[1]

                # Quality score based on checksum coverage
                quality_score = (checksum_records / total_records) * 100
                return round(quality_score, 2)

            return 0.0

        except Exception as e:
            self.logger.error(f"Failed to calculate quality score for {dataset_name}: {str(e)}")
            return 0.0

    def get_schema(self, dataset_name: str) -> list[dict] | None:
        """
        Get schema for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            List of column definitions or None if not found
        """
        metadata = self.get_dataset_metadata(dataset_name)
        if metadata:
            return metadata.get("schema")
        return None

    def validate_schema(self, dataset_name: str, df: pd.DataFrame) -> bool:
        """
        Validate DataFrame against catalog schema.

        Args:
            dataset_name: Name of the dataset
            df: DataFrame to validate

        Returns:
            True if schema matches, False otherwise
        """
        catalog_schema = self.get_schema(dataset_name)
        if not catalog_schema:
            self.logger.warning(f"No schema found for {dataset_name}")
            return False

        catalog_columns = {col.split(":")[0] for col in catalog_schema}
        df_columns = set(df.columns)

        missing_columns = catalog_columns - df_columns
        extra_columns = df_columns - catalog_columns

        if missing_columns:
            self.logger.warning(f"Missing columns in {dataset_name}: {missing_columns}")

        if extra_columns:
            self.logger.warning(f"Extra columns in {dataset_name}: {extra_columns}")

        return len(missing_columns) == 0

    def get_refresh_schedule(self) -> dict[str, str]:
        """
        Get refresh schedule for all active datasets.

        Returns:
            Dictionary mapping dataset names to refresh frequencies
        """
        active_datasets = self.get_active_datasets()
        return {
            name: metadata.get("refresh_frequency", "unknown")
            for name, metadata in active_datasets.items()
        }

    def get_data_lineage(self, dataset_name: str) -> pd.DataFrame:
        """
        Get data lineage for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            DataFrame with lineage records
        """
        try:
            conn = duckdb.connect(str(DB_PATH))

            query = """
                SELECT * FROM data_lineage
                WHERE dataset = ?
                ORDER BY downloaded_at DESC
            """

            df = conn.execute(query, [dataset_name]).df()
            conn.close()

            return df

        except Exception as e:
            self.logger.error(f"Failed to get lineage for {dataset_name}: {str(e)}")
            return pd.DataFrame()

    def get_catalog_summary(self) -> dict:
        """
        Get summary of the catalog.

        Returns:
            Dictionary with catalog summary statistics
        """
        datasets = self.get_all_datasets()
        active_datasets = self.get_active_datasets()

        summary = {
            "total_datasets": len(datasets),
            "active_datasets": len(active_datasets),
            "planned_datasets": len([d for d in datasets.values() if d.get("status") == "planned"]),
            "deprecated_datasets": len(
                [d for d in datasets.values() if d.get("status") == "deprecated"]
            ),
            "catalog_version": self.catalog.get("catalog_version"),
            "last_updated": self.catalog.get("last_updated"),
            "catalog_owner": self.catalog.get("catalog_owner"),
        }

        return summary

    def _save_catalog(self) -> None:
        """
        Save catalog to YAML file.

        Updates the last_updated timestamp.
        """
        self.catalog["last_updated"] = now_ist().strftime("%Y-%m-%d")

        with open(self.catalog_path, "w") as f:
            yaml.dump(self.catalog, f, default_flow_style=False)

        self.logger.info("Catalog saved successfully")

    def add_dataset(self, dataset_name: str, metadata: dict) -> None:
        """
        Add a new dataset to the catalog.

        Args:
            dataset_name: Name of the dataset
            metadata: Dictionary with dataset metadata
        """
        datasets = self.catalog.get("datasets", {})

        if dataset_name in datasets:
            self.logger.warning(f"Dataset {dataset_name} already exists in catalog")
            return

        # Validate required fields
        required_fields = ["dataset", "owner", "source", "refresh_frequency", "schema"]
        for field in required_fields:
            if field not in metadata:
                raise ValueError(f"Missing required field: {field}")

        datasets[dataset_name] = metadata
        self._save_catalog()

        self.logger.info(f"Added dataset {dataset_name} to catalog")

    def remove_dataset(self, dataset_name: str) -> None:
        """
        Remove a dataset from the catalog.

        Args:
            dataset_name: Name of the dataset
        """
        datasets = self.catalog.get("datasets", {})

        if dataset_name not in datasets:
            self.logger.warning(f"Dataset {dataset_name} not found in catalog")
            return

        del datasets[dataset_name]
        self._save_catalog()

        self.logger.info(f"Removed dataset {dataset_name} from catalog")
