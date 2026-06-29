"""
Data Lineage Utility

Tracks data lineage for reproducibility and audit trail.
Every data load writes one record to the data_lineage table.
"""

import hashlib
import subprocess
import uuid

import duckdb
import pandas as pd

from config.settings import DB_PATH
from utils.logger import get_logger
from utils.time_utils import now_ist

logger = get_logger("data_lineage")


def get_git_commit() -> str | None:
    """
    Get current git commit hash.

    Returns:
        Git commit hash or None if not in git repo
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        logger.warning(f"Failed to get git commit: {str(e)}")
    return None


def get_pipeline_version() -> str:
    """
    Get pipeline version (can be updated manually or from git tags).

    Returns:
        Pipeline version string
    """
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--always"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        logger.warning(f"Failed to get pipeline version: {str(e)}")
    return "1.0.0"  # Default version


def compute_checksum(df: pd.DataFrame) -> str:
    """
    Compute checksum of DataFrame for data integrity.

    Args:
        df: DataFrame to compute checksum for

    Returns:
        SHA256 hash string
    """
    # Convert DataFrame to string representation
    df_str = df.to_csv(index=False)
    return hashlib.sha256(df_str.encode()).hexdigest()


def write_lineage_record(
    dataset: str,
    source: str,
    row_count: int,
    pipeline_version: str | None = None,
    git_commit: str | None = None,
    checksum: str | None = None,
) -> str:
    """
    Write a lineage record to the data_lineage table.

    Args:
        dataset: Name of the dataset
        source: Data source (e.g., 'NSE', 'BSE')
        row_count: Number of rows in the dataset
        pipeline_version: Pipeline version (auto-detected if None)
        git_commit: Git commit hash (auto-detected if None)
        checksum: Data checksum (optional)

    Returns:
        Lineage record ID
    """
    try:
        # Auto-detect if not provided
        if pipeline_version is None:
            pipeline_version = get_pipeline_version()
        if git_commit is None:
            git_commit = get_git_commit()

        # Generate unique ID
        lineage_id = str(uuid.uuid4())

        # Current timestamp
        downloaded_at = now_ist()

        # Write to database
        conn = duckdb.connect(str(DB_PATH))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS data_lineage (
                id VARCHAR PRIMARY KEY,
                dataset VARCHAR,
                source VARCHAR,
                downloaded_at TIMESTAMP,
                pipeline_version VARCHAR,
                git_commit VARCHAR,
                checksum VARCHAR,
                row_count INTEGER
            )
        """)

        conn.execute(
            """
            INSERT INTO data_lineage (
                id, dataset, source, downloaded_at,
                pipeline_version, git_commit, checksum, row_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                lineage_id,
                dataset,
                source,
                downloaded_at,
                pipeline_version,
                git_commit,
                checksum,
                row_count,
            ],
        )

        conn.close()

        logger.info(
            f"Lineage record written: {dataset} from {source} "
            f"({row_count} rows, commit: {git_commit})"
        )

        return lineage_id

    except Exception as e:
        logger.error(f"Failed to write lineage record: {str(e)}")
        raise


def get_lineage_history(
    dataset: str | None = None, source: str | None = None, limit: int = 100
) -> pd.DataFrame:
    """
    Get lineage history from the data_lineage table.

    Args:
        dataset: Filter by dataset name
        source: Filter by source
        limit: Maximum number of records to return

    Returns:
        DataFrame with lineage records
    """
    try:
        conn = duckdb.connect(str(DB_PATH))

        conn.execute("""
            CREATE TABLE IF NOT EXISTS data_lineage (
                id VARCHAR PRIMARY KEY,
                dataset VARCHAR,
                source VARCHAR,
                downloaded_at TIMESTAMP,
                pipeline_version VARCHAR,
                git_commit VARCHAR,
                checksum VARCHAR,
                row_count INTEGER
            )
        """)

        query = "SELECT * FROM data_lineage"
        conditions = []
        params = []

        if dataset:
            conditions.append("dataset = ?")
            params.append(dataset)
        if source:
            conditions.append("source = ?")
            params.append(source)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY downloaded_at DESC LIMIT ?"
        params.append(limit)

        df = conn.execute(query, params).df()
        conn.close()

        logger.info(f"Retrieved {len(df)} lineage records")
        return df

    except Exception as e:
        logger.error(f"Failed to get lineage history: {str(e)}")
        raise


def get_latest_lineage(dataset: str) -> dict | None:
    """
    Get the most recent lineage record for a dataset.

    Args:
        dataset: Dataset name

    Returns:
        Dictionary with lineage record or None
    """
    try:
        df = get_lineage_history(dataset=dataset, limit=1)
        if not df.empty:
            return df.iloc[0].to_dict()
        return None
    except Exception as e:
        logger.error(f"Failed to get latest lineage: {str(e)}")
        return None
