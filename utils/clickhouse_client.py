"""
ClickHouse Client

Institutional-grade ClickHouse client for production workloads.
High-performance, real-time analytics database with time-series optimization.
"""

from contextlib import contextmanager
from datetime import datetime
from enum import Enum

import clickhouse_connect
import pandas as pd

from config.settings import (
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_SECURE,
    CLICKHOUSE_USER,
)
from utils.logger import get_logger

logger = get_logger("clickhouse_client")


class DataTier(Enum):
    """Data tier for hot/cold separation."""

    HOT = "hot"  # Recent data, fast access
    WARM = "warm"  # Medium-term data
    COLD = "cold"  # Historical data, compressed


class ClickHouseClient:
    """
    ClickHouse client for production workloads.

    Handles connection management, query execution, and data ingestion.
    """

    def __init__(self):
        """Initialize ClickHouse client."""
        self.client = None
        self.logger = logger

    def connect(self) -> None:
        """Establish connection to ClickHouse."""
        try:
            self.client = clickhouse_connect.get_client(
                host=CLICKHOUSE_HOST,
                port=CLICKHOUSE_PORT,
                username=CLICKHOUSE_USER,
                password=CLICKHOUSE_PASSWORD,
                database=CLICKHOUSE_DATABASE,
                secure=CLICKHOUSE_SECURE,
            )
            self.logger.info(
                f"Connected to ClickHouse: {CLICKHOUSE_HOST}:{CLICKHOUSE_PORT}/{CLICKHOUSE_DATABASE}"
            )
        except Exception as e:
            self.logger.error(f"Failed to connect to ClickHouse: {str(e)}")
            raise

    def disconnect(self) -> None:
        """Close ClickHouse connection."""
        if self.client:
            self.client.close()
            self.logger.info("Disconnected from ClickHouse")

    @contextmanager
    def connection(self):
        """Context manager for connection handling."""
        self.connect()
        try:
            yield self.client
        finally:
            self.disconnect()

    def execute_query(self, query: str, params: dict | None = None) -> None:
        """
        Execute a query without returning results.

        Args:
            query: SQL query to execute
            params: Query parameters
        """
        try:
            with self.connection() as client:
                client.command(query, parameters=params)
                self.logger.info("Executed query successfully")
        except Exception as e:
            self.logger.error(f"Failed to execute query: {str(e)}")
            raise

    def execute_query_df(self, query: str, params: dict | None = None) -> pd.DataFrame:
        """
        Execute a query and return results as DataFrame.

        Args:
            query: SQL query to execute
            params: Query parameters

        Returns:
            Query results as DataFrame
        """
        try:
            with self.connection() as client:
                result = client.query_df(query, parameters=params)
                self.logger.info(f"Query returned {len(result)} rows")
                return result
        except Exception as e:
            self.logger.error(f"Failed to execute query: {str(e)}")
            raise

    def insert_dataframe(
        self, table_name: str, df: pd.DataFrame, database: str | None = None
    ) -> None:
        """
        Insert DataFrame into ClickHouse table.

        Args:
            table_name: Target table name
            df: DataFrame to insert
            database: Database name (defaults to CLICKHOUSE_DATABASE)
        """
        try:
            db = database or CLICKHOUSE_DATABASE
            with self.connection() as client:
                client.insert_df(f"{db}.{table_name}", df)
                self.logger.info(f"Inserted {len(df)} rows into {db}.{table_name}")
        except Exception as e:
            self.logger.error(f"Failed to insert DataFrame: {str(e)}")
            raise

    def create_database(self, database_name: str) -> None:
        """
        Create a database if it doesn't exist.

        Args:
            database_name: Database name
        """
        query = f"CREATE DATABASE IF NOT EXISTS {database_name}"
        self.execute_query(query)
        self.logger.info(f"Created database: {database_name}")

    def create_table(self, table_name: str, schema: str, database: str | None = None) -> None:
        """
        Create a table with given schema.

        Args:
            table_name: Table name
            schema: Table schema SQL
            database: Database name (defaults to CLICKHOUSE_DATABASE)
        """
        db = database or CLICKHOUSE_DATABASE
        full_table_name = f"{db}.{table_name}"
        query = f"CREATE TABLE IF NOT EXISTS {full_table_name} ({schema})"
        self.execute_query(query)
        self.logger.info(f"Created table: {full_table_name}")

    def drop_table(self, table_name: str, database: str | None = None) -> None:
        """
        Drop a table if it exists.

        Args:
            table_name: Table name
            database: Database name (defaults to CLICKHOUSE_DATABASE)
        """
        db = database or CLICKHOUSE_DATABASE
        full_table_name = f"{db}.{table_name}"
        query = f"DROP TABLE IF EXISTS {full_table_name}"
        self.execute_query(query)
        self.logger.info(f"Dropped table: {full_table_name}")

    def table_exists(self, table_name: str, database: str | None = None) -> bool:
        """
        Check if a table exists.

        Args:
            table_name: Table name
            database: Database name (defaults to CLICKHOUSE_DATABASE)

        Returns:
            True if table exists, False otherwise
        """
        try:
            db = database or CLICKHOUSE_DATABASE
            query = f"EXISTS TABLE {db}.{table_name}"
            result = self.execute_query_df(query)
            return result.iloc[0, 0] == 1
        except Exception as e:
            self.logger.error(f"Failed to check table existence: {str(e)}")
            return False

    def get_table_schema(
        self, table_name: str, database: str | None = None
    ) -> list[dict[str, str]]:
        """
        Get table schema.

        Args:
            table_name: Table name
            database: Database name (defaults to CLICKHOUSE_DATABASE)

        Returns:
            List of column definitions
        """
        try:
            db = database or CLICKHOUSE_DATABASE
            query = f"DESCRIBE {db}.{table_name}"
            result = self.execute_query_df(query)

            schema = []
            for _, row in result.iterrows():
                schema.append(
                    {
                        "name": row["name"],
                        "type": row["type"],
                        "default_type": row.get("default_type", ""),
                        "default_expression": row.get("default_expression", ""),
                    }
                )

            return schema
        except Exception as e:
            self.logger.error(f"Failed to get table schema: {str(e)}")
            return []

    def get_row_count(self, table_name: str, database: str | None = None) -> int:
        """
        Get row count of a table.

        Args:
            table_name: Table name
            database: Database name (defaults to CLICKHOUSE_DATABASE)

        Returns:
            Row count
        """
        try:
            db = database or CLICKHOUSE_DATABASE
            query = f"SELECT count(*) as count FROM {db}.{table_name}"
            result = self.execute_query_df(query)
            return result.iloc[0, 0]
        except Exception as e:
            self.logger.error(f"Failed to get row count: {str(e)}")
            return 0

    def optimize_table(self, table_name: str, database: str | None = None) -> None:
        """
        Optimize table (merge parts).

        Args:
            table_name: Table name
            database: Database name (defaults to CLICKHOUSE_DATABASE)
        """
        try:
            db = database or CLICKHOUSE_DATABASE
            full_table_name = f"{db}.{table_name}"
            query = f"OPTIMIZE TABLE {full_table_name} FINAL"
            self.execute_query(query)
            self.logger.info(f"Optimized table: {full_table_name}")
        except Exception as e:
            self.logger.error(f"Failed to optimize table: {str(e)}")
            raise

    def create_time_series_table(
        self,
        table_name: str,
        schema: str,
        date_column: str = "timestamp",
        partition_by: str | None = None,
        order_by: str | None = None,
        database: str | None = None,
    ) -> None:
        """
        Create a time-series optimized table.

        Args:
            table_name: Table name
            schema: Table schema SQL
            date_column: Date column for partitioning
            partition_by: Partition expression (e.g., toYYYYMM(timestamp))
            order_by: Order by clause
            database: Database name
        """
        db = database or CLICKHOUSE_DATABASE
        full_table_name = f"{db}.{table_name}"

        # Default partitioning by month if not specified
        if partition_by is None:
            partition_by = f"toYYYYMM({date_column})"

        # Default order by date and symbol if not specified
        if order_by is None:
            order_by = f"{date_column}, symbol"

        # Create MergeTree table with time-series optimization
        query = f"""
        CREATE TABLE IF NOT EXISTS {full_table_name} (
            {schema}
        )
        ENGINE = MergeTree()
        PARTITION BY {partition_by}
        ORDER BY ({order_by})
        SETTINGS index_granularity = 8192
        """

        self.execute_query(query)
        self.logger.info(f"Created time-series table: {full_table_name}")

    def create_replicated_table(
        self,
        table_name: str,
        schema: str,
        replica_name: str,
        date_column: str = "timestamp",
        partition_by: str | None = None,
        order_by: str | None = None,
        database: str | None = None,
    ) -> None:
        """
        Create a replicated table for high availability.

        Args:
            table_name: Table name
            schema: Table schema SQL
            replica_name: Replica name
            date_column: Date column for partitioning
            partition_by: Partition expression
            order_by: Order by clause
            database: Database name
        """
        db = database or CLICKHOUSE_DATABASE
        full_table_name = f"{db}.{table_name}"

        if partition_by is None:
            partition_by = f"toYYYYMM({date_column})"

        if order_by is None:
            order_by = f"{date_column}, symbol"

        query = f"""
        CREATE TABLE IF NOT EXISTS {full_table_name} (
            {schema}
        )
        ENGINE = ReplicatedMergeTree('/clickhouse/tables/{db}/{table_name}', '{replica_name}')
        PARTITION BY {partition_by}
        ORDER BY ({order_by})
        SETTINGS index_granularity = 8192
        """

        self.execute_query(query)
        self.logger.info(f"Created replicated table: {full_table_name}")

    def insert_streaming(
        self,
        table_name: str,
        data: pd.DataFrame,
        batch_size: int = 10000,
        database: str | None = None,
    ) -> None:
        """
        Insert data in streaming batches for high-throughput ingestion.

        Args:
            table_name: Target table name
            data: DataFrame to insert
            batch_size: Batch size for streaming
            database: Database name
        """
        db = database or CLICKHOUSE_DATABASE

        for i in range(0, len(data), batch_size):
            batch = data.iloc[i : i + batch_size]
            self.insert_dataframe(table_name, batch, db)

        self.logger.info(f"Streamed {len(data)} rows into {db}.{table_name}")

    def query_time_range(
        self,
        table_name: str,
        start_date: datetime,
        end_date: datetime,
        columns: list[str] | None = None,
        symbol: str | None = None,
        database: str | None = None,
    ) -> pd.DataFrame:
        """
        Query data for a specific time range (optimized for time-series).

        Args:
            table_name: Table name
            start_date: Start date
            end_date: End date
            columns: Columns to select (None for all)
            symbol: Filter by symbol (optional)
            database: Database name

        Returns:
            Query results as DataFrame
        """
        db = database or CLICKHOUSE_DATABASE
        full_table_name = f"{db}.{table_name}"

        col_str = ", ".join(columns) if columns else "*"

        query = f"""
        SELECT {col_str}
        FROM {full_table_name}
        WHERE timestamp >= '{start_date.strftime("%Y-%m-%d %H:%M:%S")}'
        AND timestamp <= '{end_date.strftime("%Y-%m-%d %H:%M:%S")}'
        """

        if symbol:
            query += f" AND symbol = '{symbol}'"

        query += " ORDER BY timestamp"

        return self.execute_query_df(query)

    def query_latest(
        self,
        table_name: str,
        symbol: str | None = None,
        limit: int = 1000,
        database: str | None = None,
    ) -> pd.DataFrame:
        """
        Query latest data for live strategies (optimized read path).

        Args:
            table_name: Table name
            symbol: Filter by symbol (optional)
            limit: Number of rows to return
            database: Database name

        Returns:
            Query results as DataFrame
        """
        db = database or CLICKHOUSE_DATABASE
        full_table_name = f"{db}.{table_name}"

        query = f"""
        SELECT *
        FROM {full_table_name}
        """

        if symbol:
            query += f" WHERE symbol = '{symbol}'"

        query += f" ORDER BY timestamp DESC LIMIT {limit}"

        return self.execute_query_df(query)

    def create_materialized_view(
        self, view_name: str, source_table: str, query: str, database: str | None = None
    ) -> None:
        """
        Create a materialized view for real-time aggregations.

        Args:
            view_name: View name
            source_table: Source table name
            query: Aggregation query
            database: Database name
        """
        db = database or CLICKHOUSE_DATABASE
        full_view_name = f"{db}.{view_name}"

        create_query = f"""
        CREATE MATERIALIZED VIEW IF NOT EXISTS {full_view_name}
        ENGINE = AggregatingMergeTree()
        ORDER BY (timestamp, symbol)
        AS
        {query}
        """

        self.execute_query(create_query)
        self.logger.info(f"Created materialized view: {full_view_name}")

    def setup_data_tiering(
        self,
        hot_table: str,
        warm_table: str,
        cold_table: str,
        schema: str,
        date_column: str = "timestamp",
        database: str | None = None,
    ) -> None:
        """
        Set up hot/warm/cold data tiering.

        Args:
            hot_table: Hot table name (recent data, 7 days)
            warm_table: Warm table name (medium-term, 90 days)
            cold_table: Cold table name (historical, compressed)
            schema: Table schema
            date_column: Date column
            database: Database name
        """
        db = database or CLICKHOUSE_DATABASE

        # Create hot table (MergeTree, fast access)
        self.create_time_series_table(
            hot_table,
            schema,
            date_column,
            partition_by=f"toYYYYMMDD({date_column})",
            order_by=f"{date_column}, symbol",
            database=db,
        )

        # Create warm table (MergeTree, compressed)
        schema.replace("MergeTree", "MergeTree SETTINGS compress_level=3")
        self.create_time_series_table(
            warm_table,
            schema,
            date_column,
            partition_by=f"toYYYYMM({date_column})",
            order_by=f"{date_column}, symbol",
            database=db,
        )

        # Create cold table (MergeTree, highly compressed)
        schema.replace("MergeTree", "MergeTree SETTINGS compress_level=5")
        self.create_time_series_table(
            cold_table,
            schema,
            date_column,
            partition_by=f"toYYYY({date_column})",
            order_by=f"{date_column}, symbol",
            database=db,
        )

        self.logger.info(
            f"Set up data tiering: hot={hot_table}, warm={warm_table}, cold={cold_table}"
        )

    def move_data_to_tier(
        self,
        source_table: str,
        target_table: str,
        cutoff_date: datetime,
        database: str | None = None,
    ) -> int:
        """
        Move data from one tier to another based on date.

        Args:
            source_table: Source table
            target_table: Target table
            cutoff_date: Cutoff date for data movement
            database: Database name

        Returns:
            Number of rows moved
        """
        db = database or CLICKHOUSE_DATABASE
        full_source = f"{db}.{source_table}"
        full_target = f"{db}.{target_table}"

        # Insert into target
        insert_query = f"""
        INSERT INTO {full_target}
        SELECT * FROM {full_source}
        WHERE timestamp < '{cutoff_date.strftime("%Y-%m-%d %H:%M:%S")}'
        """

        self.execute_query(insert_query)

        # Delete from source
        delete_query = f"""
        ALTER TABLE {full_source}
        DELETE WHERE timestamp < '{cutoff_date.strftime("%Y-%m-%d %H:%M:%S")}'
        """

        self.execute_query(delete_query)

        # Optimize both tables
        self.optimize_table(source_table, db)
        self.optimize_table(target_table, db)

        self.logger.info(f"Moved data from {source_table} to {target_table} before {cutoff_date}")

        return self.get_row_count(target_table, db)


def get_clickhouse_client() -> ClickHouseClient:
    """
    Get a ClickHouse client instance.

    Returns:
        ClickHouseClient instance
    """
    return ClickHouseClient()
