"""
Central Configuration Settings

All paths and configuration constants are defined here.
This makes the codebase portable and maintainable.
"""

import os
import threading
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories - Data Lake Architecture (Bronze/Silver/Gold)
DATA_DIR = BASE_DIR / "data"

# Bronze Layer - Raw data from sources
BRONZE_DIR = DATA_DIR / "bronze"
BRONZE_EQUITY_HISTORY_DIR = BRONZE_DIR / "equity_history"
BRONZE_OPTIONS_DIR = BRONZE_DIR / "options"
BRONZE_CORPORATE_ACTIONS_DIR = BRONZE_DIR / "corporate_actions"
BRONZE_FLOWS_DIR = BRONZE_DIR / "flows"

# Silver Layer - Cleaned and validated data
SILVER_DIR = DATA_DIR / "silver"
SILVER_EQUITY_HISTORY_DIR = SILVER_DIR / "equity_history"
SILVER_OPTIONS_DIR = SILVER_DIR / "options"

# Gold Layer - Aggregated features and analytics
GOLD_DIR = DATA_DIR / "gold"
GOLD_FEATURES_DIR = GOLD_DIR / "features"

# Legacy directories (for backward compatibility during migration)
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
FEATURES_DIR = DATA_DIR / "features"
MASTER_DIR = DATA_DIR / "master"
CATALOG_DIR = DATA_DIR / "catalog"

# Legacy subdirectories within RAW_DIR
RAW_EQUITY_DIR = RAW_DIR / "equity"
RAW_OPTIONS_DIR = RAW_DIR / "options"
RAW_CORPORATE_ACTIONS_DIR = RAW_DIR / "corporate_actions"
RAW_FLOWS_DIR = RAW_DIR / "flows"

# Legacy subdirectories within FEATURES_DIR
FEATURES_MOMENTUM_DIR = FEATURES_DIR / "momentum"
FEATURES_VOLATILITY_DIR = FEATURES_DIR / "volatility"
FEATURES_LIQUIDITY_DIR = FEATURES_DIR / "liquidity"
FEATURES_OPTIONS_DIR = FEATURES_DIR / "options"
FEATURES_FLOW_DIR = FEATURES_DIR / "flow"
FEATURES_FUNDAMENTALS_DIR = FEATURES_DIR / "fundamentals"

# Database
DATABASE_DIR = BASE_DIR / "database"
DB_PATH = DATABASE_DIR / "market.duckdb"


def _bootstrap_duckdb_schema() -> None:
    """Create the minimal institutional schema expected by local services/tests."""
    try:
        import duckdb

        DATABASE_DIR.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(str(DB_PATH))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS security_master (
                symbol VARCHAR PRIMARY KEY,
                isin VARCHAR,
                company_name VARCHAR,
                sector VARCHAR,
                industry VARCHAR,
                listing_date DATE,
                delisting_date DATE,
                face_value DOUBLE,
                is_current BOOLEAN
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS equity_history (
                symbol VARCHAR,
                date DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS options_chain (
                timestamp TIMESTAMP,
                symbol VARCHAR,
                expiry DATE,
                strike DOUBLE,
                option_type VARCHAR,
                close DOUBLE,
                volume BIGINT,
                oi BIGINT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS market_calendar (
                date DATE PRIMARY KEY,
                is_trading_day BOOLEAN,
                is_weekend BOOLEAN,
                is_holiday BOOLEAN,
                session_type VARCHAR,
                session_start TIME,
                session_end TIME,
                is_expiry BOOLEAN,
                shifted_expiry_date DATE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS universe_members (
                symbol VARCHAR,
                start_date DATE,
                end_date DATE,
                is_active BOOLEAN,
                exit_reason VARCHAR
            )
        """)
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
        for table in [
            "fundamental_pit",
            "promoter_holding_pit",
            "financials_pit",
            "corporate_actions_pit",
            "index_membership_pit",
        ]:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    symbol VARCHAR,
                    observation_date DATE,
                    release_time TIMESTAMP,
                    available_at TIMESTAMP,
                    valid_from TIMESTAMP,
                    valid_to TIMESTAMP,
                    source VARCHAR,
                    payload JSON
                )
            """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_security_master_symbol ON security_master(symbol)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_equity_history_symbol_date ON equity_history(symbol, date)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_universe_members_symbol ON universe_members(symbol)"
        )
        conn.close()
    except Exception:
        # Settings import must not make CLI tools unusable if DuckDB is absent.
        pass


_bootstrap_duckdb_schema()

# Global database lock for DuckDB writes
DB_LOCK = threading.Lock()

# ClickHouse Configuration (Production)
# Sourced from environment variables to prevent secret leakage
CLICKHOUSE_HOST = os.environ.get("CLICKHOUSE_HOST", "localhost")
CLICKHOUSE_PORT = int(os.environ.get("CLICKHOUSE_PORT", "8123"))
CLICKHOUSE_USER = os.environ.get("CLICKHOUSE_USER", "default")
CLICKHOUSE_PASSWORD = os.environ.get("CLICKHOUSE_PASSWORD", "")
CLICKHOUSE_DATABASE = os.environ.get("CLICKHOUSE_DATABASE", "market")
CLICKHOUSE_SECURE = os.environ.get("CLICKHOUSE_SECURE", "False").lower() == "true"

# Logs
LOG_DIR = BASE_DIR / "logs"

# Reports
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DATA_QUALITY_DIR = REPORTS_DIR / "data_quality"

# Pipelines
PIPELINES_DIR = BASE_DIR / "pipelines"

# Tests
TESTS_DIR = BASE_DIR / "tests"

# NSE API Configuration
NSE_API_RETRY_ATTEMPTS = 3
NSE_API_RETRY_DELAY = 5  # seconds
NSE_API_RATE_LIMIT_MIN = 1  # seconds
NSE_API_RATE_LIMIT_MAX = 3  # seconds

# Data Quality Thresholds
MAX_MISSING_DATES_RATIO = 0.05  # 5% of trading days
MAX_DUPLICATE_ROWS = 0
MAX_ZERO_VOLUME_RATIO = 0.01  # 1% of records

# Trading Calendar
TRADING_CALENDAR_FILE = MASTER_DIR / "trading_calendar.parquet"

# Symbol Master (Legacy - for backward compatibility)
SYMBOL_MASTER_FILE = MASTER_DIR / "symbols.parquet"
SYMBOL_METADATA_FILE = MASTER_DIR / "symbol_metadata.parquet"

# Security Master (New - single source of truth)
SECURITY_MASTER_FILE = MASTER_DIR / "security_master.parquet"

# Sector Mapping
SECTOR_MAPPING_FILE = MASTER_DIR / "sectors.parquet"

# Data Catalog
DATASETS_CATALOG_FILE = CATALOG_DIR / "datasets.yaml"

# Trading Contract (Frozen Problem Definition)
TRADING_CONTRACT = {}
