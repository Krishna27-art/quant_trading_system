"""
Directory Initialization Module

Automatically creates all required directories at startup.
"""

from config.settings import (
    BRONZE_CORPORATE_ACTIONS_DIR,
    # Bronze Layer
    BRONZE_DIR,
    BRONZE_EQUITY_HISTORY_DIR,
    BRONZE_FLOWS_DIR,
    BRONZE_OPTIONS_DIR,
    CATALOG_DIR,
    CLEAN_DIR,
    DATA_DIR,
    DATABASE_DIR,
    FEATURES_DIR,
    FEATURES_FLOW_DIR,
    FEATURES_FUNDAMENTALS_DIR,
    FEATURES_LIQUIDITY_DIR,
    FEATURES_MOMENTUM_DIR,
    FEATURES_OPTIONS_DIR,
    FEATURES_VOLATILITY_DIR,
    # Gold Layer
    GOLD_DIR,
    GOLD_FEATURES_DIR,
    LOG_DIR,
    MASTER_DIR,
    RAW_CORPORATE_ACTIONS_DIR,
    # Legacy directories
    RAW_DIR,
    RAW_EQUITY_DIR,
    RAW_FLOWS_DIR,
    RAW_OPTIONS_DIR,
    REPORTS_DATA_QUALITY_DIR,
    REPORTS_DIR,
    # Silver Layer
    SILVER_DIR,
    SILVER_EQUITY_HISTORY_DIR,
    SILVER_OPTIONS_DIR,
    TESTS_DIR,
)


def init_directories() -> None:
    """
    Create all required directories if they don't exist.

    This should be called at the start of every pipeline to ensure
    the directory structure is in place.
    """
    directories = [
        DATA_DIR,
        # Bronze Layer
        BRONZE_DIR,
        BRONZE_EQUITY_HISTORY_DIR,
        BRONZE_OPTIONS_DIR,
        BRONZE_CORPORATE_ACTIONS_DIR,
        BRONZE_FLOWS_DIR,
        # Silver Layer
        SILVER_DIR,
        SILVER_EQUITY_HISTORY_DIR,
        SILVER_OPTIONS_DIR,
        # Gold Layer
        GOLD_DIR,
        GOLD_FEATURES_DIR,
        # Legacy directories (for backward compatibility during migration)
        RAW_DIR,
        CLEAN_DIR,
        FEATURES_DIR,
        MASTER_DIR,
        CATALOG_DIR,
        RAW_EQUITY_DIR,
        RAW_OPTIONS_DIR,
        RAW_CORPORATE_ACTIONS_DIR,
        RAW_FLOWS_DIR,
        FEATURES_MOMENTUM_DIR,
        FEATURES_VOLATILITY_DIR,
        FEATURES_LIQUIDITY_DIR,
        FEATURES_OPTIONS_DIR,
        FEATURES_FLOW_DIR,
        FEATURES_FUNDAMENTALS_DIR,
        DATABASE_DIR,
        LOG_DIR,
        REPORTS_DIR,
        REPORTS_DATA_QUALITY_DIR,
        TESTS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    init_directories()
    print("All directories initialized successfully")
