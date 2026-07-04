"""
Institutional Alpha Discovery Platform - Database Schema & Connection Engine

Manages SQLite/PostgreSQL tables for:
1. Experiment Tracking (hyperparameters, walk-forward OOS Sharpe, IC, ECE)
2. Feature Sets & Versioning
3. Model Registry
4. Alpha Predictions & Outcomes
"""

import os
import sqlite3
import json
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger("QuantPlatformDB")

# Default database path inside quant_platform/database/alpha_platform.db
DEFAULT_DB_PATH = Path(__file__).resolve().parent / "alpha_platform.db"

def get_db_path() -> str:
    return os.getenv("ALPHA_PLATFORM_DB", str(DEFAULT_DB_PATH))

@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Yields a transactional SQLite connection with foreign keys and WAL mode enabled.
    """
    db_path = get_db_path()
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise e
    finally:
        conn.close()

def init_db() -> None:
    """
    Initialize schema tables if they do not exist.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # 1. Experiments Table (Institutional Experiment Tracker)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            git_commit_hash TEXT,
            dataset_version TEXT NOT NULL,
            feature_set_id TEXT NOT NULL,
            model_type TEXT NOT NULL,
            hyperparameters TEXT NOT NULL,
            oos_sharpe REAL,
            information_coefficient_ic REAL,
            ece_calibration_error REAL,
            win_rate REAL,
            top_shap_features TEXT,
            notes TEXT
        );
        """)

        # 2. Feature Sets Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS feature_sets (
            feature_set_id TEXT PRIMARY KEY,
            description TEXT,
            feature_names TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """)

        # 3. Model Registry Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_registry (
            model_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            version TEXT NOT NULL,
            architecture TEXT NOT NULL,
            target_horizon TEXT NOT NULL,
            training_period_start TEXT,
            training_period_end TEXT,
            status TEXT DEFAULT 'STAGING',
            created_at TEXT NOT NULL
        );
        """)

        # 4. Alpha Predictions Table (The Output Contract)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS alpha_predictions (
            prediction_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            prediction TEXT NOT NULL,
            entry_price REAL,
            target_price REAL,
            stop_loss REAL,
            calibrated_probability REAL,
            confidence_score REAL,
            expected_return_pct REAL,
            reasons TEXT,
            model_id TEXT,
            actual_outcome TEXT DEFAULT 'PENDING',
            resolved_at TEXT,
            FOREIGN KEY(model_id) REFERENCES model_registry(model_id)
        );
        """)

        # Create indices for fast querying
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_exp_timestamp ON experiments(timestamp);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_exp_sharpe ON experiments(oos_sharpe DESC);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_symbol ON alpha_predictions(symbol);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_timestamp ON alpha_predictions(timestamp);")

    logger.info("✅ Institutional Alpha Platform database schema initialized.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
    print("Database initialized successfully.")
