"""
Interaction Database

Stores and retrieves interaction test results.
Provides persistent storage for factor-condition performance data.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import sqlite3
import json

from research.interactions.interaction_engine.interaction_engine import InteractionResult
from research.interactions.condition_engine.condition import Condition
from utils.logger import get_logger

logger = get_logger("research.interactions.interaction_database")


@dataclass
class InteractionRecord:
    """Database record for an interaction test."""
    interaction_id: str
    factor_name: str
    condition: dict
    ic: float
    rank_ic: float
    sharpe: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    num_trades: int
    decision: str
    test_date: datetime
    validation_status: Optional[str] = None
    confidence_level: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "interaction_id": self.interaction_id,
            "factor_name": self.factor_name,
            "condition": self.condition,
            "ic": round(self.ic, 4),
            "rank_ic": round(self.rank_ic, 4),
            "sharpe": round(self.sharpe, 4),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "num_trades": self.num_trades,
            "decision": self.decision,
            "test_date": self.test_date.isoformat(),
            "validation_status": self.validation_status,
            "confidence_level": self.confidence_level,
        }


class InteractionDatabase:
    """
    Stores and retrieves interaction test results.
    
    Provides:
    - Persistent storage
    - Query by factor
    - Query by condition
    - Query by performance
    - Update capabilities
    """
    
    def __init__(self, db_path: str = "research/interactions/interaction_database/interactions.db"):
        """
        Initialize interaction database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._logger = get_logger("research.interactions.interaction_database")
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                interaction_id TEXT PRIMARY KEY,
                factor_name TEXT NOT NULL,
                condition TEXT NOT NULL,
                ic REAL,
                rank_ic REAL,
                sharpe REAL,
                win_rate REAL,
                profit_factor REAL,
                max_drawdown REAL,
                num_trades INTEGER,
                decision TEXT,
                test_date TEXT NOT NULL,
                validation_status TEXT,
                confidence_level TEXT
            )
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_factor_name ON interactions(factor_name)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_decision ON interactions(decision)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_test_date ON interactions(test_date)
        """)
        
        conn.commit()
        conn.close()
        
        self._logger.info(f"Initialized database at {self.db_path}")
    
    def store_result(self, result: InteractionResult) -> str:
        """
        Store an interaction result.
        
        Args:
            result: InteractionResult to store
            
        Returns:
            Interaction ID
        """
        # Generate interaction ID
        interaction_id = f"{result.factor_name}_{hash(str(result.condition.serialize()))}"
        
        record = InteractionRecord(
            interaction_id=interaction_id,
            factor_name=result.factor_name,
            condition=result.condition.serialize(),
            ic=result.ic,
            rank_ic=result.rank_ic,
            sharpe=result.sharpe,
            win_rate=result.win_rate,
            profit_factor=result.profit_factor,
            max_drawdown=result.max_drawdown,
            num_trades=result.num_trades,
            decision=result.decision,
            test_date=datetime.now(),
        )
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO interactions
            (interaction_id, factor_name, condition, ic, rank_ic, sharpe, win_rate,
             profit_factor, max_drawdown, num_trades, decision, test_date,
             validation_status, confidence_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.interaction_id,
            record.factor_name,
            json.dumps(record.condition),
            record.ic,
            record.rank_ic,
            record.sharpe,
            record.win_rate,
            record.profit_factor,
            record.max_drawdown,
            record.num_trades,
            record.decision,
            record.test_date.isoformat(),
            record.validation_status,
            record.confidence_level,
        ))
        
        conn.commit()
        conn.close()
        
        self._logger.info(f"Stored interaction result: {interaction_id}")
        return interaction_id
    
    def get_result(self, interaction_id: str) -> Optional[InteractionRecord]:
        """
        Retrieve an interaction result by ID.
        
        Args:
            interaction_id: Interaction ID
            
        Returns:
            InteractionRecord or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT interaction_id, factor_name, condition, ic, rank_ic, sharpe, win_rate,
                   profit_factor, max_drawdown, num_trades, decision, test_date,
                   validation_status, confidence_level
            FROM interactions
            WHERE interaction_id = ?
        """, (interaction_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return InteractionRecord(
            interaction_id=row[0],
            factor_name=row[1],
            condition=json.loads(row[2]),
            ic=row[3],
            rank_ic=row[4],
            sharpe=row[5],
            win_rate=row[6],
            profit_factor=row[7],
            max_drawdown=row[8],
            num_trades=row[9],
            decision=row[10],
            test_date=datetime.fromisoformat(row[11]),
            validation_status=row[12],
            confidence_level=row[13],
        )
    
    def query_by_factor(self, factor_name: str) -> List[InteractionRecord]:
        """
        Query results by factor name.
        
        Args:
            factor_name: Factor name to query
            
        Returns:
            List of InteractionRecord
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT interaction_id, factor_name, condition, ic, rank_ic, sharpe, win_rate,
                   profit_factor, max_drawdown, num_trades, decision, test_date,
                   validation_status, confidence_level
            FROM interactions
            WHERE factor_name = ?
            ORDER BY ic DESC
        """, (factor_name,))
        
        rows = cursor.fetchall()
        conn.close()
        
        records = []
        for row in rows:
            records.append(InteractionRecord(
                interaction_id=row[0],
                factor_name=row[1],
                condition=json.loads(row[2]),
                ic=row[3],
                rank_ic=row[4],
                sharpe=row[5],
                win_rate=row[6],
                profit_factor=row[7],
                max_drawdown=row[8],
                num_trades=row[9],
                decision=row[10],
                test_date=datetime.fromisoformat(row[11]),
                validation_status=row[12],
                confidence_level=row[13],
            ))
        
        return records
    
    def query_by_decision(self, decision: str) -> List[InteractionRecord]:
        """
        Query results by decision.
        
        Args:
            decision: Decision to query (PASS, FAIL, NEUTRAL)
            
        Returns:
            List of InteractionRecord
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT interaction_id, factor_name, condition, ic, rank_ic, sharpe, win_rate,
                   profit_factor, max_drawdown, num_trades, decision, test_date,
                   validation_status, confidence_level
            FROM interactions
            WHERE decision = ?
            ORDER BY ic DESC
        """, (decision,))
        
        rows = cursor.fetchall()
        conn.close()
        
        records = []
        for row in rows:
            records.append(InteractionRecord(
                interaction_id=row[0],
                factor_name=row[1],
                condition=json.loads(row[2]),
                ic=row[3],
                rank_ic=row[4],
                sharpe=row[5],
                win_rate=row[6],
                profit_factor=row[7],
                max_drawdown=row[8],
                num_trades=row[9],
                decision=row[10],
                test_date=datetime.fromisoformat(row[11]),
                validation_status=row[12],
                confidence_level=row[13],
            ))
        
        return records
    
    def get_top_performers(self, metric: str = "ic", n: int = 10) -> List[InteractionRecord]:
        """
        Get top performing interactions.
        
        Args:
            metric: Metric to sort by (ic, sharpe, win_rate)
            n: Number of top performers to return
            
        Returns:
            List of InteractionRecord
        """
        valid_metrics = ["ic", "sharpe", "win_rate", "profit_factor"]
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric: {metric}. Must be one of {valid_metrics}")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f"""
            SELECT interaction_id, factor_name, condition, ic, rank_ic, sharpe, win_rate,
                   profit_factor, max_drawdown, num_trades, decision, test_date,
                   validation_status, confidence_level
            FROM interactions
            WHERE decision = 'PASS'
            ORDER BY {metric} DESC
            LIMIT ?
        """, (n,))
        
        rows = cursor.fetchall()
        conn.close()
        
        records = []
        for row in rows:
            records.append(InteractionRecord(
                interaction_id=row[0],
                factor_name=row[1],
                condition=json.loads(row[2]),
                ic=row[3],
                rank_ic=row[4],
                sharpe=row[5],
                win_rate=row[6],
                profit_factor=row[7],
                max_drawdown=row[8],
                num_trades=row[9],
                decision=row[10],
                test_date=datetime.fromisoformat(row[11]),
                validation_status=row[12],
                confidence_level=row[13],
            ))
        
        return records
    
    def get_all(self) -> List[InteractionRecord]:
        """
        Get all stored interaction results.
        
        Returns:
            List of InteractionRecord
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT interaction_id, factor_name, condition, ic, rank_ic, sharpe, win_rate,
                   profit_factor, max_drawdown, num_trades, decision, test_date,
                   validation_status, confidence_level
            FROM interactions
            ORDER BY test_date DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        records = []
        for row in rows:
            records.append(InteractionRecord(
                interaction_id=row[0],
                factor_name=row[1],
                condition=json.loads(row[2]),
                ic=row[3],
                rank_ic=row[4],
                sharpe=row[5],
                win_rate=row[6],
                profit_factor=row[7],
                max_drawdown=row[8],
                num_trades=row[9],
                decision=row[10],
                test_date=datetime.fromisoformat(row[11]),
                validation_status=row[12],
                confidence_level=row[13],
            ))
        
        return records
    
    def count(self) -> int:
        """
        Get count of stored interactions.
        
        Returns:
            Number of stored interactions
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM interactions")
        count = cursor.fetchone()[0]
        
        conn.close()
        
        return count
    
    def clear(self) -> None:
        """Clear all stored interactions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM interactions")
        
        conn.commit()
        conn.close()
        
        self._logger.info("Cleared all interactions from database")


def get_database(db_path: Optional[str] = None) -> InteractionDatabase:
    """
    Convenience function to get database instance.
    
    Args:
        db_path: Optional database path
        
    Returns:
        InteractionDatabase instance
    """
    return InteractionDatabase(db_path=db_path)
