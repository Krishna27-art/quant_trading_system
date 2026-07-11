"""
Knowledge Database

Persistent storage for all completed predictions and associated learning data.
Stores predictions, outcomes, factor performance, regime, weights, calibration, drift, and lessons learned.
Never deletes data.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime
import sqlite3
from pathlib import Path
import json

from utils.logger import get_logger

logger = get_logger("continuous_learning.retraining")


@dataclass
class KnowledgeRecord:
    """Complete knowledge record for a prediction."""
    prediction_id: str
    symbol: str
    prediction_timestamp: datetime
    outcome_timestamp: Optional[datetime]
    action: str
    predicted_probability: float
    predicted_confidence: str
    expected_return: float
    actual_return: Optional[float]
    is_successful: Optional[bool]
    outcome_type: Optional[str]
    evidence_ids: str  # JSON string
    weights: str  # JSON string
    market_regime: Optional[str]
    calibration_error: Optional[float]
    drift_metrics: Optional[str]  # JSON string
    factor_performance: Optional[str]  # JSON string
    lessons_learned: Optional[str]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "prediction_id": self.prediction_id,
            "symbol": self.symbol,
            "prediction_timestamp": self.prediction_timestamp.isoformat(),
            "outcome_timestamp": self.outcome_timestamp.isoformat() if self.outcome_timestamp else None,
            "action": self.action,
            "predicted_probability": round(self.predicted_probability, 4),
            "predicted_confidence": self.predicted_confidence,
            "expected_return": round(self.expected_return, 4),
            "actual_return": round(self.actual_return, 4) if self.actual_return is not None else None,
            "is_successful": self.is_successful,
            "outcome_type": self.outcome_type,
            "evidence_ids": self.evidence_ids,
            "weights": self.weights,
            "market_regime": self.market_regime,
            "calibration_error": round(self.calibration_error, 4) if self.calibration_error is not None else None,
            "drift_metrics": self.drift_metrics,
            "factor_performance": self.factor_performance,
            "lessons_learned": self.lessons_learned,
        }


class KnowledgeDatabase:
    """
    Persistent storage for all completed predictions.
    
    Stores:
    - Predictions
    - Outcomes
    - Factor performance
    - Regime
    - Weights
    - Calibration
    - Drift
    - Lessons learned
    """
    
    def __init__(self, db_path: str = "data/knowledge.db"):
        """
        Initialize knowledge database.
        
        Args:
            db_path: Path to database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = get_logger("continuous_learning.retraining")
        
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Knowledge table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge (
                    prediction_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    prediction_timestamp TEXT NOT NULL,
                    outcome_timestamp TEXT,
                    action TEXT NOT NULL,
                    predicted_probability REAL NOT NULL,
                    predicted_confidence TEXT NOT NULL,
                    expected_return REAL NOT NULL,
                    actual_return REAL,
                    is_successful BOOLEAN,
                    outcome_type TEXT,
                    evidence_ids TEXT,
                    weights TEXT,
                    market_regime TEXT,
                    calibration_error REAL,
                    drift_metrics TEXT,
                    factor_performance TEXT,
                    lessons_learned TEXT
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_symbol ON knowledge(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_timestamp ON knowledge(prediction_timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_regime ON knowledge(market_regime)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_outcome ON knowledge(outcome_type)")
            
            conn.commit()
        
        self._logger.info(f"Initialized knowledge database at {self.db_path}")
    
    def store_record(self, record: KnowledgeRecord) -> None:
        """
        Store knowledge record.
        
        Args:
            record: KnowledgeRecord to store
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO knowledge
                (prediction_id, symbol, prediction_timestamp, outcome_timestamp, action, predicted_probability, predicted_confidence, expected_return, actual_return, is_successful, outcome_type, evidence_ids, weights, market_regime, calibration_error, drift_metrics, factor_performance, lessons_learned)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.prediction_id,
                record.symbol,
                record.prediction_timestamp.isoformat(),
                record.outcome_timestamp.isoformat() if record.outcome_timestamp else None,
                record.action,
                record.predicted_probability,
                record.predicted_confidence,
                record.expected_return,
                record.actual_return,
                record.is_successful,
                record.outcome_type,
                record.evidence_ids,
                record.weights,
                record.market_regime,
                record.calibration_error,
                record.drift_metrics,
                record.factor_performance,
                record.lessons_learned,
            ))
            conn.commit()
    
    def get_record(self, prediction_id: str) -> Optional[KnowledgeRecord]:
        """
        Get knowledge record by ID.
        
        Args:
            prediction_id: Prediction ID
            
        Returns:
            KnowledgeRecord or None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM knowledge WHERE prediction_id = ?
            """, (prediction_id,))
            
            row = cursor.fetchone()
            if row:
                return KnowledgeRecord(
                    prediction_id=row[0],
                    symbol=row[1],
                    prediction_timestamp=datetime.fromisoformat(row[2]),
                    outcome_timestamp=datetime.fromisoformat(row[3]) if row[3] else None,
                    action=row[4],
                    predicted_probability=row[5],
                    predicted_confidence=row[6],
                    expected_return=row[7],
                    actual_return=row[8],
                    is_successful=row[9],
                    outcome_type=row[10],
                    evidence_ids=row[11],
                    weights=row[12],
                    market_regime=row[13],
                    calibration_error=row[14],
                    drift_metrics=row[15],
                    factor_performance=row[16],
                    lessons_learned=row[17],
                )
        return None
    
    def get_records_by_symbol(self, symbol: str, limit: int = 100) -> List[KnowledgeRecord]:
        """
        Get knowledge records for a symbol.
        
        Args:
            symbol: Stock symbol
            limit: Maximum number of records
            
        Returns:
            List of KnowledgeRecord
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM knowledge WHERE symbol = ? ORDER BY prediction_timestamp DESC LIMIT ?
            """, (symbol, limit))
            
            records = []
            for row in cursor.fetchall():
                records.append(KnowledgeRecord(
                    prediction_id=row[0],
                    symbol=row[1],
                    prediction_timestamp=datetime.fromisoformat(row[2]),
                    outcome_timestamp=datetime.fromisoformat(row[3]) if row[3] else None,
                    action=row[4],
                    predicted_probability=row[5],
                    predicted_confidence=row[6],
                    expected_return=row[7],
                    actual_return=row[8],
                    is_successful=row[9],
                    outcome_type=row[10],
                    evidence_ids=row[11],
                    weights=row[12],
                    market_regime=row[13],
                    calibration_error=row[14],
                    drift_metrics=row[15],
                    factor_performance=row[16],
                    lessons_learned=row[17],
                ))
        
        return records
    
    def get_records_by_regime(self, regime: str, limit: int = 100) -> List[KnowledgeRecord]:
        """
        Get knowledge records for a regime.
        
        Args:
            regime: Market regime
            limit: Maximum number of records
            
        Returns:
            List of KnowledgeRecord
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM knowledge WHERE market_regime = ? ORDER BY prediction_timestamp DESC LIMIT ?
            """, (regime, limit))
            
            records = []
            for row in cursor.fetchall():
                records.append(KnowledgeRecord(
                    prediction_id=row[0],
                    symbol=row[1],
                    prediction_timestamp=datetime.fromisoformat(row[2]),
                    outcome_timestamp=datetime.fromisoformat(row[3]) if row[3] else None,
                    action=row[4],
                    predicted_probability=row[5],
                    predicted_confidence=row[6],
                    expected_return=row[7],
                    actual_return=row[8],
                    is_successful=row[9],
                    outcome_type=row[10],
                    evidence_ids=row[11],
                    weights=row[12],
                    market_regime=row[13],
                    calibration_error=row[14],
                    drift_metrics=row[15],
                    factor_performance=row[16],
                    lessons_learned=row[17],
                ))
        
        return records
    
    def get_completed_records(self, limit: int = 100) -> List[KnowledgeRecord]:
        """
        Get completed knowledge records.
        
        Args:
            limit: Maximum number of records
            
        Returns:
            List of KnowledgeRecord
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM knowledge WHERE actual_return IS NOT NULL ORDER BY prediction_timestamp DESC LIMIT ?
            """, (limit,))
            
            records = []
            for row in cursor.fetchall():
                records.append(KnowledgeRecord(
                    prediction_id=row[0],
                    symbol=row[1],
                    prediction_timestamp=datetime.fromisoformat(row[2]),
                    outcome_timestamp=datetime.fromisoformat(row[3]) if row[3] else None,
                    action=row[4],
                    predicted_probability=row[5],
                    predicted_confidence=row[6],
                    expected_return=row[7],
                    actual_return=row[8],
                    is_successful=row[9],
                    outcome_type=row[10],
                    evidence_ids=row[11],
                    weights=row[12],
                    market_regime=row[13],
                    calibration_error=row[14],
                    drift_metrics=row[15],
                    factor_performance=row[16],
                    lessons_learned=row[17],
                ))
        
        return records
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics from knowledge database.
        
        Returns:
            Dictionary with statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total records
            cursor.execute("SELECT COUNT(*) FROM knowledge")
            total = cursor.fetchone()[0]
            
            # Completed records
            cursor.execute("SELECT COUNT(*) FROM knowledge WHERE actual_return IS NOT NULL")
            completed = cursor.fetchone()[0]
            
            # Successful records
            cursor.execute("SELECT COUNT(*) FROM knowledge WHERE is_successful = 1")
            successful = cursor.fetchone()[0]
            
            # Average return
            cursor.execute("SELECT AVG(actual_return) FROM knowledge WHERE actual_return IS NOT NULL")
            avg_return = cursor.fetchone()[0] or 0.0
            
            # By regime
            cursor.execute("SELECT market_regime, COUNT(*) FROM knowledge GROUP BY market_regime")
            by_regime = {row[0]: row[1] for row in cursor.fetchall()}
            
            # By outcome type
            cursor.execute("SELECT outcome_type, COUNT(*) FROM knowledge WHERE outcome_type IS NOT NULL GROUP BY outcome_type")
            by_outcome = {row[0]: row[1] for row in cursor.fetchall()}
            
            return {
                "total_predictions": total,
                "completed_predictions": completed,
                "successful_predictions": successful,
                "average_return": avg_return,
                "by_regime": by_regime,
                "by_outcome": by_outcome,
            }
    
    def export_to_json(self, output_path: str) -> None:
        """
        Export all knowledge records to JSON.
        
        Args:
            output_path: Path to output JSON file
        """
        records = self.get_completed_records(limit=10000)
        
        data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_records": len(records),
            "records": [r.to_dict() for r in records],
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self._logger.info(f"Exported {len(records)} records to {output_path}")


# Global database instance
_global_knowledge_db = None


def get_knowledge_database(db_path: str = "data/knowledge.db") -> KnowledgeDatabase:
    """
    Get global knowledge database instance.
    
    Args:
        db_path: Path to database file
        
    Returns:
        KnowledgeDatabase instance
    """
    global _global_knowledge_db
    if _global_knowledge_db is None:
        _global_knowledge_db = KnowledgeDatabase(db_path)
    return _global_knowledge_db
