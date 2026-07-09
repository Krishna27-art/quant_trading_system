"""
Meta Alpha Database

Persistent storage for evidence intelligence engine.
Stores evidence, quality scores, weights, fusion results, probability, confidence, recommendations, and outcomes.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime
import sqlite3
from pathlib import Path

from utils.logger import get_logger

logger = get_logger("meta_alpha.database")


@dataclass
class EvidenceRecord:
    """Database record for evidence."""
    evidence_id: str
    source: str
    factor_name: str
    category: str
    signal_direction: str
    strength: float
    confidence: float
    timestamp: datetime
    metadata: Optional[str]
    quality_score: Optional[float]
    weight: Optional[float]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "evidence_id": self.evidence_id,
            "source": self.source,
            "factor_name": self.factor_name,
            "category": self.category,
            "signal_direction": self.signal_direction,
            "strength": round(self.strength, 4),
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "quality_score": round(self.quality_score, 4) if self.quality_score is not None else None,
            "weight": round(self.weight, 4) if self.weight is not None else None,
        }


@dataclass
class PredictionRecord:
    """Database record for predictions."""
    prediction_id: str
    symbol: str
    evidence_ids: str  # JSON string
    fusion_result: str  # JSON string
    probability: float
    confidence_level: str
    expected_return: float
    action: str
    entry_price: Optional[float]
    stop_loss: Optional[float]
    target_price: Optional[float]
    holding_period: int
    timestamp: datetime
    actual_outcome: Optional[float]
    outcome_timestamp: Optional[datetime]
    is_correct: Optional[bool]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "prediction_id": self.prediction_id,
            "symbol": self.symbol,
            "evidence_ids": self.evidence_ids,
            "fusion_result": self.fusion_result,
            "probability": round(self.probability, 4),
            "confidence_level": self.confidence_level,
            "expected_return": round(self.expected_return, 4),
            "action": self.action,
            "entry_price": round(self.entry_price, 2) if self.entry_price else None,
            "stop_loss": round(self.stop_loss, 2) if self.stop_loss else None,
            "target_price": round(self.target_price, 2) if self.target_price else None,
            "holding_period": self.holding_period,
            "timestamp": self.timestamp.isoformat(),
            "actual_outcome": round(self.actual_outcome, 4) if self.actual_outcome is not None else None,
            "outcome_timestamp": self.outcome_timestamp.isoformat() if self.outcome_timestamp else None,
            "is_correct": self.is_correct,
        }


class MetaAlphaDatabase:
    """
    Persistent storage for evidence intelligence engine.
    
    Tables:
    - evidence: Evidence records
    - predictions: Prediction records with outcomes
    - quality_scores: Quality score history
    - weights: Weight history
    """
    
    def __init__(self, db_path: str = "data/meta_alpha.db"):
        """
        Initialize database.
        
        Args:
            db_path: Path to database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = get_logger("meta_alpha.database")
        
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Evidence table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evidence (
                    evidence_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    factor_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    signal_direction TEXT NOT NULL,
                    strength REAL NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    quality_score REAL,
                    weight REAL
                )
            """)
            
            # Predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    evidence_ids TEXT NOT NULL,
                    fusion_result TEXT NOT NULL,
                    probability REAL NOT NULL,
                    confidence_level TEXT NOT NULL,
                    expected_return REAL NOT NULL,
                    action TEXT NOT NULL,
                    entry_price REAL,
                    stop_loss REAL,
                    target_price REAL,
                    holding_period INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    actual_outcome REAL,
                    outcome_timestamp TEXT,
                    is_correct BOOLEAN
                )
            """)
            
            # Quality scores table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quality_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evidence_id TEXT NOT NULL,
                    data_quality REAL NOT NULL,
                    historical_ic REAL NOT NULL,
                    historical_sharpe REAL NOT NULL,
                    regime_stability REAL NOT NULL,
                    missing_values REAL NOT NULL,
                    overall_score REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (evidence_id) REFERENCES evidence(evidence_id)
                )
            """)
            
            # Weights table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS weights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id TEXT NOT NULL,
                    evidence_id TEXT NOT NULL,
                    weight REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_evidence_source ON evidence(source)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_evidence_category ON evidence(category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON predictions(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_quality_evidence ON quality_scores(evidence_id)")
            
            conn.commit()
        
        self._logger.info(f"Initialized database at {self.db_path}")
    
    def store_evidence(self, record: EvidenceRecord) -> None:
        """
        Store evidence record.
        
        Args:
            record: EvidenceRecord to store
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO evidence
                (evidence_id, source, factor_name, category, signal_direction, strength, confidence, timestamp, metadata, quality_score, weight)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.evidence_id,
                record.source,
                record.factor_name,
                record.category,
                record.signal_direction,
                record.strength,
                record.confidence,
                record.timestamp.isoformat(),
                record.metadata,
                record.quality_score,
                record.weight,
            ))
            conn.commit()
    
    def store_prediction(self, record: PredictionRecord) -> None:
        """
        Store prediction record.
        
        Args:
            record: PredictionRecord to store
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO predictions
                (prediction_id, symbol, evidence_ids, fusion_result, probability, confidence_level, expected_return, action, entry_price, stop_loss, target_price, holding_period, timestamp, actual_outcome, outcome_timestamp, is_correct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.prediction_id,
                record.symbol,
                record.evidence_ids,
                record.fusion_result,
                record.probability,
                record.confidence_level,
                record.expected_return,
                record.action,
                record.entry_price,
                record.stop_loss,
                record.target_price,
                record.holding_period,
                record.timestamp.isoformat(),
                record.actual_outcome,
                record.outcome_timestamp.isoformat() if record.outcome_timestamp else None,
                record.is_correct,
            ))
            conn.commit()
    
    def update_prediction_outcome(
        self,
        prediction_id: str,
        actual_outcome: float,
        is_correct: bool,
    ) -> None:
        """
        Update prediction with outcome.
        
        Args:
            prediction_id: Prediction ID
            actual_outcome: Actual return
            is_correct: Whether prediction was correct
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE predictions
                SET actual_outcome = ?, outcome_timestamp = ?, is_correct = ?
                WHERE prediction_id = ?
            """, (
                actual_outcome,
                datetime.now().isoformat(),
                is_correct,
                prediction_id,
            ))
            conn.commit()
    
    def get_prediction(self, prediction_id: str) -> Optional[PredictionRecord]:
        """
        Get prediction by ID.
        
        Args:
            prediction_id: Prediction ID
            
        Returns:
            PredictionRecord or None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM predictions WHERE prediction_id = ?
            """, (prediction_id,))
            
            row = cursor.fetchone()
            if row:
                return PredictionRecord(
                    prediction_id=row[0],
                    symbol=row[1],
                    evidence_ids=row[2],
                    fusion_result=row[3],
                    probability=row[4],
                    confidence_level=row[5],
                    expected_return=row[6],
                    action=row[7],
                    entry_price=row[8],
                    stop_loss=row[9],
                    target_price=row[10],
                    holding_period=row[11],
                    timestamp=datetime.fromisoformat(row[12]),
                    actual_outcome=row[13],
                    outcome_timestamp=datetime.fromisoformat(row[14]) if row[14] else None,
                    is_correct=row[15],
                )
        return None
    
    def get_predictions_by_symbol(self, symbol: str) -> List[PredictionRecord]:
        """
        Get predictions for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of PredictionRecord
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM predictions WHERE symbol = ? ORDER BY timestamp DESC
            """, (symbol,))
            
            records = []
            for row in cursor.fetchall():
                records.append(PredictionRecord(
                    prediction_id=row[0],
                    symbol=row[1],
                    evidence_ids=row[2],
                    fusion_result=row[3],
                    probability=row[4],
                    confidence_level=row[5],
                    expected_return=row[6],
                    action=row[7],
                    entry_price=row[8],
                    stop_loss=row[9],
                    target_price=row[10],
                    holding_period=row[11],
                    timestamp=datetime.fromisoformat(row[12]),
                    actual_outcome=row[13],
                    outcome_timestamp=datetime.fromisoformat(row[14]) if row[14] else None,
                    is_correct=row[15],
                ))
        
        return records
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get overall performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total predictions
            cursor.execute("SELECT COUNT(*) FROM predictions")
            total = cursor.fetchone()[0]
            
            # Completed predictions
            cursor.execute("SELECT COUNT(*) FROM predictions WHERE actual_outcome IS NOT NULL")
            completed = cursor.fetchone()[0]
            
            # Correct predictions
            cursor.execute("SELECT COUNT(*) FROM predictions WHERE is_correct = 1")
            correct = cursor.fetchone()[0]
            
            # Accuracy
            accuracy = correct / completed if completed > 0 else 0.0
            
            # Average return
            cursor.execute("SELECT AVG(actual_outcome) FROM predictions WHERE actual_outcome IS NOT NULL")
            avg_return = cursor.fetchone()[0] or 0.0
            
            return {
                "total_predictions": total,
                "completed_predictions": completed,
                "correct_predictions": correct,
                "accuracy": accuracy,
                "average_return": avg_return,
            }
    
    def clear_old_data(self, days: int = 90) -> int:
        """
        Clear old data from database.
        
        Args:
            days: Number of days to keep
            
        Returns:
            Number of records removed
        """
        cutoff = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Delete old predictions
            cursor.execute("""
                DELETE FROM predictions WHERE timestamp < ?
            """, (cutoff,))
            
            deleted = cursor.rowcount
            conn.commit()
        
        self._logger.info(f"Cleared {deleted} old records")
        return deleted


# Global database instance
_global_database = None


def get_database(db_path: str = "data/meta_alpha.db") -> MetaAlphaDatabase:
    """
    Get global database instance.
    
    Args:
        db_path: Path to database file
        
    Returns:
        MetaAlphaDatabase instance
    """
    global _global_database
    if _global_database is None:
        _global_database = MetaAlphaDatabase(db_path)
    return _global_database
