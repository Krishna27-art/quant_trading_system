"""
Immutable Prediction Tracker

Tracks all predictions with full immutability for trustworthy performance measurement.
Every prediction is stored once and never edited.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import sqlite3
from pathlib import Path


class PredictionOutcome(Enum):
    """Possible outcomes for a prediction."""
    PENDING = "PENDING"
    HIT_TARGET = "HIT_TARGET"
    HIT_STOP = "HIT_STOP"
    EXPIRED = "EXPIRED"
    PARTIAL = "PARTIAL"


class SignalDirection(Enum):
    """Signal direction."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Prediction:
    """Immutable prediction record."""
    prediction_id: str
    symbol: str
    direction: SignalDirection
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: Optional[float]
    probability: float
    confidence: float
    expected_return_pct: float
    expected_holding_days: int
    worst_case_pct: float
    best_case_pct: float
    win_probability: float
    
    # Versioning
    model_version: str
    feature_version: str
    dataset_version: str
    
    # Explainability
    reasons: List[str]
    feature_importance: Dict[str, float]
    
    # Tracking
    created_at: datetime
    outcome: PredictionOutcome = PredictionOutcome.PENDING
    outcome_price: Optional[float] = None
    outcome_timestamp: Optional[datetime] = None
    max_profit_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    actual_holding_days: Optional[int] = None


class PredictionTracker:
    """
    Immutable prediction tracking database.
    
    Once a prediction is created, it cannot be modified.
    Only outcome tracking is allowed.
    """
    
    def __init__(self, db_path: str = "data/predictions.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize the predictions database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                target_1 REAL NOT NULL,
                target_2 REAL,
                probability REAL NOT NULL,
                confidence REAL NOT NULL,
                expected_return_pct REAL NOT NULL,
                expected_holding_days INTEGER NOT NULL,
                worst_case_pct REAL NOT NULL,
                best_case_pct REAL NOT NULL,
                win_probability REAL NOT NULL,
                model_version TEXT NOT NULL,
                feature_version TEXT NOT NULL,
                dataset_version TEXT NOT NULL,
                reasons TEXT NOT NULL,
                feature_importance TEXT NOT NULL,
                created_at TEXT NOT NULL,
                outcome TEXT NOT NULL,
                outcome_price REAL,
                outcome_timestamp TEXT,
                max_profit_pct REAL,
                max_drawdown_pct REAL,
                actual_holding_days INTEGER
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_prediction(self, prediction: Prediction) -> str:
        """
        Create a new immutable prediction.
        
        Args:
            prediction: Prediction object
            
        Returns:
            Prediction ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO predictions VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            prediction.prediction_id,
            prediction.symbol,
            prediction.direction.value,
            prediction.entry_price,
            prediction.stop_loss,
            prediction.target_1,
            prediction.target_2,
            prediction.probability,
            prediction.confidence,
            prediction.expected_return_pct,
            prediction.expected_holding_days,
            prediction.worst_case_pct,
            prediction.best_case_pct,
            prediction.win_probability,
            prediction.model_version,
            prediction.feature_version,
            prediction.dataset_version,
            json.dumps(prediction.reasons),
            json.dumps(prediction.feature_importance),
            prediction.created_at.isoformat(),
            prediction.outcome.value,
            prediction.outcome_price,
            prediction.outcome_timestamp.isoformat() if prediction.outcome_timestamp else None,
            prediction.max_profit_pct,
            prediction.max_drawdown_pct,
            prediction.actual_holding_days
        ))
        
        conn.commit()
        conn.close()
        
        return prediction.prediction_id
    
    def update_outcome(
        self,
        prediction_id: str,
        outcome: PredictionOutcome,
        outcome_price: Optional[float] = None,
        max_profit_pct: Optional[float] = None,
        max_drawdown_pct: Optional[float] = None,
        actual_holding_days: Optional[int] = None
    ) -> None:
        """
        Update prediction outcome (only this field is mutable).
        
        Args:
            prediction_id: Prediction ID
            outcome: Final outcome
            outcome_price: Price at outcome
            max_profit_pct: Maximum profit percentage achieved
            max_drawdown_pct: Maximum drawdown percentage experienced
            actual_holding_days: Actual holding period
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE predictions
            SET outcome = ?,
                outcome_price = ?,
                outcome_timestamp = ?,
                max_profit_pct = ?,
                max_drawdown_pct = ?,
                actual_holding_days = ?
            WHERE prediction_id = ?
        """, (
            outcome.value,
            outcome_price,
            datetime.now().isoformat(),
            max_profit_pct,
            max_drawdown_pct,
            actual_holding_days,
            prediction_id
        ))
        
        conn.commit()
        conn.close()
    
    def get_prediction(self, prediction_id: str) -> Optional[Prediction]:
        """Get a prediction by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM predictions WHERE prediction_id = ?", (prediction_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return self._row_to_prediction(row)
    
    def get_pending_predictions(self, symbol: Optional[str] = None) -> List[Prediction]:
        """Get all pending predictions, optionally filtered by symbol."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if symbol:
            cursor.execute(
                "SELECT * FROM predictions WHERE outcome = 'PENDING' AND symbol = ?",
                (symbol,)
            )
        else:
            cursor.execute("SELECT * FROM predictions WHERE outcome = 'PENDING'")
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_prediction(row) for row in rows]
    
    def get_completed_predictions(self, limit: int = 100) -> List[Prediction]:
        """Get completed predictions for evaluation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM predictions WHERE outcome != 'PENDING' ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_prediction(row) for row in rows]
    
    def _row_to_prediction(self, row) -> Prediction:
        """Convert database row to Prediction object."""
        return Prediction(
            prediction_id=row[0],
            symbol=row[1],
            direction=SignalDirection(row[2]),
            entry_price=row[3],
            stop_loss=row[4],
            target_1=row[5],
            target_2=row[6],
            probability=row[7],
            confidence=row[8],
            expected_return_pct=row[9],
            expected_holding_days=row[10],
            worst_case_pct=row[11],
            best_case_pct=row[12],
            win_probability=row[13],
            model_version=row[14],
            feature_version=row[15],
            dataset_version=row[16],
            reasons=json.loads(row[17]),
            feature_importance=json.loads(row[18]),
            created_at=datetime.fromisoformat(row[19]),
            outcome=PredictionOutcome(row[20]),
            outcome_price=row[21],
            outcome_timestamp=datetime.fromisoformat(row[22]) if row[22] else None,
            max_profit_pct=row[23],
            max_drawdown_pct=row[24],
            actual_holding_days=row[25]
        )


import json
