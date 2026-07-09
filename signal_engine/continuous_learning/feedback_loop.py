"""
Continuous Learning Feedback Loop

Tracks prediction outcomes and improves model performance.
Creates a feedback loop for continuous improvement.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from utils.logger import get_logger

logger = get_logger("signal_engine.continuous_learning")


@dataclass
class PredictionRecord:
    """Record of a prediction and its outcome."""
    prediction_id: str
    symbol: str
    prediction: float
    confidence: float
    prediction_time: datetime
    outcome: Optional[float]
    outcome_time: Optional[datetime]
    holding_period: Optional[int]
    realized_return: Optional[float]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "prediction_id": self.prediction_id,
            "symbol": self.symbol,
            "prediction": round(self.prediction, 4),
            "confidence": round(self.confidence, 4),
            "prediction_time": self.prediction_time.isoformat() if self.prediction_time else None,
            "outcome": self.outcome,
            "outcome_time": self.outcome_time.isoformat() if self.outcome_time else None,
            "holding_period": self.holding_period,
            "realized_return": round(self.realized_return, 4) if self.realized_return else None,
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for continuous learning."""
    total_predictions: int
    completed_predictions: int
    accuracy: float
    calibration_error: float
    average_return: float
    sharpe_ratio: float
    win_rate: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "total_predictions": self.total_predictions,
            "completed_predictions": self.completed_predictions,
            "accuracy": round(self.accuracy, 4),
            "calibration_error": round(self.calibration_error, 4),
            "average_return": round(self.average_return, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "win_rate": round(self.win_rate, 4),
        }


class FeedbackLoop:
    """
    Tracks prediction outcomes and improves model performance.
    
    Creates a feedback loop for:
    - Recording predictions
    - Tracking outcomes
    - Calculating performance metrics
    - Updating model statistics
    """
    
    def __init__(self, storage_path: str = "signal_engine/continuous_learning/data"):
        """
        Initialize feedback loop.
        
        Args:
            storage_path: Path to store prediction records
        """
        from pathlib import Path
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._predictions: Dict[str, PredictionRecord] = {}
        self._logger = get_logger("signal_engine.continuous_learning")
    
    def record_prediction(
        self,
        prediction_id: str,
        symbol: str,
        prediction: float,
        confidence: float,
    ) -> PredictionRecord:
        """
        Record a new prediction.
        
        Args:
            prediction_id: Unique prediction identifier
            symbol: Stock symbol
            prediction: Predicted probability (0 to 1)
            confidence: Prediction confidence (0 to 1)
            
        Returns:
            PredictionRecord
        """
        record = PredictionRecord(
            prediction_id=prediction_id,
            symbol=symbol,
            prediction=prediction,
            confidence=confidence,
            prediction_time=datetime.now(),
            outcome=None,
            outcome_time=None,
            holding_period=None,
            realized_return=None,
        )
        
        self._predictions[prediction_id] = record
        
        self._logger.info(f"Recorded prediction {prediction_id} for {symbol}")
        return record
    
    def record_outcome(
        self,
        prediction_id: str,
        outcome: float,
        holding_period: int,
        realized_return: float,
    ) -> PredictionRecord:
        """
        Record outcome for a prediction.
        
        Args:
            prediction_id: Prediction identifier
            outcome: Actual outcome (0 or 1)
            holding_period: Holding period in days
            realized_return: Realized return
            
        Returns:
            Updated PredictionRecord
        """
        if prediction_id not in self._predictions:
            raise ValueError(f"Prediction {prediction_id} not found")
        
        record = self._predictions[prediction_id]
        record.outcome = outcome
        record.outcome_time = datetime.now()
        record.holding_period = holding_period
        record.realized_return = realized_return
        
        self._logger.info(f"Recorded outcome for prediction {prediction_id}")
        return record
    
    def calculate_performance_metrics(self) -> PerformanceMetrics:
        """
        Calculate performance metrics from recorded predictions.
        
        Returns:
            PerformanceMetrics
        """
        total = len(self._predictions)
        completed = sum(1 for r in self._predictions.values() if r.outcome is not None)
        
        if completed == 0:
            return PerformanceMetrics(
                total_predictions=total,
                completed_predictions=completed,
                accuracy=0.0,
                calibration_error=0.0,
                average_return=0.0,
                sharpe_ratio=0.0,
                win_rate=0.0,
            )
        
        # Calculate accuracy
        correct = sum(
            1 for r in self._predictions.values()
            if r.outcome is not None and
            ((r.prediction > 0.5 and r.outcome == 1) or (r.prediction <= 0.5 and r.outcome == 0))
        )
        accuracy = correct / completed
        
        # Calculate calibration error
        predictions = [r.prediction for r in self._predictions.values() if r.outcome is not None]
        outcomes = [r.outcome for r in self._predictions.values() if r.outcome is not None]
        calibration_error = sum(abs(p - o) for p, o in zip(predictions, outcomes)) / completed
        
        # Calculate average return
        returns = [r.realized_return for r in self._predictions.values() if r.realized_return is not None]
        average_return = sum(returns) / len(returns) if returns else 0.0
        
        # Calculate Sharpe ratio
        if returns:
            returns_array = pd.Series(returns)
            sharpe = returns_array.mean() / returns_array.std() if returns_array.std() > 0 else 0.0
            sharpe_annualized = sharpe * np.sqrt(252) if len(returns) > 1 else 0.0
        else:
            sharpe_annualized = 0.0
        
        # Calculate win rate
        wins = sum(1 for r in returns if r > 0)
        win_rate = wins / len(returns) if returns else 0.0
        
        return PerformanceMetrics(
            total_predictions=total,
            completed_predictions=completed,
            accuracy=accuracy,
            calibration_error=calibration_error,
            average_return=average_return,
            sharpe_ratio=sharpe_annualized,
            win_rate=win_rate,
        )
    
    def get_pending_predictions(self) -> List[PredictionRecord]:
        """
        Get predictions waiting for outcomes.
        
        Returns:
            List of pending PredictionRecord
        """
        pending = [r for r in self._predictions.values() if r.outcome is None]
        return pending
    
    def get_completed_predictions(self) -> List[PredictionRecord]:
        """
        Get completed predictions with outcomes.
        
        Returns:
            List of completed PredictionRecord
        """
        completed = [r for r in self._predictions.values() if r.outcome is not None]
        return completed
    
    def save_to_disk(self) -> None:
        """Save prediction records to disk."""
        import json
        
        records = [r.to_dict() for r in self._predictions.values()]
        
        filepath = self.storage_path / "predictions.json"
        with open(filepath, "w") as f:
            json.dump(records, f, indent=2, default=str)
        
        self._logger.info(f"Saved {len(records)} predictions to disk")
    
    def load_from_disk(self) -> None:
        """Load prediction records from disk."""
        import json
        
        filepath = self.storage_path / "predictions.json"
        
        if not filepath.exists():
            self._logger.warning("No saved predictions found")
            return
        
        with open(filepath, "r") as f:
            records = json.load(f)
        
        for record_data in records:
            record = PredictionRecord(
                prediction_id=record_data["prediction_id"],
                symbol=record_data["symbol"],
                prediction=record_data["prediction"],
                confidence=record_data["confidence"],
                prediction_time=datetime.fromisoformat(record_data["prediction_time"]) if record_data["prediction_time"] else None,
                outcome=record_data.get("outcome"),
                outcome_time=datetime.fromisoformat(record_data["outcome_time"]) if record_data.get("outcome_time") else None,
                holding_period=record_data.get("holding_period"),
                realized_return=record_data.get("realized_return"),
            )
            self._predictions[record.prediction_id] = record
        
        self._logger.info(f"Loaded {len(records)} predictions from disk")


def record_prediction(
    prediction_id: str,
    symbol: str,
    prediction: float,
    confidence: float,
    storage_path: str = "signal_engine/continuous_learning/data",
) -> PredictionRecord:
    """
    Convenience function to record a prediction.
    
    Args:
        prediction_id: Unique prediction identifier
        symbol: Stock symbol
        prediction: Predicted probability
        confidence: Prediction confidence
        storage_path: Path to store records
        
    Returns:
        PredictionRecord
    """
    feedback = FeedbackLoop(storage_path=storage_path)
    return feedback.record_prediction(prediction_id, symbol, prediction, confidence)
