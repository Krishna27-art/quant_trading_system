"""
Learning Loop

Learns from every prediction to update evidence quality and weights.
Continuous learning from completed trades.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from meta_alpha.evidence_engine.evidence import Evidence
from meta_alpha.evidence_weighting.quality_score import QualityScore
from utils.logger import get_logger

logger = get_logger("meta_alpha.monitoring")


@dataclass
class PredictionRecord:
    """Record of a prediction and its outcome."""
    prediction_id: str
    symbol: str
    evidence_ids: List[str]
    predicted_probability: float
    predicted_action: str
    actual_outcome: Optional[float]  # Actual return
    is_correct: Optional[bool]
    timestamp: datetime
    outcome_timestamp: Optional[datetime]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "prediction_id": self.prediction_id,
            "symbol": self.symbol,
            "evidence_ids": self.evidence_ids,
            "predicted_probability": round(self.predicted_probability, 4),
            "predicted_action": self.predicted_action,
            "actual_outcome": round(self.actual_outcome, 4) if self.actual_outcome is not None else None,
            "is_correct": self.is_correct,
            "timestamp": self.timestamp.isoformat(),
            "outcome_timestamp": self.outcome_timestamp.isoformat() if self.outcome_timestamp else None,
        }


@dataclass
class LearningMetrics:
    """Metrics from learning loop."""
    total_predictions: int
    correct_predictions: int
    accuracy: float
    calibration_error: float
    sharpe_ratio: float
    win_rate: float
    evidence_performance: Dict[str, float]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "total_predictions": self.total_predictions,
            "correct_predictions": self.correct_predictions,
            "accuracy": round(self.accuracy, 4),
            "calibration_error": round(self.calibration_error, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "win_rate": round(self.win_rate, 4),
            "evidence_performance": {k: round(v, 4) for k, v in self.evidence_performance.items()},
        }


class LearningLoop:
    """
    Learns from every prediction to update evidence quality and weights.
    
    Process:
    - Record predictions
    - Record outcomes
    - Calculate performance metrics
    - Update evidence quality scores
    - Adjust weights based on performance
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        min_samples: int = 30,
    ):
        """
        Initialize learning loop.
        
        Args:
            learning_rate: Rate at which to update weights
            min_samples: Minimum samples before updating
        """
        self.learning_rate = learning_rate
        self.min_samples = min_samples
        self._predictions: Dict[str, PredictionRecord] = {}
        self._evidence_performance: Dict[str, List[float]] = {}
        self._counter = 0
        self._logger = get_logger("meta_alpha.monitoring")
    
    def record_prediction(
        self,
        symbol: str,
        evidence_list: List[Evidence],
        predicted_probability: float,
        predicted_action: str,
    ) -> str:
        """
        Record a prediction.
        
        Args:
            symbol: Stock symbol
            evidence_list: List of Evidence used
            predicted_probability: Predicted probability
            predicted_action: Predicted action
            
        Returns:
            Prediction ID
        """
        prediction_id = f"pred_{self._counter}"
        self._counter += 1
        
        evidence_ids = [f"{e.source}_{e.factor_name}" for e in evidence_list]
        
        record = PredictionRecord(
            prediction_id=prediction_id,
            symbol=symbol,
            evidence_ids=evidence_ids,
            predicted_probability=predicted_probability,
            predicted_action=predicted_action,
            actual_outcome=None,
            is_correct=None,
            timestamp=datetime.now(),
            outcome_timestamp=None,
        )
        
        self._predictions[prediction_id] = record
        
        self._logger.info(f"Recorded prediction {prediction_id} for {symbol}")
        return prediction_id
    
    def record_outcome(
        self,
        prediction_id: str,
        actual_outcome: float,
    ) -> None:
        """
        Record outcome for a prediction.
        
        Args:
            prediction_id: Prediction ID
            actual_outcome: Actual return
        """
        if prediction_id not in self._predictions:
            self._logger.warning(f"Prediction {prediction_id} not found")
            return
        
        record = self._predictions[prediction_id]
        record.actual_outcome = actual_outcome
        record.outcome_timestamp = datetime.now()
        
        # Determine if prediction was correct
        if record.predicted_action == "BUY":
            record.is_correct = actual_outcome > 0
        elif record.predicted_action == "SELL":
            record.is_correct = actual_outcome < 0
        else:
            record.is_correct = True  # HOLD is always "correct"
        
        # Update evidence performance
        self._update_evidence_performance(record)
        
        self._logger.info(f"Recorded outcome for {prediction_id}: {actual_outcome:.4f}")
    
    def _update_evidence_performance(self, record: PredictionRecord) -> None:
        """
        Update evidence performance tracking.
        
        Args:
            record: Prediction record with outcome
        """
        outcome = record.actual_outcome if record.actual_outcome is not None else 0.0
        
        for evidence_id in record.evidence_ids:
            if evidence_id not in self._evidence_performance:
                self._evidence_performance[evidence_id] = []
            
            self._evidence_performance[evidence_id].append(outcome)
    
    def calculate_metrics(self) -> LearningMetrics:
        """
        Calculate learning metrics.
        
        Returns:
            LearningMetrics
        """
        completed = [r for r in self._predictions.values() if r.actual_outcome is not None]
        
        if not completed:
            return LearningMetrics(
                total_predictions=len(self._predictions),
                correct_predictions=0,
                accuracy=0.0,
                calibration_error=0.0,
                sharpe_ratio=0.0,
                win_rate=0.0,
                evidence_performance={},
            )
        
        # Calculate accuracy
        correct = sum(1 for r in completed if r.is_correct)
        accuracy = correct / len(completed)
        
        # Calculate calibration error
        predicted_probs = [r.predicted_probability for r in completed]
        actual_outcomes = [1.0 if r.is_correct else 0.0 for r in completed]
        calibration_error = np.mean(np.array(predicted_probs) - np.array(actual_outcomes))
        
        # Calculate Sharpe ratio
        returns = [r.actual_outcome for r in completed if r.actual_outcome is not None]
        if returns:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0.0
        else:
            sharpe = 0.0
        
        # Calculate win rate
        win_rate = accuracy
        
        # Calculate evidence performance
        evidence_performance = {}
        for evidence_id, outcomes in self._evidence_performance.items():
            if len(outcomes) >= self.min_samples:
                evidence_performance[evidence_id] = np.mean(outcomes)
        
        return LearningMetrics(
            total_predictions=len(self._predictions),
            correct_predictions=correct,
            accuracy=accuracy,
            calibration_error=calibration_error,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            evidence_performance=evidence_performance,
        )
    
    def update_quality_scores(
        self,
        quality_scores: Dict[str, QualityScore],
    ) -> Dict[str, QualityScore]:
        """
        Update quality scores based on learning.
        
        Args:
            quality_scores: Current quality scores
            
        Returns:
            Updated quality scores
        """
        metrics = self.calculate_metrics()
        
        updated_scores = {}
        
        for evidence_id, current_score in quality_scores.items():
            if evidence_id in metrics.evidence_performance:
                # Update based on recent performance
                recent_performance = metrics.evidence_performance[evidence_id]
                
                # Adjust historical IC based on performance
                # Positive performance -> higher IC
                ic_adjustment = (recent_performance + 0.1) / 0.2  # Normalize to 0-1
                new_ic = current_score.historical_ic * (1.0 - self.learning_rate) + ic_adjustment * 100 * self.learning_rate
                
                # Update quality score
                updated_score = QualityScore(
                    data_quality=current_score.data_quality,
                    historical_ic=new_ic,
                    historical_sharpe=current_score.historical_sharpe,
                    regime_stability=current_score.regime_stability,
                    missing_values=current_score.missing_values,
                    overall_score=0.0,  # Will be recalculated
                )
                
                # Recalculate overall score
                weights = {
                    "data_quality": 0.2,
                    "historical_ic": 0.25,
                    "historical_sharpe": 0.25,
                    "regime_stability": 0.2,
                    "missing_values": 0.1,
                }
                
                updated_score.overall_score = (
                    updated_score.data_quality * weights["data_quality"] +
                    updated_score.historical_ic * weights["historical_ic"] +
                    updated_score.historical_sharpe * weights["historical_sharpe"] +
                    updated_score.regime_stability * weights["regime_stability"] +
                    updated_score.missing_values * weights["missing_values"]
                )
                
                updated_scores[evidence_id] = updated_score
            else:
                updated_scores[evidence_id] = current_score
        
        return updated_scores
    
    def get_recent_performance(
        self,
        evidence_id: str,
        n: int = 30,
    ) -> Optional[List[float]]:
        """
        Get recent performance for an evidence source.
        
        Args:
            evidence_id: Evidence identifier
            n: Number of recent outcomes
            
        Returns:
            List of recent outcomes
        """
        if evidence_id not in self._evidence_performance:
            return None
        
        outcomes = self._evidence_performance[evidence_id]
        return outcomes[-n:] if len(outcomes) >= n else outcomes
    
    def get_prediction_history(self, symbol: Optional[str] = None) -> List[PredictionRecord]:
        """
        Get prediction history.
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            List of PredictionRecord
        """
        records = list(self._predictions.values())
        
        if symbol:
            records = [r for r in records if r.symbol == symbol]
        
        return sorted(records, key=lambda x: x.timestamp, reverse=True)
    
    def clear_old_predictions(self, days: int = 90) -> int:
        """
        Clear old predictions from memory.
        
        Args:
            days: Number of days to keep
            
        Returns:
            Number of predictions removed
        """
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(days=days)
        
        to_remove = [
            pid for pid, record in self._predictions.items()
            if record.timestamp < cutoff
        ]
        
        for pid in to_remove:
            del self._predictions[pid]
        
        self._logger.info(f"Cleared {len(to_remove)} old predictions")
        return len(to_remove)


def record_prediction(
    symbol: str,
    evidence_list: List[Evidence],
    predicted_probability: float,
    predicted_action: str,
) -> str:
    """
    Convenience function to record prediction.
    
    Args:
        symbol: Stock symbol
        evidence_list: List of Evidence
        predicted_probability: Predicted probability
        predicted_action: Predicted action
        
    Returns:
        Prediction ID
    """
    loop = LearningLoop()
    return loop.record_prediction(symbol, evidence_list, predicted_probability, predicted_action)
