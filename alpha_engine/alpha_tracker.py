"""
Alpha Tracker

Tracks alpha performance over time for calibration and improvement.

STEP 13: Alpha Tracking

This module:
1. Stores alpha predictions with actual returns
2. Calculates performance metrics by alpha grade
3. Tracks win rates, average returns, and other metrics
4. Provides data for calibration and research
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from alpha_engine.alpha_builder import AlphaGrade
from utils.logger import get_logger

logger = get_logger("alpha_engine.tracker")


class PredictionOutcome(Enum):
    """Outcome classifications for predictions."""
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PENDING = "pending"


@dataclass
class AlphaRecord:
    """
    Single alpha prediction record.
    """
    date: date
    symbol: str
    alpha_score: float
    grade: AlphaGrade
    predicted_direction: str  # "long" or "short"
    entry_price: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    actual_return: Optional[float]
    outcome: PredictionOutcome
    regime: str
    category_scores: Dict[str, float]
    signals_used: List[str]
    features_used: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "date": self.date.isoformat(),
            "symbol": self.symbol,
            "alpha_score": round(self.alpha_score, 2),
            "grade": self.grade.value,
            "predicted_direction": self.predicted_direction,
            "entry_price": self.entry_price,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "actual_return": round(self.actual_return, 4) if self.actual_return else None,
            "outcome": self.outcome.value,
            "regime": self.regime,
            "category_scores": {k: round(v, 2) for k, v in self.category_scores.items()},
            "signals_used": self.signals_used,
            "features_used": self.features_used,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class GradePerformance:
    """
    Performance metrics for a specific alpha grade.
    """
    grade: AlphaGrade
    total_predictions: int
    correct_predictions: int
    incorrect_predictions: int
    pending_predictions: int
    win_rate: float
    average_return: float
    average_win: float
    average_loss: float
    sharpe_ratio: float
    max_drawdown: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "grade": self.grade.value,
            "total_predictions": self.total_predictions,
            "correct_predictions": self.correct_predictions,
            "incorrect_predictions": self.incorrect_predictions,
            "pending_predictions": self.pending_predictions,
            "win_rate": round(self.win_rate, 4),
            "average_return": round(self.average_return, 4),
            "average_win": round(self.average_win, 4),
            "average_loss": round(self.average_loss, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
        }


class AlphaTracker:
    """
    Tracks alpha performance over time.
    
    This module stores all alpha predictions and their outcomes,
    enabling performance analysis and calibration.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize Alpha Tracker.
        
        Args:
            storage_path: Optional path to store tracking data
        """
        self._logger = logger
        self.storage_path = storage_path
        
        # In-memory storage
        self.records: List[AlphaRecord] = []
        
        # Performance cache
        self._grade_performance_cache: Optional[Dict[AlphaGrade, GradePerformance]] = None
    
    def record_prediction(
        self,
        date: date,
        symbol: str,
        alpha_score: float,
        grade: AlphaGrade,
        predicted_direction: str,
        entry_price: float,
        target_price: Optional[float],
        stop_loss: Optional[float],
        regime: str,
        category_scores: Dict[str, float],
        signals_used: List[str],
        features_used: List[str],
    ) -> AlphaRecord:
        """
        Record a new alpha prediction.
        
        Args:
            date: Prediction date
            symbol: Stock symbol
            alpha_score: Alpha score
            grade: Alpha grade
            predicted_direction: "long" or "short"
            entry_price: Entry price
            target_price: Optional target price
            stop_loss: Optional stop loss
            regime: Market regime at prediction time
            category_scores: Category scores
            signals_used: List of signals used
            features_used: List of features used
            
        Returns:
            AlphaRecord
        """
        record = AlphaRecord(
            date=date,
            symbol=symbol,
            alpha_score=alpha_score,
            grade=grade,
            predicted_direction=predicted_direction,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            actual_return=None,
            outcome=PredictionOutcome.PENDING,
            regime=regime,
            category_scores=category_scores,
            signals_used=signals_used,
            features_used=features_used,
        )
        
        self.records.append(record)
        self._grade_performance_cache = None  # Invalidate cache
        
        self._logger.info(
            f"Recorded prediction for {symbol} on {date}",
            extra={
                "alpha_score": round(alpha_score, 2),
                "grade": grade.value,
                "direction": predicted_direction,
            },
        )
        
        return record
    
    def update_outcome(
        self,
        symbol: str,
        date: date,
        actual_return: float,
        outcome: PredictionOutcome,
    ) -> Optional[AlphaRecord]:
        """
        Update outcome for a prediction.
        
        Args:
            symbol: Stock symbol
            date: Prediction date
            actual_return: Actual return achieved
            outcome: Prediction outcome
            
        Returns:
            Updated AlphaRecord if found, None otherwise
        """
        for record in self.records:
            if record.symbol == symbol and record.date == date:
                record.actual_return = actual_return
                record.outcome = outcome
                self._grade_performance_cache = None  # Invalidate cache
                
                self._logger.info(
                    f"Updated outcome for {symbol} on {date}",
                    extra={
                        "actual_return": round(actual_return, 4),
                        "outcome": outcome.value,
                    },
                )
                
                return record
        
        self._logger.warning(f"Record not found for {symbol} on {date}")
        return None
    
    def get_grade_performance(self) -> Dict[AlphaGrade, GradePerformance]:
        """
        Calculate performance metrics by alpha grade.
        
        Returns:
            Dictionary of AlphaGrade -> GradePerformance
        """
        if self._grade_performance_cache is not None:
            return self._grade_performance_cache
        
        # Group records by grade
        grade_records: Dict[AlphaGrade, List[AlphaRecord]] = {}
        for record in self.records:
            if record.grade not in grade_records:
                grade_records[record.grade] = []
            grade_records[record.grade].append(record)
        
        # Calculate performance for each grade
        performance = {}
        
        for grade, records in grade_records.items():
            completed = [r for r in records if r.outcome != PredictionOutcome.PENDING]
            
            if not completed:
                # No completed predictions yet
                performance[grade] = GradePerformance(
                    grade=grade,
                    total_predictions=len(records),
                    correct_predictions=0,
                    incorrect_predictions=0,
                    pending_predictions=len(records),
                    win_rate=0.0,
                    average_return=0.0,
                    average_win=0.0,
                    average_loss=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                )
                continue
            
            correct = [r for r in completed if r.outcome == PredictionOutcome.CORRECT]
            incorrect = [r for r in completed if r.outcome == PredictionOutcome.INCORRECT]
            
            returns = [r.actual_return for r in completed if r.actual_return is not None]
            wins = [r.actual_return for r in correct if r.actual_return is not None]
            losses = [r.actual_return for r in incorrect if r.actual_return is not None]
            
            win_rate = len(correct) / len(completed) if completed else 0.0
            avg_return = np.mean(returns) if returns else 0.0
            avg_win = np.mean(wins) if wins else 0.0
            avg_loss = np.mean(losses) if losses else 0.0
            
            # Calculate Sharpe ratio (assuming risk-free rate = 0)
            sharpe = (
                np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0.0
            )
            
            # Calculate max drawdown
            cumulative_returns = np.cumprod(1 + np.array(returns))
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
            
            performance[grade] = GradePerformance(
                grade=grade,
                total_predictions=len(records),
                correct_predictions=len(correct),
                incorrect_predictions=len(incorrect),
                pending_predictions=len(records) - len(completed),
                win_rate=win_rate,
                average_return=avg_return,
                average_win=avg_win,
                average_loss=avg_loss,
                sharpe_ratio=sharpe,
                max_drawdown=max_dd,
            )
        
        self._grade_performance_cache = performance
        return performance
    
    def get_records_by_date_range(
        self,
        start_date: date,
        end_date: date,
    ) -> List[AlphaRecord]:
        """
        Get records within a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of AlphaRecord
        """
        return [
            r for r in self.records
            if start_date <= r.date <= end_date
        ]
    
    def get_records_by_symbol(self, symbol: str) -> List[AlphaRecord]:
        """
        Get all records for a specific symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of AlphaRecord
        """
        return [r for r in self.records if r.symbol == symbol]
    
    def get_records_by_regime(self, regime: str) -> List[AlphaRecord]:
        """
        Get all records for a specific regime.
        
        Args:
            regime: Regime name
            
        Returns:
            List of AlphaRecord
        """
        return [r for r in self.records if r.regime == regime]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get overall performance summary.
        
        Returns:
            Dictionary with performance summary
        """
        grade_performance = self.get_grade_performance()
        
        total_predictions = len(self.records)
        completed = sum(1 for r in self.records if r.outcome != PredictionOutcome.PENDING)
        pending = total_predictions - completed
        
        overall_correct = sum(1 for r in self.records if r.outcome == PredictionOutcome.CORRECT)
        overall_win_rate = overall_correct / completed if completed > 0 else 0.0
        
        all_returns = [r.actual_return for r in self.records if r.actual_return is not None]
        overall_avg_return = np.mean(all_returns) if all_returns else 0.0
        
        return {
            "total_predictions": total_predictions,
            "completed_predictions": completed,
            "pending_predictions": pending,
            "overall_win_rate": round(overall_win_rate, 4),
            "overall_average_return": round(overall_avg_return, 4),
            "grade_performance": {
                grade.value: perf.to_dict()
                for grade, perf in grade_performance.items()
            },
        }
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Convert all records to pandas DataFrame for analysis.
        
        Returns:
            DataFrame with all alpha records
        """
        data = [record.to_dict() for record in self.records]
        return pd.DataFrame(data)
    
    def save_to_file(self, path: Optional[str] = None) -> None:
        """
        Save records to file.
        
        Args:
            path: Optional path to save to (defaults to storage_path)
        """
        save_path = path or self.storage_path
        if save_path is None:
            self._logger.warning("No storage path provided, cannot save")
            return
        
        df = self.get_dataframe()
        df.to_csv(save_path, index=False)
        
        self._logger.info(f"Saved {len(self.records)} records to {save_path}")
    
    def load_from_file(self, path: Optional[str] = None) -> None:
        """
        Load records from file.
        
        Args:
            path: Optional path to load from (defaults to storage_path)
        """
        load_path = path or self.storage_path
        if load_path is None:
            self._logger.warning("No storage path provided, cannot load")
            return
        
        try:
            df = pd.read_csv(load_path)
            
            self.records = []
            for _, row in df.iterrows():
                record = AlphaRecord(
                    date=pd.to_datetime(row["date"]).date(),
                    symbol=row["symbol"],
                    alpha_score=row["alpha_score"],
                    grade=AlphaGrade(row["grade"]),
                    predicted_direction=row["predicted_direction"],
                    entry_price=row["entry_price"],
                    target_price=row.get("target_price"),
                    stop_loss=row.get("stop_loss"),
                    actual_return=row.get("actual_return"),
                    outcome=PredictionOutcome(row["outcome"]),
                    regime=row["regime"],
                    category_scores=eval(row["category_scores"]) if isinstance(row["category_scores"], str) else row["category_scores"],
                    signals_used=eval(row["signals_used"]) if isinstance(row["signals_used"], str) else row["signals_used"],
                    features_used=eval(row["features_used"]) if isinstance(row["features_used"], str) else row["features_used"],
                    timestamp=pd.to_datetime(row["timestamp"]) if "timestamp" in row else datetime.now(),
                )
                self.records.append(record)
            
            self._grade_performance_cache = None
            self._logger.info(f"Loaded {len(self.records)} records from {load_path}")
            
        except Exception as e:
            self._logger.error(f"Failed to load records from {load_path}: {e}")
    
    def clear_old_records(self, days_to_keep: int = 365) -> int:
        """
        Clear records older than specified days.
        
        Args:
            days_to_keep: Number of days of records to keep
            
        Returns:
            Number of records removed
        """
        cutoff_date = date.today() - pd.Timedelta(days=days_to_keep)
        
        original_count = len(self.records)
        self.records = [r for r in self.records if r.date >= cutoff_date]
        removed_count = original_count - len(self.records)
        
        self._grade_performance_cache = None
        
        self._logger.info(f"Removed {removed_count} old records (kept last {days_to_keep} days)")
        
        return removed_count
