"""
Prediction Evaluator

Evaluates prediction accuracy, calibration, and performance metrics.
Computes confusion matrix, ROC, precision, recall, F1, Brier score, etc.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix,
    calibration_curve
)
from ..prediction_tracker.prediction_tracker import Prediction, PredictionOutcome, SignalDirection


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics."""
    # Accuracy metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Calibration metrics
    brier_score: float
    expected_calibration_error: float
    
    # ROC metrics
    roc_auc: float
    
    # Confusion matrix
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    
    # Performance metrics
    avg_expected_return: float
    avg_actual_return: float
    win_rate: float
    profit_factor: float
    
    # Timing
    avg_holding_days: float
    
    # Sample size
    total_predictions: int
    completed_predictions: int


class PredictionEvaluator:
    """
    Evaluates prediction performance across multiple dimensions.
    
    Metrics computed:
    - Accuracy, Precision, Recall, F1
    - Brier Score, Expected Calibration Error
    - ROC AUC
    - Confusion Matrix
    - Win Rate, Profit Factor
    - Average Holding Period
    """
    
    def evaluate_predictions(
        self,
        predictions: List[Prediction],
        min_confidence: Optional[float] = None
    ) -> EvaluationMetrics:
        """
        Evaluate a list of predictions.
        
        Args:
            predictions: List of predictions to evaluate
            min_confidence: Minimum confidence threshold to filter
            
        Returns:
            Evaluation metrics
        """
        # Filter by confidence if specified
        if min_confidence:
            predictions = [p for p in predictions if p.confidence >= min_confidence]
        
        # Filter completed predictions
        completed = [p for p in predictions if p.outcome != PredictionOutcome.PENDING]
        
        if not completed:
            return self._empty_metrics(len(predictions))
        
        # Prepare data for sklearn metrics
        y_true = []
        y_pred = []
        y_proba = []
        
        actual_returns = []
        expected_returns = []
        
        for p in completed:
            # Convert direction to binary (BUY=1, SELL=0)
            y_true.append(1 if p.direction == SignalDirection.BUY else 0)
            
            # Convert outcome to binary (HIT_TARGET=1, others=0)
            y_pred.append(1 if p.outcome == PredictionOutcome.HIT_TARGET else 0)
            
            # Use probability
            y_proba.append(p.probability if p.direction == SignalDirection.BUY else 1 - p.probability)
            
            # Returns
            if p.outcome_price:
                actual_return = (p.outcome_price - p.entry_price) / p.entry_price * 100
                if p.direction == SignalDirection.SELL:
                    actual_return = -actual_return
                actual_returns.append(actual_return)
            else:
                actual_returns.append(0.0)
            
            expected_returns.append(p.expected_return_pct if p.direction == SignalDirection.BUY else -p.expected_return_pct)
        
        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        brier = brier_score_loss(y_true, y_proba)
        
        try:
            roc_auc = roc_auc_score(y_true, y_proba)
        except ValueError:
            roc_auc = 0.5  # Default if only one class
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
        
        # Calibration
        ece = self._calculate_ece(y_true, y_proba, n_bins=10)
        
        # Performance metrics
        wins = [r for r in actual_returns if r > 0]
        losses = [r for r in actual_returns if r < 0]
        
        win_rate = len(wins) / len(actual_returns) if actual_returns else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        avg_expected = np.mean(expected_returns) if expected_returns else 0
        avg_actual = np.mean(actual_returns) if actual_returns else 0
        
        # Holding period
        holding_days = [p.actual_holding_days for p in completed if p.actual_holding_days]
        avg_holding = np.mean(holding_days) if holding_days else 0
        
        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            brier_score=brier,
            expected_calibration_error=ece,
            roc_auc=roc_auc,
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
            avg_expected_return=avg_expected,
            avg_actual_return=avg_actual,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_holding_days=avg_holding,
            total_predictions=len(predictions),
            completed_predictions=len(completed)
        )
    
    def _calculate_ece(self, y_true: List, y_proba: List, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
        
        ece = 0.0
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        for i in range(n_bins):
            bin_mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
            if np.sum(bin_mask) > 0:
                ece += np.sum(bin_mask) / len(y_proba) * abs(prob_true[i] - prob_pred[i])
        
        return ece
    
    def _empty_metrics(self, total: int) -> EvaluationMetrics:
        """Return empty metrics for no completed predictions."""
        return EvaluationMetrics(
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            brier_score=0.0,
            expected_calibration_error=0.0,
            roc_auc=0.5,
            true_positives=0,
            true_negatives=0,
            false_positives=0,
            false_negatives=0,
            avg_expected_return=0.0,
            avg_actual_return=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_holding_days=0.0,
            total_predictions=total,
            completed_predictions=0
        )
    
    def get_calibration_curve(
        self,
        predictions: List[Prediction]
    ) -> Tuple[List[float], List[float]]:
        """
        Get calibration curve data.
        
        Returns:
            (mean_predicted_values, fraction_of_positives)
        """
        completed = [p for p in predictions if p.outcome != PredictionOutcome.PENDING]
        
        if not completed:
            return [], []
        
        y_true = []
        y_proba = []
        
        for p in completed:
            y_true.append(1 if p.outcome == PredictionOutcome.HIT_TARGET else 0)
            y_proba.append(p.probability if p.direction == SignalDirection.BUY else 1 - p.probability)
        
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
        
        return prob_pred.tolist(), prob_true.tolist()
