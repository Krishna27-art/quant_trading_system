"""
Prediction Result

Stores actual results and compares them to predictions.
Tracks multiple dimensions: direction, entry quality, target quality, stop-loss quality, probability calibration.
"""

from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime
from enum import Enum

from prediction_layer.prediction_learning.prediction_history import PredictionMetadata

from utils.logger import get_logger

logger = get_logger("prediction_layer.prediction_learning.prediction_result")


class OutcomeType(Enum):
    """Outcome type enumeration."""
    TARGET_HIT = "TARGET_HIT"
    STOP_LOSS_HIT = "STOP_LOSS_HIT"
    TIME_EXIT = "TIME_EXIT"
    MANUAL_EXIT = "MANUAL_EXIT"
    CANCELLED = "CANCELLED"


class PredictionQuality(Enum):
    """Prediction quality classification."""
    PERFECT = "PERFECT"
    GOOD = "GOOD"
    NEUTRAL = "NEUTRAL"
    POOR = "POOR"
    FAILURE = "FAILURE"


@dataclass
class PredictionResult:
    """Actual result of a prediction."""
    prediction_id: str
    actual_entry_price: float
    actual_exit_price: float
    actual_return_percentage: float
    outcome_type: OutcomeType
    exit_timestamp: datetime
    holding_time_hours: float
    max_drawdown_percentage: float
    max_runup_percentage: float
    hit_target: bool
    hit_stop_loss: bool
    direction_correct: bool
    entry_quality_score: float
    target_quality_score: float
    stop_loss_quality_score: float
    probability_calibration_error: float
    confidence_calibration_score: float
    quality_classification: PredictionQuality
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "prediction_id": self.prediction_id,
            "actual_entry_price": self.actual_entry_price,
            "actual_exit_price": self.actual_exit_price,
            "actual_return_percentage": round(self.actual_return_percentage, 4),
            "outcome_type": self.outcome_type.value,
            "exit_timestamp": self.exit_timestamp.isoformat(),
            "holding_time_hours": round(self.holding_time_hours, 2),
            "max_drawdown_percentage": round(self.max_drawdown_percentage, 4),
            "max_runup_percentage": round(self.max_runup_percentage, 4),
            "hit_target": self.hit_target,
            "hit_stop_loss": self.hit_stop_loss,
            "direction_correct": self.direction_correct,
            "entry_quality_score": round(self.entry_quality_score, 4),
            "target_quality_score": round(self.target_quality_score, 4),
            "stop_loss_quality_score": round(self.stop_loss_quality_score, 4),
            "probability_calibration_error": round(self.probability_calibration_error, 4),
            "confidence_calibration_score": round(self.confidence_calibration_score, 4),
            "quality_classification": self.quality_classification.value,
        }


class PredictionResultCalculator:
    """
    Calculates prediction results and compares to predictions.
    
    Evaluates multiple dimensions:
    - Direction accuracy
    - Entry quality (how close to optimal entry)
    - Target quality (was target realistic)
    - Stop-loss quality (was stop-loss appropriate)
    - Probability calibration (did probability match actual success rate)
    - Confidence calibration (did high confidence predictions perform better)
    """
    
    def __init__(self):
        """Initialize prediction result calculator."""
        self._logger = get_logger("prediction_layer.prediction_learning.prediction_result")
    
    def calculate_result(
        self,
        prediction: PredictionMetadata,
        actual_entry_price: float,
        actual_exit_price: float,
        exit_timestamp: datetime,
        optimal_entry_price: Optional[float] = None,
        optimal_exit_price: Optional[float] = None,
        historical_probability_calibration: Optional[Dict[str, float]] = None,
    ) -> PredictionResult:
        """
        Calculate prediction result.
        
        Args:
            prediction: Original prediction metadata
            actual_entry_price: Actual entry price achieved
            actual_exit_price: Actual exit price
            exit_timestamp: Exit timestamp
            optimal_entry_price: Optional optimal entry price for quality assessment
            optimal_exit_price: Optional optimal exit price for quality assessment
            historical_probability_calibration: Historical calibration data
            
        Returns:
            PredictionResult
        """
        # Calculate return percentage
        actual_return = self._calculate_return_percentage(
            prediction.action.value,
            actual_entry_price,
            actual_exit_price,
        )
        
        # Determine outcome type
        outcome_type = self._determine_outcome_type(
            prediction,
            actual_exit_price,
        )
        
        # Calculate holding time
        holding_time = self._calculate_holding_time(
            prediction.prediction_timestamp,
            exit_timestamp,
        )
        
        # Calculate max drawdown and runup (would need intraday data)
        max_drawdown = 0.0  # Placeholder
        max_runup = 0.0  # Placeholder
        
        # Check if target/stop were hit
        hit_target = self._check_target_hit(prediction, actual_exit_price)
        hit_stop_loss = self._check_stop_loss_hit(prediction, actual_exit_price)
        
        # Check direction correctness
        direction_correct = self._check_direction_correct(
            prediction.action.value,
            actual_entry_price,
            actual_exit_price,
        )
        
        # Calculate quality scores
        entry_quality = self._calculate_entry_quality(
            prediction.entry_price,
            actual_entry_price,
            optimal_entry_price,
        )
        
        target_quality = self._calculate_target_quality(
            prediction.target_price,
            actual_exit_price,
            optimal_exit_price,
        )
        
        stop_loss_quality = self._calculate_stop_loss_quality(
            prediction.stop_loss,
            actual_exit_price,
            optimal_exit_price,
        )
        
        # Calculate probability calibration error
        probability_calibration_error = self._calculate_probability_calibration_error(
            prediction.probability,
            actual_return > 0,
            historical_probability_calibration,
        )
        
        # Calculate confidence calibration score
        confidence_calibration_score = self._calculate_confidence_calibration_score(
            prediction.confidence,
            actual_return,
            historical_probability_calibration,
        )
        
        # Classify overall quality
        quality_classification = self._classify_quality(
            actual_return,
            direction_correct,
            entry_quality,
            target_quality,
            stop_loss_quality,
        )
        
        self._logger.info(
            f"Calculated result for prediction {prediction.prediction_id}: "
            f"{quality_classification.value} (return={actual_return:.2f}%)"
        )
        
        return PredictionResult(
            prediction_id=prediction.prediction_id,
            actual_entry_price=actual_entry_price,
            actual_exit_price=actual_exit_price,
            actual_return_percentage=actual_return,
            outcome_type=outcome_type,
            exit_timestamp=exit_timestamp,
            holding_time_hours=holding_time,
            max_drawdown_percentage=max_drawdown,
            max_runup_percentage=max_runup,
            hit_target=hit_target,
            hit_stop_loss=hit_stop_loss,
            direction_correct=direction_correct,
            entry_quality_score=entry_quality,
            target_quality_score=target_quality,
            stop_loss_quality_score=stop_loss_quality,
            probability_calibration_error=probability_calibration_error,
            confidence_calibration_score=confidence_calibration_score,
            quality_classification=quality_classification,
        )
    
    def _calculate_return_percentage(
        self,
        action: str,
        entry_price: float,
        exit_price: float,
    ) -> float:
        """
        Calculate return percentage.
        
        Args:
            action: Trading action
            entry_price: Entry price
            exit_price: Exit price
            
        Returns:
            Return percentage
        """
        if action == "BUY":
            return ((exit_price - entry_price) / entry_price) * 100
        elif action == "SELL":
            return ((entry_price - exit_price) / entry_price) * 100
        else:
            return 0.0
    
    def _determine_outcome_type(
        self,
        prediction: PredictionMetadata,
        actual_exit_price: float,
    ) -> OutcomeType:
        """
        Determine outcome type.
        
        Args:
            prediction: Prediction metadata
            actual_exit_price: Actual exit price
            
        Returns:
            OutcomeType
        """
        if prediction.target_price and actual_exit_price >= prediction.target_price:
            return OutcomeType.TARGET_HIT
        elif prediction.stop_loss and actual_exit_price <= prediction.stop_loss:
            return OutcomeType.STOP_LOSS_HIT
        else:
            return OutcomeType.TIME_EXIT
    
    def _calculate_holding_time(
        self,
        entry_timestamp: datetime,
        exit_timestamp: datetime,
    ) -> float:
        """
        Calculate holding time in hours.
        
        Args:
            entry_timestamp: Entry timestamp
            exit_timestamp: Exit timestamp
            
        Returns:
            Holding time in hours
        """
        delta = exit_timestamp - entry_timestamp
        return delta.total_seconds() / 3600
    
    def _check_target_hit(
        self,
        prediction: PredictionMetadata,
        actual_exit_price: float,
    ) -> bool:
        """
        Check if target was hit.
        
        Args:
            prediction: Prediction metadata
            actual_exit_price: Actual exit price
            
        Returns:
            True if target was hit
        """
        if prediction.target_price is None:
            return False
        
        if prediction.action.value == "BUY":
            return actual_exit_price >= prediction.target_price
        else:
            return actual_exit_price <= prediction.target_price
    
    def _check_stop_loss_hit(
        self,
        prediction: PredictionMetadata,
        actual_exit_price: float,
    ) -> bool:
        """
        Check if stop loss was hit.
        
        Args:
            prediction: Prediction metadata
            actual_exit_price: Actual exit price
            
        Returns:
            True if stop loss was hit
        """
        if prediction.stop_loss is None:
            return False
        
        if prediction.action.value == "BUY":
            return actual_exit_price <= prediction.stop_loss
        else:
            return actual_exit_price >= prediction.stop_loss
    
    def _check_direction_correct(
        self,
        action: str,
        entry_price: float,
        exit_price: float,
    ) -> bool:
        """
        Check if direction prediction was correct.
        
        Args:
            action: Trading action
            entry_price: Entry price
            exit_price: Exit price
            
        Returns:
            True if direction was correct
        """
        if action == "BUY":
            return exit_price > entry_price
        elif action == "SELL":
            return exit_price < entry_price
        else:
            return True
    
    def _calculate_entry_quality(
        self,
        predicted_entry: float,
        actual_entry: float,
        optimal_entry: Optional[float],
    ) -> float:
        """
        Calculate entry quality score.
        
        Args:
            predicted_entry: Predicted entry price
            actual_entry: Actual entry price achieved
            optimal_entry: Optional optimal entry price
            
        Returns:
            Quality score (0-1)
        """
        if optimal_entry is None:
            # If no optimal entry, score based on how close actual was to predicted
            diff_pct = abs(actual_entry - predicted_entry) / predicted_entry
            return max(0.0, 1.0 - diff_pct * 10)  # 10% difference = 0 score
        
        # Score based on how close actual was to optimal
        diff_pct = abs(actual_entry - optimal_entry) / optimal_entry
        return max(0.0, 1.0 - diff_pct * 5)  # 20% difference = 0 score
    
    def _calculate_target_quality(
        self,
        predicted_target: Optional[float],
        actual_exit: float,
        optimal_exit: Optional[float],
    ) -> float:
        """
        Calculate target quality score.
        
        Args:
            predicted_target: Predicted target price
            actual_exit: Actual exit price
            optimal_exit: Optional optimal exit price
            
        Returns:
            Quality score (0-1)
        """
        if predicted_target is None:
            return 0.5  # Neutral if no target
        
        if optimal_exit is None:
            # Score based on whether target was realistic
            # If actual exit is close to target, target was good
            diff_pct = abs(actual_exit - predicted_target) / predicted_target
            return max(0.0, 1.0 - diff_pct * 5)
        
        # Score based on how close predicted target was to optimal
        diff_pct = abs(predicted_target - optimal_exit) / optimal_exit
        return max(0.0, 1.0 - diff_pct * 5)
    
    def _calculate_stop_loss_quality(
        self,
        predicted_stop_loss: Optional[float],
        actual_exit: float,
        optimal_exit: Optional[float],
    ) -> float:
        """
        Calculate stop-loss quality score.
        
        Args:
            predicted_stop_loss: Predicted stop-loss price
            actual_exit: Actual exit price
            optimal_exit: Optional optimal exit price
            
        Returns:
            Quality score (0-1)
        """
        if predicted_stop_loss is None:
            return 0.5  # Neutral if no stop loss
        
        # Score based on whether stop loss was appropriate
        # If actual exit is far from stop loss, stop loss was not hit (good)
        # If actual exit is close to stop loss, it might have been too tight
        diff_pct = abs(actual_exit - predicted_stop_loss) / predicted_stop_loss
        
        if diff_pct < 0.02:  # Within 2% - stop loss was hit
            return 0.3  # Low quality
        elif diff_pct < 0.05:  # Within 5% - close call
            return 0.6  # Medium quality
        else:
            return 0.9  # High quality
    
    def _calculate_probability_calibration_error(
        self,
        predicted_probability: float,
        actual_success: bool,
        historical_calibration: Optional[Dict[str, float]],
    ) -> float:
        """
        Calculate probability calibration error.
        
        Args:
            predicted_probability: Predicted probability
            actual_success: Whether prediction was successful
            historical_calibration: Historical calibration data
            
        Returns:
            Calibration error (lower is better)
        """
        # Simple calibration error: difference between predicted probability and actual outcome
        actual_outcome = 1.0 if actual_success else 0.0
        error = abs(predicted_probability - actual_outcome)
        
        return error
    
    def _calculate_confidence_calibration_score(
        self,
        predicted_confidence: str,
        actual_return: float,
        historical_calibration: Optional[Dict[str, float]],
    ) -> float:
        """
        Calculate confidence calibration score.
        
        Args:
            predicted_confidence: Predicted confidence level
            actual_return: Actual return percentage
            historical_calibration: Historical calibration data
            
        Returns:
            Calibration score (0-1)
        """
        # Map confidence to expected return range
        expected_returns = {
            "HIGH": (2.0, 10.0),  # 2-10% expected return
            "MEDIUM": (0.0, 5.0),  # 0-5% expected return
            "LOW": (-5.0, 2.0),  # -5 to 2% expected return
        }
        
        expected_min, expected_max = expected_returns.get(predicted_confidence, (0.0, 5.0))
        
        # Score based on whether actual return was in expected range
        if expected_min <= actual_return <= expected_max:
            return 1.0
        elif actual_return > expected_max:
            # Exceeded expectations - still good
            return 0.8
        else:
            # Below expectations
            diff = expected_min - actual_return
            return max(0.0, 1.0 - diff / 10.0)  # 10% below = 0 score
    
    def _classify_quality(
        self,
        actual_return: float,
        direction_correct: bool,
        entry_quality: float,
        target_quality: float,
        stop_loss_quality: float,
    ) -> PredictionQuality:
        """
        Classify overall prediction quality.
        
        Args:
            actual_return: Actual return percentage
            direction_correct: Whether direction was correct
            entry_quality: Entry quality score
            target_quality: Target quality score
            stop_loss_quality: Stop-loss quality score
            
        Returns:
            PredictionQuality
        """
        if not direction_correct:
            return PredictionQuality.FAILURE
        
        if actual_return < -2.0:
            return PredictionQuality.FAILURE
        
        # Calculate average quality score
        avg_quality = (entry_quality + target_quality + stop_loss_quality) / 3
        
        if actual_return >= 5.0 and avg_quality >= 0.8:
            return PredictionQuality.PERFECT
        elif actual_return >= 2.0 and avg_quality >= 0.6:
            return PredictionQuality.GOOD
        elif actual_return >= 0.0 and avg_quality >= 0.4:
            return PredictionQuality.NEUTRAL
        else:
            return PredictionQuality.POOR


def calculate_prediction_result(
    prediction: PredictionMetadata,
    actual_entry_price: float,
    actual_exit_price: float,
    exit_timestamp: datetime,
    optimal_entry_price: Optional[float] = None,
    optimal_exit_price: Optional[float] = None,
) -> PredictionResult:
    """
    Convenience function to calculate prediction result.
    
    Args:
        prediction: Original prediction metadata
        actual_entry_price: Actual entry price achieved
        actual_exit_price: Actual exit price
        exit_timestamp: Exit timestamp
        optimal_entry_price: Optional optimal entry price
        optimal_exit_price: Optional optimal exit price
        
    Returns:
        PredictionResult
    """
    calculator = PredictionResultCalculator()
    return calculator.calculate_result(
        prediction,
        actual_entry_price,
        actual_exit_price,
        exit_timestamp,
        optimal_entry_price,
        optimal_exit_price,
    )
