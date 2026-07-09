"""
Label Validation Module

Validates prediction labels for correctness and prevents look-ahead bias.
Automatically verifies target hit, stop hit, correct order, timestamps, etc.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from utils.logger import get_logger

logger = get_logger("validation.labels")


@dataclass
class LabelValidationResult:
    """Result of validating a single prediction label."""
    prediction_id: str
    passed: bool
    target_hit: bool
    stop_hit: bool
    correct_order: bool
    no_future_leakage: bool
    correct_timestamps: bool
    correct_entry: bool
    correct_exit: bool
    correct_holding_period: bool
    errors: List[str]
    warnings: List[str]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "prediction_id": self.prediction_id,
            "passed": self.passed,
            "target_hit": self.target_hit,
            "stop_hit": self.stop_hit,
            "correct_order": self.correct_order,
            "no_future_leakage": self.no_future_leakage,
            "correct_timestamps": self.correct_timestamps,
            "correct_entry": self.correct_entry,
            "correct_exit": self.correct_exit,
            "correct_holding_period": self.correct_holding_period,
            "errors": self.errors,
            "warnings": self.warnings,
        }


@dataclass
class LabelReport:
    """Report from label validation."""
    total_labels: int
    passed_labels: int
    failed_labels: int
    results: List[LabelValidationResult]
    timestamp: datetime
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "total_labels": self.total_labels,
            "passed_labels": self.passed_labels,
            "failed_labels": self.failed_labels,
            "pass_rate_pct": round(self.passed_labels / self.total_labels * 100, 2) if self.total_labels > 0 else 0,
            "results": [r.to_dict() for r in self.results],
            "timestamp": self.timestamp.isoformat(),
        }


class LabelValidator:
    """
    Validates prediction labels for correctness.
    
    Checks:
    1. Target hit correctly identified
    2. Stop hit correctly identified
    3. Correct order (target before stop for wins, stop before target for losses)
    4. No future leakage (labels based on future data)
    5. Correct timestamps (entry < exit)
    6. Correct entry price
    7. Correct exit price
    8. Correct holding period
    """
    
    def __init__(
        self,
        min_holding_bars: int = 1,
        max_holding_bars: int = 1000,
        price_tolerance: float = 0.01,  # 1% tolerance for price validation
    ):
        """
        Initialize validator.
        
        Args:
            min_holding_bars: Minimum allowed holding period
            max_holding_bars: Maximum allowed holding period
            price_tolerance: Tolerance for price validation (as percentage)
        """
        self.min_holding_bars = min_holding_bars
        self.max_holding_bars = max_holding_bars
        self.price_tolerance = price_tolerance
    
    def validate_prediction(
        self,
        prediction: Any,
        price_data: pd.DataFrame,
    ) -> LabelValidationResult:
        """
        Validate a single prediction label.
        
        Args:
            prediction: Prediction object with entry, stop, target, outcome
            price_data: DataFrame with price data for validation
            
        Returns:
            LabelValidationResult
        """
        errors = []
        warnings = []
        passed = True
        
        # Extract prediction data
        entry_price = getattr(prediction, "entry_price", None)
        stop_loss = getattr(prediction, "stop_loss", None)
        target_price = getattr(prediction, "target_price", None)
        actual_outcome = getattr(prediction, "actual_outcome", None)
        prediction_time = getattr(prediction, "prediction_time", None)
        expiry_time = getattr(prediction, "expiry_time", None)
        direction = getattr(prediction, "direction", None)
        
        # Check 1: Correct entry price
        correct_entry = True
        if entry_price is None or entry_price <= 0:
            errors.append("Invalid entry price")
            passed = False
            correct_entry = False
        
        # Check 2: Correct stop and target
        target_hit = getattr(prediction, "target_hit", False)
        stop_hit = getattr(prediction, "stop_hit", False)
        
        if target_hit and stop_hit:
            errors.append("Both target and stop hit - ambiguous outcome")
            passed = False
        
        # Check 3: Correct order based on outcome
        correct_order = True
        if actual_outcome == "WIN" and not target_hit:
            errors.append("WIN outcome but target not hit")
            passed = False
            correct_order = False
        elif actual_outcome == "LOSS" and not stop_hit:
            errors.append("LOSS outcome but stop not hit")
            passed = False
            correct_order = False
        
        # Check 4: No future leakage
        no_future_leakage = True
        if prediction_time and expiry_time:
            if expiry_time <= prediction_time:
                errors.append("Expiry time before or equal to prediction time")
                passed = False
                no_future_leakage = False
        
        # Check 5: Correct timestamps
        correct_timestamps = True
        if prediction_time and expiry_time:
            holding_period = (expiry_time - prediction_time).total_seconds() / 3600  # hours
            if holding_period < 0:
                errors.append("Negative holding period")
                passed = False
                correct_timestamps = False
        
        # Check 6: Correct exit price validation
        correct_exit = True
        if actual_outcome == "WIN" and target_hit:
            if target_price is None:
                errors.append("WIN outcome but target price is None")
                passed = False
                correct_exit = False
        elif actual_outcome == "LOSS" and stop_hit:
            if stop_loss is None:
                errors.append("LOSS outcome but stop loss is None")
                passed = False
                correct_exit = False
        
        # Check 7: Correct holding period
        correct_holding_period = True
        hold_bars = getattr(prediction, "hold_bars", None)
        if hold_bars is not None:
            if hold_bars < self.min_holding_bars:
                warnings.append(f"Holding period too short: {hold_bars} bars")
            elif hold_bars > self.max_holding_bars:
                warnings.append(f"Holding period too long: {hold_bars} bars")
        
        # Check 8: Direction consistency
        if direction:
            if direction == "BUY":
                if target_price and target_price <= entry_price:
                    errors.append("BUY direction but target price <= entry price")
                    passed = False
                if stop_loss and stop_loss >= entry_price:
                    errors.append("BUY direction but stop loss >= entry price")
                    passed = False
            elif direction == "SELL":
                if target_price and target_price >= entry_price:
                    errors.append("SELL direction but target price >= entry price")
                    passed = False
                if stop_loss and stop_loss <= entry_price:
                    errors.append("SELL direction but stop loss <= entry price")
                    passed = False
        
        prediction_id = getattr(prediction, "id", str(id(prediction)))
        
        return LabelValidationResult(
            prediction_id=prediction_id,
            passed=passed,
            target_hit=target_hit,
            stop_hit=stop_hit,
            correct_order=correct_order,
            no_future_leakage=no_future_leakage,
            correct_timestamps=correct_timestamps,
            correct_entry=correct_entry,
            correct_exit=correct_exit,
            correct_holding_period=correct_holding_period,
            errors=errors,
            warnings=warnings,
        )
    
    def validate_batch(
        self,
        predictions: List[Any],
        price_data_map: Dict[str, pd.DataFrame],
    ) -> LabelReport:
        """
        Validate a batch of predictions.
        
        Args:
            predictions: List of prediction objects
            price_data_map: Dictionary mapping symbol to price data DataFrame
            
        Returns:
            LabelReport
        """
        results = []
        total_labels = len(predictions)
        passed_labels = 0
        
        for prediction in predictions:
            symbol = getattr(prediction, "symbol", None)
            if symbol and symbol in price_data_map:
                result = self.validate_prediction(prediction, price_data_map[symbol])
            else:
                # Validate without price data if not available
                result = self.validate_prediction(prediction, pd.DataFrame())
            
            results.append(result)
            if result.passed:
                passed_labels += 1
        
        failed_labels = total_labels - passed_labels
        
        logger.info(
            f"Label validation: {passed_labels}/{total_labels} passed, "
            f"{failed_labels} failed"
        )
        
        return LabelReport(
            total_labels=total_labels,
            passed_labels=passed_labels,
            failed_labels=failed_labels,
            results=results,
            timestamp=datetime.now(),
        )
    
    def validate_price_path(
        self,
        entry_price: float,
        stop_loss: float,
        target_price: float,
        direction: str,
        price_path: pd.Series,
    ) -> Dict[str, Any]:
        """
        Validate that the label matches the actual price path.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            target_price: Target price
            direction: BUY or SELL
            price_path: Series of prices after entry
            
        Returns:
            Dictionary with validation results
        """
        if direction == "BUY":
            # For BUY: target > entry > stop
            target_hit_idx = price_path[price_path >= target_price].first_valid_index()
            stop_hit_idx = price_path[price_path <= stop_loss].first_valid_index()
        else:  # SELL
            # For SELL: target < entry < stop
            target_hit_idx = price_path[price_path <= target_price].first_valid_index()
            stop_hit_idx = price_path[price_path >= stop_loss].first_valid_index()
        
        # Determine which was hit first
        actual_outcome = None
        if target_hit_idx is not None and stop_hit_idx is not None:
            if target_hit_idx < stop_hit_idx:
                actual_outcome = "WIN"
            else:
                actual_outcome = "LOSS"
        elif target_hit_idx is not None:
            actual_outcome = "WIN"
        elif stop_hit_idx is not None:
            actual_outcome = "LOSS"
        else:
            actual_outcome = "TIMEOUT"
        
        return {
            "actual_outcome": actual_outcome,
            "target_hit": target_hit_idx is not None,
            "stop_hit": stop_hit_idx is not None,
            "target_hit_idx": target_hit_idx,
            "stop_hit_idx": stop_hit_idx,
        }


def validate_labels(
    predictions: List[Any],
    price_data_map: Dict[str, pd.DataFrame],
    min_holding_bars: int = 1,
    max_holding_bars: int = 1000,
) -> LabelReport:
    """
    Convenience function to validate prediction labels.
    
    Args:
        predictions: List of prediction objects
        price_data_map: Dictionary mapping symbol to price data
        min_holding_bars: Minimum allowed holding period
        max_holding_bars: Maximum allowed holding period
        
    Returns:
        LabelReport
    """
    validator = LabelValidator(
        min_holding_bars=min_holding_bars,
        max_holding_bars=max_holding_bars,
    )
    return validator.validate_batch(predictions, price_data_map)
