"""
Prediction Pipeline Validation Module

Validates every prediction before it's stored or used.
Checks model existence, feature count/order, probability validity, entry/stop/target validity.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger("validation.prediction")


@dataclass
class PredictionValidationResult:
    """Result of validating a single prediction."""
    prediction_id: str
    passed: bool
    model_exists: bool
    feature_count_correct: bool
    feature_order_correct: bool
    probability_valid: bool
    entry_valid: bool
    stop_valid: bool
    target_valid: bool
    expected_return_valid: bool
    risk_reward_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "prediction_id": self.prediction_id,
            "passed": self.passed,
            "model_exists": self.model_exists,
            "feature_count_correct": self.feature_count_correct,
            "feature_order_correct": self.feature_order_correct,
            "probability_valid": self.probability_valid,
            "entry_valid": self.entry_valid,
            "stop_valid": self.stop_valid,
            "target_valid": self.target_valid,
            "expected_return_valid": self.expected_return_valid,
            "risk_reward_valid": self.risk_reward_valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class PredictionValidator:
    """
    Validates predictions before they're stored or used.
    
    Checks:
    1. Model exists and is loaded
    2. Feature count matches expected
    3. Feature order matches expected
    4. Probability is valid (0-1 range)
    5. Entry price is valid
    6. Stop loss is valid
    7. Target price is valid
    8. Expected return is valid
    9. Risk/reward ratio is valid
    """
    
    def __init__(
        self,
        min_probability: float = 0.0,
        max_probability: float = 1.0,
        min_risk_reward: float = 0.5,
        max_expected_return: float = 0.5,  # 50%
        min_expected_return: float = -0.2,  # -20%
    ):
        """
        Initialize validator.
        
        Args:
            min_probability: Minimum allowed probability
            max_probability: Maximum allowed probability
            min_risk_reward: Minimum allowed risk/reward ratio
            max_expected_return: Maximum allowed expected return
            min_expected_return: Minimum allowed expected return
        """
        self.min_probability = min_probability
        self.max_probability = max_probability
        self.min_risk_reward = min_risk_reward
        self.max_expected_return = max_expected_return
        self.min_expected_return = min_expected_return
    
    def validate_prediction(
        self,
        prediction: Any,
        model: Any,
        expected_feature_count: int,
        expected_feature_order: List[str],
    ) -> PredictionValidationResult:
        """
        Validate a single prediction.
        
        Args:
            prediction: Prediction object to validate
            model: Model object used for prediction
            expected_feature_count: Expected number of features
            expected_feature_order: Expected feature order
            
        Returns:
            PredictionValidationResult
        """
        errors = []
        warnings = []
        passed = True
        
        # Check 1: Model exists and is ready
        model_exists = model is not None
        if not model_exists:
            errors.append("Model does not exist or is not loaded")
            passed = False
        elif hasattr(model, 'is_ready') and not model.is_ready():
            errors.append("Model exists but is not ready")
            passed = False
            model_exists = False
        
        # Check 2: Feature count correct
        feature_count_correct = True
        if hasattr(prediction, 'features'):
            actual_count = len(prediction.features) if prediction.features else 0
            if actual_count != expected_feature_count:
                errors.append(
                    f"Feature count mismatch: expected {expected_feature_count}, "
                    f"got {actual_count}"
                )
                passed = False
                feature_count_correct = False
        
        # Check 3: Feature order correct
        feature_order_correct = True
        if hasattr(prediction, 'features') and hasattr(prediction, 'feature_names'):
            if prediction.feature_names != expected_feature_order:
                errors.append("Feature order mismatch")
                passed = False
                feature_order_correct = False
        
        # Check 4: Probability valid
        probability_valid = True
        probability = getattr(prediction, 'confidence', None) or getattr(prediction, 'win_probability', None)
        if probability is not None:
            if not (self.min_probability <= probability <= self.max_probability):
                errors.append(
                    f"Probability out of valid range: {probability:.3f} "
                    f"(expected {self.min_probability}-{self.max_probability})"
                )
                passed = False
                probability_valid = False
        else:
            warnings.append("Probability not provided")
        
        # Check 5: Entry price valid
        entry_valid = True
        entry_price = getattr(prediction, 'entry_price', None)
        if entry_price is None or entry_price <= 0:
            errors.append("Invalid entry price")
            passed = False
            entry_valid = False
        
        # Check 6: Stop loss valid
        stop_valid = True
        stop_loss = getattr(prediction, 'stop_loss', None)
        if stop_loss is None or stop_loss <= 0:
            errors.append("Invalid stop loss")
            passed = False
            stop_valid = False
        
        # Check 7: Target price valid
        target_valid = True
        target_price = getattr(prediction, 'target_price', None)
        if target_price is None or target_price <= 0:
            errors.append("Invalid target price")
            passed = False
            target_valid = False
        
        # Check 8: Expected return valid
        expected_return_valid = True
        expected_return = getattr(prediction, 'expected_return', None)
        if expected_return is not None:
            if not (self.min_expected_return <= expected_return <= self.max_expected_return):
                errors.append(
                    f"Expected return out of valid range: {expected_return:.3f} "
                    f"(expected {self.min_expected_return}-{self.max_expected_return})"
                )
                passed = False
                expected_return_valid = False
        
        # Check 9: Risk/reward ratio valid
        risk_reward_valid = True
        risk_reward = getattr(prediction, 'risk_reward_ratio', None)
        if risk_reward is not None:
            if risk_reward < self.min_risk_reward:
                errors.append(
                    f"Risk/reward ratio too low: {risk_reward:.2f} "
                    f"(minimum {self.min_risk_reward})"
                )
                passed = False
                risk_reward_valid = False
        
        # Check 10: Direction consistency
        direction = getattr(prediction, 'direction', None) or getattr(prediction, 'prediction', None)
        if direction:
            if isinstance(direction, str):
                direction = direction.upper()
            if direction in ["BUY", "SELL"]:
                if entry_price and stop_loss and target_price:
                    if direction == "BUY":
                        if target_price <= entry_price:
                            errors.append("BUY direction but target <= entry")
                            passed = False
                        if stop_loss >= entry_price:
                            errors.append("BUY direction but stop >= entry")
                            passed = False
                    elif direction == "SELL":
                        if target_price >= entry_price:
                            errors.append("SELL direction but target >= entry")
                            passed = False
                        if stop_loss <= entry_price:
                            errors.append("SELL direction but stop <= entry")
                            passed = False
        
        prediction_id = getattr(prediction, 'id', str(id(prediction)))
        
        return PredictionValidationResult(
            prediction_id=prediction_id,
            passed=passed,
            model_exists=model_exists,
            feature_count_correct=feature_count_correct,
            feature_order_correct=feature_order_correct,
            probability_valid=probability_valid,
            entry_valid=entry_valid,
            stop_valid=stop_valid,
            target_valid=target_valid,
            expected_return_valid=expected_return_valid,
            risk_reward_valid=risk_reward_valid,
            errors=errors,
            warnings=warnings,
        )
    
    def validate_batch(
        self,
        predictions: List[Any],
        model: Any,
        expected_feature_count: int,
        expected_feature_order: List[str],
    ) -> Dict[str, Any]:
        """
        Validate a batch of predictions.
        
        Args:
            predictions: List of prediction objects
            model: Model object
            expected_feature_count: Expected number of features
            expected_feature_order: Expected feature order
            
        Returns:
            Dictionary with validation summary
        """
        results = []
        total = len(predictions)
        passed = 0
        failed = 0
        
        for prediction in predictions:
            result = self.validate_prediction(
                prediction, model, expected_feature_count, expected_feature_order
            )
            results.append(result)
            if result.passed:
                passed += 1
            else:
                failed += 1
        
        logger.info(
            f"Prediction validation: {passed}/{total} passed, {failed} failed"
        )
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate_pct": round(passed / total * 100, 2) if total > 0 else 0,
            "results": [r.to_dict() for r in results],
        }


def validate_prediction(
    prediction: Any,
    model: Any,
    expected_feature_count: int,
    expected_feature_order: List[str],
    min_probability: float = 0.0,
    max_probability: float = 1.0,
) -> PredictionValidationResult:
    """
    Convenience function to validate a single prediction.
    
    Args:
        prediction: Prediction object
        model: Model object
        expected_feature_count: Expected feature count
        expected_feature_order: Expected feature order
        min_probability: Minimum allowed probability
        max_probability: Maximum allowed probability
        
    Returns:
        PredictionValidationResult
    """
    validator = PredictionValidator(
        min_probability=min_probability,
        max_probability=max_probability,
    )
    return validator.validate_prediction(
        prediction, model, expected_feature_count, expected_feature_order
    )
