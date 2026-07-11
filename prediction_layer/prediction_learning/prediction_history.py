"""
Prediction History

Stores prediction metadata for learning and analysis.
Every prediction is stored with its features, signals, and market context.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum
import json

from utils.logger import get_logger

logger = get_logger("prediction_layer.prediction_learning.prediction_history")


class PredictionAction(Enum):
    """Prediction action enumeration."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class PredictionStatus(Enum):
    """Prediction status enumeration."""
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


@dataclass
class PredictionMetadata:
    """Metadata for a prediction."""
    prediction_id: str
    symbol: str
    action: PredictionAction
    entry_price: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    probability: float
    confidence: str
    expected_return: float
    prediction_timestamp: datetime
    features: Dict[str, float]
    signals: List[Dict[str, any]]
    market_regime: str
    model_name: str
    timeframe: str
    status: PredictionStatus = PredictionStatus.PENDING
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "prediction_id": self.prediction_id,
            "symbol": self.symbol,
            "action": self.action.value,
            "entry_price": self.entry_price,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "probability": round(self.probability, 4),
            "confidence": self.confidence,
            "expected_return": round(self.expected_return, 4),
            "prediction_timestamp": self.prediction_timestamp.isoformat(),
            "features": {k: round(v, 4) for k, v in self.features.items()},
            "signals": self.signals,
            "market_regime": self.market_regime,
            "model_name": self.model_name,
            "timeframe": self.timeframe,
            "status": self.status.value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "PredictionMetadata":
        """Create from dictionary."""
        return cls(
            prediction_id=data["prediction_id"],
            symbol=data["symbol"],
            action=PredictionAction(data["action"]),
            entry_price=data["entry_price"],
            target_price=data.get("target_price"),
            stop_loss=data.get("stop_loss"),
            probability=data["probability"],
            confidence=data["confidence"],
            expected_return=data["expected_return"],
            prediction_timestamp=datetime.fromisoformat(data["prediction_timestamp"]),
            features=data["features"],
            signals=data["signals"],
            market_regime=data["market_regime"],
            model_name=data["model_name"],
            timeframe=data["timeframe"],
            status=PredictionStatus(data.get("status", "PENDING")),
        )


class PredictionHistory:
    """
    Stores and manages prediction history.
    
    This is the foundation of the learning engine - all predictions
    are stored here for later analysis and learning.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize prediction history.
        
        Args:
            storage_path: Optional path to store prediction history
        """
        self.storage_path = storage_path
        self._predictions: Dict[str, PredictionMetadata] = {}
        self._logger = get_logger("prediction_layer.prediction_learning.prediction_history")
    
    def store_prediction(self, metadata: PredictionMetadata) -> bool:
        """
        Store a prediction in history.
        
        Args:
            metadata: PredictionMetadata object
            
        Returns:
            True if stored successfully
        """
        self._predictions[metadata.prediction_id] = metadata
        
        self._logger.info(
            f"Stored prediction {metadata.prediction_id} "
            f"for {metadata.symbol} ({metadata.action.value})"
        )
        
        return True
    
    def get_prediction(self, prediction_id: str) -> Optional[PredictionMetadata]:
        """
        Get a prediction by ID.
        
        Args:
            prediction_id: Prediction ID
            
        Returns:
            PredictionMetadata if found, None otherwise
        """
        return self._predictions.get(prediction_id)
    
    def update_prediction_status(
        self,
        prediction_id: str,
        status: PredictionStatus,
    ) -> bool:
        """
        Update prediction status.
        
        Args:
            prediction_id: Prediction ID
            status: New status
            
        Returns:
            True if updated successfully
        """
        if prediction_id not in self._predictions:
            self._logger.warning(f"Prediction {prediction_id} not found")
            return False
        
        self._predictions[prediction_id].status = status
        
        self._logger.info(
            f"Updated prediction {prediction_id} status to {status.value}"
        )
        
        return True
    
    def get_predictions_by_symbol(
        self,
        symbol: str,
    ) -> List[PredictionMetadata]:
        """
        Get all predictions for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of PredictionMetadata objects
        """
        return [
            pred for pred in self._predictions.values()
            if pred.symbol == symbol
        ]
    
    def get_predictions_by_status(
        self,
        status: PredictionStatus,
    ) -> List[PredictionMetadata]:
        """
        Get all predictions with a specific status.
        
        Args:
            status: Prediction status
            
        Returns:
            List of PredictionMetadata objects
        """
        return [
            pred for pred in self._predictions.values()
            if pred.status == status
        ]
    
    def get_predictions_by_regime(
        self,
        regime: str,
    ) -> List[PredictionMetadata]:
        """
        Get all predictions made in a specific regime.
        
        Args:
            regime: Market regime
            
        Returns:
            List of PredictionMetadata objects
        """
        return [
            pred for pred in self._predictions.values()
            if pred.market_regime == regime
        ]
    
    def get_predictions_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[PredictionMetadata]:
        """
        Get predictions within a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of PredictionMetadata objects
        """
        return [
            pred for pred in self._predictions.values()
            if start_date <= pred.prediction_timestamp <= end_date
        ]
    
    def get_all_predictions(self) -> List[PredictionMetadata]:
        """
        Get all predictions.
        
        Returns:
            List of all PredictionMetadata objects
        """
        return list(self._predictions.values())
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about prediction history.
        
        Returns:
            Dictionary with statistics
        """
        total_predictions = len(self._predictions)
        
        if total_predictions == 0:
            return {
                "total_predictions": 0,
                "by_status": {},
                "by_action": {},
                "by_regime": {},
                "by_symbol": {},
            }
        
        # Count by status
        by_status = {}
        for pred in self._predictions.values():
            status = pred.status.value
            by_status[status] = by_status.get(status, 0) + 1
        
        # Count by action
        by_action = {}
        for pred in self._predictions.values():
            action = pred.action.value
            by_action[action] = by_action.get(action, 0) + 1
        
        # Count by regime
        by_regime = {}
        for pred in self._predictions.values():
            regime = pred.market_regime
            by_regime[regime] = by_regime.get(regime, 0) + 1
        
        # Count by symbol
        by_symbol = {}
        for pred in self._predictions.values():
            symbol = pred.symbol
            by_symbol[symbol] = by_symbol.get(symbol, 0) + 1
        
        return {
            "total_predictions": total_predictions,
            "by_status": by_status,
            "by_action": by_action,
            "by_regime": by_regime,
            "by_symbol": by_symbol,
        }
    
    def save_to_file(self, filepath: Optional[str] = None) -> bool:
        """
        Save prediction history to file.
        
        Args:
            filepath: Optional filepath (uses storage_path if not provided)
            
        Returns:
            True if saved successfully
        """
        filepath = filepath or self.storage_path
        
        if not filepath:
            self._logger.error("No filepath provided for saving")
            return False
        
        try:
            data = {
                "predictions": [
                    pred.to_dict() for pred in self._predictions.values()
                ],
                "saved_at": datetime.now().isoformat(),
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            self._logger.info(f"Saved prediction history to {filepath}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to save prediction history: {e}")
            return False
    
    def load_from_file(self, filepath: Optional[str] = None) -> bool:
        """
        Load prediction history from file.
        
        Args:
            filepath: Optional filepath (uses storage_path if not provided)
            
        Returns:
            True if loaded successfully
        """
        filepath = filepath or self.storage_path
        
        if not filepath:
            self._logger.error("No filepath provided for loading")
            return False
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self._predictions.clear()
            
            for pred_data in data.get("predictions", []):
                metadata = PredictionMetadata.from_dict(pred_data)
                self._predictions[metadata.prediction_id] = metadata
            
            self._logger.info(f"Loaded prediction history from {filepath}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to load prediction history: {e}")
            return False


def create_prediction_metadata(
    prediction_id: str,
    symbol: str,
    action: str,
    entry_price: float,
    target_price: Optional[float],
    stop_loss: Optional[float],
    probability: float,
    confidence: str,
    expected_return: float,
    features: Dict[str, float],
    signals: List[Dict[str, any]],
    market_regime: str,
    model_name: str,
    timeframe: str,
) -> PredictionMetadata:
    """
    Convenience function to create prediction metadata.
    
    Args:
        prediction_id: Prediction ID
        symbol: Stock symbol
        action: Trading action
        entry_price: Entry price
        target_price: Target price
        stop_loss: Stop loss price
        probability: Predicted probability
        confidence: Confidence level
        expected_return: Expected return
        features: Feature dictionary
        signals: Signal list
        market_regime: Market regime
        model_name: Model name
        timeframe: Timeframe
        
    Returns:
        PredictionMetadata
    """
    return PredictionMetadata(
        prediction_id=prediction_id,
        symbol=symbol,
        action=PredictionAction(action),
        entry_price=entry_price,
        target_price=target_price,
        stop_loss=stop_loss,
        probability=probability,
        confidence=confidence,
        expected_return=expected_return,
        prediction_timestamp=datetime.now(),
        features=features,
        signals=signals,
        market_regime=market_regime,
        model_name=model_name,
        timeframe=timeframe,
    )
