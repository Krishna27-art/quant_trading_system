"""
Prediction API Endpoints

Exposes prediction-related endpoints for the prediction-only system.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime
import numpy as np

from prediction_layer.models.model_registry import registry
from prediction_layer.signal_generator.signal_generator import SignalGenerator
from prediction_layer.signal_generator.ranking_engine import SignalRankingEngine
from evaluation_layer.prediction_tracker.prediction_tracker import PredictionTracker
from evaluation_layer.accuracy.prediction_evaluator import PredictionEvaluator
from feature_layer.feature_store.feature_store import FeatureStore

router = APIRouter(prefix="/api/predictions")

# Initialize components
signal_generator = SignalGenerator()
ranking_engine = SignalRankingEngine()
prediction_tracker = PredictionTracker()
prediction_evaluator = PredictionEvaluator()
feature_store = FeatureStore()


@router.get("/generate/{symbol}")
def generate_prediction(
    symbol: str,
    model_name: str = Query(default="ensemble", description="Model to use"),
    timeframe: str = Query(default="INTRADAY", description="Prediction timeframe")
):
    """
    Generate a prediction for a single symbol.
    
    Returns complete trading signal with entry, stop loss, targets, probability, and reasons.
    """
    try:
        # Load model
        model = registry.load_model(model_name)
        
        # Get current price and features
        from data_platform.upstox_client import get_stock_quote
        quote = get_stock_quote(symbol.upper())
        current_price = float(quote.get("last_price", 0))
        
        if current_price == 0:
            raise HTTPException(status_code=404, detail=f"Could not fetch price for {symbol}")
        
        # Get latest features
        features = feature_store.get_latest_features(symbol)
        
        if not features:
            raise HTTPException(status_code=404, detail=f"No features available for {symbol}")
        
        # Prepare feature vector
        feature_names = list(features.keys())
        feature_vector = np.array([features[name] for name in feature_names]).reshape(1, -1)
        
        # Get prediction
        prediction = model.predict(feature_vector)[0]
        probability = model.predict_proba(feature_vector)[0].max()
        
        # Get feature importance
        feature_importance = model.get_feature_importance()
        
        # Generate signal
        signal = signal_generator.generate_signal(
            symbol=symbol.upper(),
            current_price=current_price,
            prediction=prediction,
            probability=probability,
            feature_importance=feature_importance,
            model_version=model.metadata.model_version if model.metadata else "v1.0.0",
            feature_version=model.metadata.feature_version if model.metadata else "v1.0.0",
            dataset_version=model.metadata.dataset_version if model.metadata else "v1.0.0"
        )
        
        # Track prediction
        from evaluation_layer.prediction_tracker.prediction_tracker import Prediction, PredictionOutcome, SignalDirection
        prediction_record = Prediction(
            prediction_id=str(datetime.now().timestamp()),
            symbol=signal.symbol,
            direction=SignalDirection(signal.direction.value),
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            target_1=signal.target_1,
            target_2=signal.target_2,
            probability=signal.probability,
            confidence=signal.confidence,
            expected_return_pct=signal.expected_return_pct,
            expected_holding_days=signal.expected_holding_days,
            worst_case_pct=signal.worst_case_pct,
            best_case_pct=signal.best_case_pct,
            win_probability=signal.win_probability,
            model_version=signal.model_version,
            feature_version=signal.feature_version,
            dataset_version=signal.dataset_version,
            reasons=signal.reasons,
            feature_importance=signal.feature_importance,
            created_at=datetime.now()
        )
        prediction_tracker.create_prediction(prediction_record)
        
        return {
            "symbol": signal.symbol,
            "direction": signal.direction.value,
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "target_1": signal.target_1,
            "target_2": signal.target_2,
            "expected_return_pct": signal.expected_return_pct,
            "expected_holding_days": signal.expected_holding_days,
            "probability": signal.probability,
            "confidence": signal.confidence,
            "worst_case_pct": signal.worst_case_pct,
            "best_case_pct": signal.best_case_pct,
            "win_probability": signal.win_probability,
            "reasons": signal.reasons,
            "quality_scores": {
                "overall_score": signal.overall_score,
                "trend_score": signal.trend_score,
                "momentum_score": signal.momentum_score,
                "liquidity_score": signal.liquidity_score,
                "volume_score": signal.volume_score,
                "options_score": signal.options_score,
                "news_score": signal.news_score,
                "macro_score": signal.macro_score,
                "sector_score": signal.sector_score,
                "institutional_score": signal.institutional_score
            },
            "versioning": {
                "model_version": signal.model_version,
                "feature_version": signal.feature_version,
                "dataset_version": signal.dataset_version
            },
            "timestamp": signal.timestamp.isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/top-signals")
def get_top_signals(
    direction: str = Query(default="BUY", description="BUY or SELL"),
    top_n: int = Query(default=10, description="Number of top signals")
):
    """
    Get top N ranked signals for the specified direction.
    """
    try:
        # Get all pending predictions
        pending = prediction_tracker.get_pending_predictions()
        
        # Convert to signals
        signals = []
        for pred in pending:
            from prediction_layer.signal_generator.signal_generator import TradingSignal, SignalDirection
            signal = TradingSignal(
                symbol=pred.symbol,
                direction=SignalDirection(pred.direction.value),
                entry_price=pred.entry_price,
                stop_loss=pred.stop_loss,
                target_1=pred.target_1,
                target_2=pred.target_2,
                expected_return_pct=pred.expected_return_pct,
                expected_holding_days=pred.expected_holding_days,
                probability=pred.probability,
                confidence=pred.confidence,
                worst_case_pct=pred.worst_case_pct,
                best_case_pct=pred.best_case_pct,
                win_probability=pred.win_probability,
                reasons=pred.reasons,
                feature_importance=pred.feature_importance,
                trend_score=0,  # Would need to be stored
                momentum_score=0,
                liquidity_score=0,
                volume_score=0,
                options_score=0,
                news_score=0,
                macro_score=0,
                sector_score=0,
                institutional_score=0,
                overall_score=pred.probability * 100,  # Simplified
                model_version=pred.model_version,
                feature_version=pred.feature_version,
                dataset_version=pred.dataset_version,
                timestamp=pred.created_at
            )
            signals.append(signal)
        
        # Rank signals
        if direction.upper() == "BUY":
            ranked = ranking_engine.get_top_buy_signals(signals, top_n)
        else:
            ranked = ranking_engine.get_top_sell_signals(signals, top_n)
        
        return {
            "signals": [
                {
                    "rank": r.rank,
                    "score": r.score,
                    "symbol": r.signal.symbol,
                    "direction": r.signal.direction.value,
                    "entry_price": r.signal.entry_price,
                    "stop_loss": r.signal.stop_loss,
                    "target_1": r.signal.target_1,
                    "expected_return_pct": r.signal.expected_return_pct,
                    "probability": r.signal.probability,
                    "confidence": r.signal.confidence,
                    "reasons": r.signal.reasons
                }
                for r in ranked
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluation")
def get_evaluation_metrics(
    min_confidence: Optional[float] = Query(default=None, description="Minimum confidence threshold")
):
    """
    Get prediction evaluation metrics.
    
    Returns accuracy, precision, recall, F1, calibration metrics, etc.
    """
    try:
        completed = prediction_tracker.get_completed_predictions(limit=1000)
        
        metrics = prediction_evaluator.evaluate_predictions(completed, min_confidence)
        
        return {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "brier_score": metrics.brier_score,
            "expected_calibration_error": metrics.expected_calibration_error,
            "roc_auc": metrics.roc_auc,
            "confusion_matrix": {
                "true_positives": metrics.true_positives,
                "true_negatives": metrics.true_negatives,
                "false_positives": metrics.false_positives,
                "false_negatives": metrics.false_negatives
            },
            "performance": {
                "avg_expected_return": metrics.avg_expected_return,
                "avg_actual_return": metrics.avg_actual_return,
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "avg_holding_days": metrics.avg_holding_days
            },
            "sample_size": {
                "total_predictions": metrics.total_predictions,
                "completed_predictions": metrics.completed_predictions
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/calibration")
def get_calibration_curve():
    """
    Get calibration curve data for reliability diagram.
    """
    try:
        completed = prediction_tracker.get_completed_predictions(limit=1000)
        prob_pred, prob_true = prediction_evaluator.get_calibration_curve(completed)
        
        return {
            "mean_predicted_values": prob_pred,
            "fraction_of_positives": prob_true
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
def list_models():
    """List all available models and their versions."""
    try:
        return registry.list_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
