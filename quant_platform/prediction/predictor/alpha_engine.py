"""
Canonical Institutional Alpha Engine

The core predictive orchestration pipeline.
Takes raw market data, computes features, evaluates ML probability models,
applies risk-reward sizing (target/stop loss), generates SHAP reasons,
and logs the canonical Alpha Prediction contract to the database.
"""

import uuid
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from features.technical.feature_builder import FeatureBuilder
from models.lightgbm.lgbm_predictor import LGBMPredictor
from prediction.explanation.shap_explainer import SHAPExplainer
from database.schema import get_db_connection, init_db

logger = logging.getLogger("AlphaEngine")

class AlphaEngine:
    def __init__(self, model: Optional[Any] = None, windows: List[int] = [3, 5, 15]):
        init_db()
        self.feature_builder = FeatureBuilder(windows=windows)
        self.model = model if model else LGBMPredictor()
        self.explainer = None

    def train_baseline(self, historical_ohlcv: List[Dict[str, Any]], labels: List[int]) -> None:
        """
        Trains the underlying model on historical OHLCV data.
        """
        features = self.feature_builder.build_features(historical_ohlcv)
        feature_names = self.feature_builder.get_feature_names()
        self.model.fit(features, labels, feature_names)
        self.explainer = SHAPExplainer(feature_names)
        self._register_model_in_db()
        logger.info("✅ Alpha Engine baseline training complete.")

    def _register_model_in_db(self) -> None:
        try:
            now_str = datetime.now(timezone.utc).isoformat()
            model_id = getattr(self.model, "model_id", "default_alpha_model")
            name = getattr(self.model, "name", "LGBM Alpha")
            ver = getattr(self.model, "version", "1.0")
            with get_db_connection() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO model_registry (
                        model_id, name, version, architecture, target_horizon, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (model_id, name, ver, "GradientBoostedTrees", "Intraday_15M", "PRODUCTION", now_str)
                )
        except Exception as e:
            logger.error(f"Failed to register model in DB: {e}")

    def generate_predictions(self, market_data: List[Dict[str, Any]], store_in_db: bool = True) -> List[Dict[str, Any]]:
        """
        Generates canonical institutional prediction contracts from raw OHLCV bars.
        """
        if not self.model.is_fitted:
            raise RuntimeError("Alpha Engine model must be trained/fitted before inference.")

        features = self.feature_builder.build_features(market_data)
        if not features:
            return []

        if not self.explainer:
            self.explainer = SHAPExplainer(self.model.feature_names)

        probs = self.model.predict_proba(features)
        importances = self.model.get_feature_importance()

        predictions = []
        now_str = datetime.now(timezone.utc).isoformat()

        # We generate predictions for the latest bar of each symbol
        symbol_latest_idx: Dict[str, int] = {}
        for idx, bar in enumerate(features):
            symbol_latest_idx[bar["symbol"]] = idx

        for sym, idx in symbol_latest_idx.items():
            feat = features[idx]
            prob = probs[idx]
            close = float(feat["close"])
            ret_1 = float(feat.get("ret_1", 0.0))
            vol_spread = abs(ret_1) if abs(ret_1) > 0.005 else 0.015

            # Direction decision based on calibrated probability threshold
            if prob >= 0.55:
                direction = "BUY"
                target_price = round(close * (1.0 + vol_spread * 2.0), 2)
                stop_loss = round(close * (1.0 - vol_spread * 1.0), 2)
                exp_ret = round((target_price - close) / close * 100.0, 2)
            elif prob <= 0.45:
                direction = "SELL"
                target_price = round(close * (1.0 - vol_spread * 2.0), 2)
                stop_loss = round(close * (1.0 + vol_spread * 1.0), 2)
                exp_ret = round((close - target_price) / close * 100.0, 2)
            else:
                direction = "HOLD"
                target_price = close
                stop_loss = close
                exp_ret = 0.0

            # Confidence score combines probability distance from 0.5 and volume ratio
            vol_ratio = float(feat.get("vol_ratio_3", 1.0))
            confidence = min(0.99, round(abs(prob - 0.5) * 2.0 * min(1.5, vol_ratio), 2))

            reasons = self.explainer.explain_prediction(feat, importances, top_k=3)
            pred_id = f"pred_{uuid.uuid4().hex[:8]}"

            contract = {
                "prediction_id": pred_id,
                "timestamp": feat.get("timestamp", now_str),
                "symbol": sym,
                "prediction": direction,
                "entry_price": close,
                "target_price": target_price,
                "stop_loss": stop_loss,
                "calibrated_probability": round(prob, 4),
                "confidence_score": confidence,
                "expected_return_pct": exp_ret,
                "reasons": reasons,
                "model_id": getattr(self.model, "model_id", "default_alpha_model")
            }
            predictions.append(contract)

            if store_in_db and direction != "HOLD":
                self._store_prediction(contract)

        return predictions

    def _store_prediction(self, contract: Dict[str, Any]) -> None:
        try:
            with get_db_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO alpha_predictions (
                        prediction_id, timestamp, symbol, prediction,
                        entry_price, target_price, stop_loss, calibrated_probability,
                        confidence_score, expected_return_pct, reasons, model_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        contract["prediction_id"],
                        contract["timestamp"],
                        contract["symbol"],
                        contract["prediction"],
                        contract["entry_price"],
                        contract["target_price"],
                        contract["stop_loss"],
                        contract["calibrated_probability"],
                        contract["confidence_score"],
                        contract["expected_return_pct"],
                        json.dumps(contract["reasons"]),
                        contract["model_id"]
                    )
                )
        except Exception as e:
            logger.error(f"Failed to store prediction in DB: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = AlphaEngine()
    
    # Generate sample historical data for training
    train_data = [
        {"symbol": "RELIANCE", "timestamp": "2026-07-01T09:15:00Z", "open": 2400, "high": 2410, "low": 2395, "close": 2405, "volume": 10000},
        {"symbol": "RELIANCE", "timestamp": "2026-07-01T09:30:00Z", "open": 2405, "high": 2425, "low": 2402, "close": 2420, "volume": 15000},
        {"symbol": "TCS", "timestamp": "2026-07-01T09:15:00Z", "open": 3500, "high": 3510, "low": 3490, "close": 3495, "volume": 8000},
        {"symbol": "TCS", "timestamp": "2026-07-01T09:30:00Z", "open": 3495, "high": 3500, "low": 3480, "close": 3485, "volume": 12000},
    ]
    labels = [1, 1, 0, 0]
    engine.train_baseline(train_data, labels)

    # Generate live predictions for new bars
    live_bars = [
        {"symbol": "RELIANCE", "timestamp": "2026-07-01T09:45:00Z", "open": 2420, "high": 2445, "low": 2418, "close": 2440, "volume": 25000},
        {"symbol": "TCS", "timestamp": "2026-07-01T09:45:00Z", "open": 3485, "high": 3488, "low": 3460, "close": 3465, "volume": 18000},
    ]
    preds = engine.generate_predictions(live_bars)
    print("\n--- CANONICAL INSTITUTIONAL ALPHA CONTRACTS ---")
    for p in preds:
        print(json.dumps(p, indent=2))
