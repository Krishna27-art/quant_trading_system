"""
Automated Verification Suite for Institutional Alpha Discovery Platform

Tests database schema, experiment tracker, feature builder, and alpha prediction engine.
"""

import os
import pytest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from database.schema import init_db, get_db_connection
from evaluation.tracker.experiment_tracker import ExperimentTracker
from features.technical.feature_builder import FeatureBuilder
from prediction.predictor.alpha_engine import AlphaEngine

def test_database_initialization():
    init_db()
    with get_db_connection() as conn:
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        table_names = {t["name"] for t in tables}
        assert "experiments" in table_names
        assert "alpha_predictions" in table_names
        assert "model_registry" in table_names

def test_experiment_tracker():
    tracker = ExperimentTracker()
    exp_id = tracker.log_experiment(
        dataset_version="test_v1",
        feature_set_id="fset_test",
        model_type="Test_LGBM",
        hyperparameters={"lr": 0.01},
        oos_sharpe=3.14,
        information_coefficient_ic=0.065,
        ece_calibration_error=0.012,
        win_rate=0.72,
        top_shap_features=["ret_1", "flow_proxy"],
        notes="Pytest verification run"
    )
    assert exp_id.startswith("exp_")
    
    best = tracker.get_best_experiments(metric="oos_sharpe", limit=5)
    assert any(b["experiment_id"] == exp_id for b in best)

def test_feature_builder_no_lookahead():
    builder = FeatureBuilder(windows=[2, 3])
    data = [
        {"symbol": "TEST", "timestamp": "2026-07-01T09:15:00Z", "open": 100, "high": 105, "low": 98, "close": 102, "volume": 1000},
        {"symbol": "TEST", "timestamp": "2026-07-01T09:30:00Z", "open": 102, "high": 108, "low": 101, "close": 106, "volume": 1500},
    ]
    feats = builder.build_features(data)
    assert len(feats) == 2
    # First bar should have ret_1 = 0.0 because no prior history
    assert feats[0]["ret_1"] == 0.0
    # Second bar should have positive ret_1
    assert feats[1]["ret_1"] > 0.0

def test_alpha_engine_contract():
    engine = AlphaEngine()
    train_data = [
        {"symbol": "TEST", "timestamp": f"2026-07-01T09:{i:02d}:00Z", "open": 100+i, "high": 102+i, "low": 99+i, "close": 101+i, "volume": 1000*(i+1)}
        for i in range(10)
    ]
    labels = [1 if i % 2 == 0 else 0 for i in range(10)]
    engine.train_baseline(train_data, labels)

    live_bar = [{"symbol": "TEST", "timestamp": "2026-07-01T10:00:00Z", "open": 110, "high": 115, "low": 108, "close": 114, "volume": 5000}]
    preds = engine.generate_predictions(live_bar, store_in_db=True)
    
    assert len(preds) == 1
    contract = preds[0]
    assert "prediction_id" in contract
    assert contract["symbol"] == "TEST"
    assert contract["prediction"] in {"BUY", "SELL", "HOLD"}
    assert "reasons" in contract
    assert isinstance(contract["reasons"], list)
