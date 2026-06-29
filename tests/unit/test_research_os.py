import json
import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from data_platform.processing.feature_store import FeatureStore
from portfolio_execution.event_sourcing import rebuild_positions
from prediction_intelligence.base_logistic import EnsembleModel
from research_platform.experiments.experiment_tracker import ExperimentTracker
from research_platform.simulation.market_simulator import MarketSimulator
from research_platform.strategies.strategy_config import StrategyConfig


@pytest.fixture
def setup_dirs():
    os.makedirs("test_data/features", exist_ok=True)
    os.makedirs("test_data/experiments", exist_ok=True)
    yield "test_data"
    if os.path.exists("test_data"):
        shutil.rmtree("test_data")


def test_feature_store_pit(setup_dirs):
    fs = FeatureStore(offline_dir="test_data/features")
    symbol = "TEST_STOCK"

    # Generate feature series with timestamps
    timestamps = [datetime(2026, 6, 29, 10, i) for i in range(5)]
    df = pd.DataFrame({"timestamp": timestamps, "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0]})

    fs.push_offline(symbol, df)

    # Test point-in-time retrieval (Time Travel query)
    val_at_10_02 = fs.get_feature_pit(symbol, "feature_1", datetime(2026, 6, 29, 10, 2))
    assert val_at_10_02 == 3.0

    val_at_10_04 = fs.get_feature_pit(symbol, "feature_1", datetime(2026, 6, 29, 10, 4))
    assert val_at_10_04 == 5.0

    val_before_start = fs.get_feature_pit(symbol, "feature_1", datetime(2026, 6, 29, 9, 59))
    assert val_before_start is None


def test_market_simulator():
    sim = MarketSimulator(
        spread_bps=10.0, commission_pct=0.1, max_bar_vol_pct=0.20, slippage_coef=0.05
    )

    # Test buy order
    res = sim.simulate_fill(
        side="BUY",
        target_qty=1000,
        next_bar_open=100.0,
        next_bar_high=105.0,
        next_bar_low=98.0,
        next_bar_volume=10000,
        adv=500000.0,
    )

    assert res["filled_qty"] == 1000
    assert res["status"] == "FILLED"
    assert res["execution_price"] > 100.0  # open + spread + slippage
    assert res["commission"] > 0.0


def test_event_sourcing():
    events = [
        {
            "event_type": "TRADE_EXECUTION",
            "payload": {"symbol": "INFY", "side": "BUY", "qty": 100, "price": 1500.0},
        },
        {
            "event_type": "TRADE_EXECUTION",
            "payload": {"symbol": "INFY", "side": "BUY", "qty": 50, "price": 1510.0},
        },
        {
            "event_type": "TRADE_EXECUTION",
            "payload": {"symbol": "INFY", "side": "SELL", "qty": 100, "price": 1520.0},
        },
    ]

    positions = rebuild_positions(events)

    assert "INFY" in positions
    assert positions["INFY"]["qty"] == 50
    # Average price calculation: ((100 * 1500) + (50 * 1510)) / 150 = 1503.33
    assert np.isclose(positions["INFY"]["avg_price"], 1503.333333)


def test_strategy_config(tmp_path):
    config_data = {
        "strategy_name": "orb_reversal",
        "parameters": {"rsi_length": 14, "rsi_threshold": 30},
        "risk": {"max_position_size": 500},
    }

    config_file = tmp_path / "config.json"
    with open(config_file, "w") as f:
        json.dump(config_data, f)

    sc = StrategyConfig.from_json(str(config_file))
    assert sc.get_strategy_name() == "orb_reversal"
    assert sc.get_param("rsi_length") == 14
    assert sc.get_risk_param("max_position_size") == 500


def test_ensemble_model():
    features = ["feat1", "feat2"]
    ensemble = EnsembleModel(feature_cols=features)

    # Mock data for training
    X = pd.DataFrame(np.random.rand(100, 2), columns=features)
    y = pd.Series(np.random.randint(0, 2, 100))

    ensemble.train(X, y)

    X_test = pd.DataFrame(np.random.rand(10, 2), columns=features)
    probs = ensemble.predict_proba(X_test)

    assert len(probs) == 10
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_experiment_tracker(setup_dirs):
    tracker = ExperimentTracker(root_dir="test_data")

    tracker.log_hypothesis("Volatility Arb", "Test option basis spreads", "Sharpe > 2.0")
    tracker.log_experiment("Volatility Arb", {"param1": 1}, {"sharpe": 2.1})
    tracker.log_benchmark("vol_arb", 2.1, 15.0, 4.2)

    assert os.path.exists("test_data/hypotheses/volatility_arb.json")
    assert os.path.exists("test_data/benchmarks/leaderboard.json")
