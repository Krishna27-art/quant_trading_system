"""
Unit Tests for the Prediction Validation Framework
"""

import pytest
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base, Prediction
from validation.prediction_record import PredictionRecord
from validation.prediction_store import PredictionStore
from validation.validation_report import build_report
from utils.time_utils import now_ist


@pytest.fixture
def db_session():
    """Create in-memory SQLite database session for unit tests."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_prediction_record_valid():
    """Verify that a valid PredictionRecord can be constructed."""
    now = now_ist()
    record = PredictionRecord(
        symbol="RELIANCE",
        prediction_time=now,
        model_version="META_SWING_v1",
        feature_schema_version="v2.1",
        feature_snapshot={"rsi_14d": 52.5, "ma20_slope": 0.01},
        direction="BUY",
        timeframe="SWING",
        win_probability=0.62,
        confidence=0.62,
        entry_price=2450.0,
        stop_loss=2400.0,
        target_price=2550.0,
        expected_return=0.015,
        reason="Moving averages pointing upwards",
    )
    assert record.symbol == "RELIANCE"
    assert record.risk_reward_ratio() == 2.0  # (2550-2450)/(2450-2400) = 100/50 = 2.0
    assert not record.is_resolved()


def test_prediction_record_invalid_levels():
    """Verify that validation raises when stop-loss/target prices are invalid for direction."""
    now = now_ist()

    # BUY: stop-loss must be below entry
    with pytest.raises(ValueError, match="Stop-loss must be below entry for a long trade"):
        PredictionRecord(
            symbol="RELIANCE",
            prediction_time=now,
            model_version="META_SWING_v1",
            feature_schema_version="v2.1",
            feature_snapshot={"rsi": 50},
            direction="BUY",
            timeframe="SWING",
            win_probability=0.6,
            confidence=0.6,
            entry_price=2450.0,
            stop_loss=2460.0,  # Invalid: above entry
            target_price=2550.0,
        )

    # BUY: target must be above entry
    with pytest.raises(ValueError, match="Target must be above entry for a long trade"):
        PredictionRecord(
            symbol="RELIANCE",
            prediction_time=now,
            model_version="META_SWING_v1",
            feature_schema_version="v2.1",
            feature_snapshot={"rsi": 50},
            direction="BUY",
            timeframe="SWING",
            win_probability=0.6,
            confidence=0.6,
            entry_price=2450.0,
            stop_loss=2400.0,
            target_price=2440.0,  # Invalid: below entry
        )

    # SELL: stop-loss must be above entry
    with pytest.raises(ValueError, match="Stop-loss must be above entry for a short trade"):
        PredictionRecord(
            symbol="RELIANCE",
            prediction_time=now,
            model_version="META_SWING_v1",
            feature_schema_version="v2.1",
            feature_snapshot={"rsi": 50},
            direction="SELL",
            timeframe="SWING",
            win_probability=0.6,
            confidence=0.6,
            entry_price=2450.0,
            stop_loss=2400.0,  # Invalid: below entry
            target_price=2350.0,
        )


def test_prediction_record_missing_timezone():
    """Verify that naive prediction_time datetimes are rejected."""
    naive_time = datetime.now()  # no timezone info
    with pytest.raises(ValueError, match="prediction_time must be timezone-aware"):
        PredictionRecord(
            symbol="RELIANCE",
            prediction_time=naive_time,
            model_version="META_SWING_v1",
            feature_schema_version="v2.1",
            feature_snapshot={"rsi": 50},
            direction="BUY",
            timeframe="SWING",
            win_probability=0.6,
            confidence=0.6,
            entry_price=2450.0,
            stop_loss=2400.0,
            target_price=2550.0,
        )


def test_prediction_store_store_and_resolve(db_session):
    """Verify storing, preventing duplicates, and resolving predictions."""
    store = PredictionStore()
    now = now_ist()

    record = PredictionRecord(
        symbol="TCS",
        prediction_time=now,
        model_version="META_SWING_v1",
        feature_schema_version="v2.1",
        feature_snapshot={"rsi_14d": 45.0, "z_score": 1.2},
        direction="BUY",
        timeframe="SWING",
        win_probability=0.58,
        confidence=0.58,
        entry_price=3500.0,
        stop_loss=3450.0,
        target_price=3600.0,
    )

    # 1. Store the prediction
    pred_id = store.store(record, db_session)
    db_session.commit()
    assert pred_id is not None

    # Check database content
    db_pred = db_session.query(Prediction).filter(Prediction.id == pred_id).one()
    assert db_pred.symbol == "TCS"
    assert db_pred.actual_outcome == "OPEN"
    assert db_pred.feature_schema_version == "v2.1"

    # 2. Try to store duplicate prediction — should return None (skip, not raise)
    dup_id = store.store(record, db_session)
    assert dup_id is None

    # 3. Resolve the prediction
    exit_time = now + timedelta(days=2)
    resolved = store.resolve(
        pred_id,
        outcome_fields={
            "actual_outcome": "WIN",
            "exit_price": 3600.0,
            "exit_time": exit_time,
            "actual_return": 0.0286,
            "mfe": 0.031,
            "mae": 0.005,
            "hold_bars": 3,
            "target_hit": True,
            "stop_hit": False,
            "is_correct": True,
        },
        db=db_session,
    )
    db_session.commit()
    assert resolved is True

    # Check updated database content
    db_pred_updated = db_session.query(Prediction).filter(Prediction.id == pred_id).one()
    assert db_pred_updated.actual_outcome == "WIN"
    assert db_pred_updated.hold_bars == 3
    assert float(db_pred_updated.actual_return) == 0.0286
    assert db_pred_updated.exit_time == exit_time.replace(tzinfo=None)



def test_validation_report_generation(db_session):
    """Verify that build_report constructs a complete, structured report."""
    store = PredictionStore()
    now = now_ist()

    # Seed 3 resolved predictions (1 WIN, 1 LOSS, 1 TIMEOUT)
    preds_to_seed = [
        # Win
        ("RELIANCE", "BUY", 2450.0, 2400.0, 2550.0, 0.65, "WIN", 2550.0, 0.0408, 0.045, 0.002, 4),
        # Loss
        ("TCS", "BUY", 3500.0, 3450.0, 3600.0, 0.60, "LOSS", 3450.0, -0.0143, 0.005, 0.015, 2),
        # Timeout
        ("INFY", "SELL", 1500.0, 1530.0, 1440.0, 0.58, "TIMEOUT", 1485.0, 0.0100, 0.015, 0.005, 10),
    ]

    for symbol, direction, entry, sl, tp, prob, outcome, exit_p, ret, mfe, mae, hold_b in preds_to_seed:
        record = PredictionRecord(
            symbol=symbol,
            prediction_time=now,
            model_version="META_SWING_v1",
            feature_schema_version="v2.1",
            feature_snapshot={"rsi_14d": 50.0},
            direction=direction,
            timeframe="SWING",
            win_probability=prob,
            confidence=prob,
            entry_price=entry,
            stop_loss=sl,
            target_price=tp,
        )
        pred_id = store.store(record, db_session)
        store.resolve(
            pred_id,
            outcome_fields={
                "actual_outcome": outcome,
                "exit_price": exit_p,
                "exit_time": now + timedelta(days=hold_b),
                "actual_return": ret,
                "mfe": mfe,
                "mae": mae,
                "hold_bars": hold_b,
                "target_hit": outcome == "WIN",
                "stop_hit": outcome == "LOSS",
                "is_correct": outcome == "WIN",
            },
            db=db_session,
        )

    db_session.commit()

    # Generate the report
    report = build_report(db_session)

    # Assert sections and key metrics are computed correctly
    assert "overview" in report
    overview = report["overview"]
    assert overview["total_resolved"] == 3
    assert overview["wins"] == 1
    assert overview["losses"] == 1
    assert overview["timeouts"] == 1
    assert overview["win_rate_pct"] == 33.33  # 1/3

    assert "mfe_mae" in report
    assert "winners" in report["mfe_mae"]
    assert "losers" in report["mfe_mae"]

    assert "duration" in report
    assert report["duration"]["wins"]["mean_bars"] == 4.0
    assert report["duration"]["losses"]["mean_bars"] == 2.0
    assert report["duration"]["all"]["mean_bars"] == 5.3  # (4 + 2 + 10) / 3 = 5.33

    assert "barriers" in report
    assert report["barriers"]["target_hit"] == 1
    assert report["barriers"]["sl_hit"] == 1
    assert report["barriers"]["timeout"] == 1

    assert "feature_audit" in report
    assert report["feature_audit"]["snapshot_coverage_pct"] == 100.0
