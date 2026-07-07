"""
tests/unit/test_postmortem.py
=============================
Tests the daily post-mortem generation and DB serialization.
"""

from datetime import datetime, date
import json
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base, Prediction, ModelPostmortem
from validation.daily_postmortem import run_daily_postmortem


@pytest.fixture(name="db_session")
def fixture_db_session():
    """Create in-memory SQLite database session for unit testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_daily_postmortem_empty_db(db_session):
    """Test run_daily_postmortem when there are no resolved predictions."""
    res_id = run_daily_postmortem(db_session, target_date=date(2026, 7, 7))
    assert res_id is None


def test_daily_postmortem_generation(db_session):
    """Test run_daily_postmortem compiles stats and stores postmortem correctly."""
    target_dt = date(2026, 7, 7)
    
    # 1. Insert winning trade
    win_pred = Prediction(
        id="pred-1",
        symbol="TCS",
        prediction="BUY",
        horizon="INTRADAY",
        confidence=0.75,
        entry_price=4000.0,
        stop_loss=3960.0,
        target_price=4080.0,
        actual_outcome="WIN",
        actual_return=0.02,
        is_correct=True,
        feature_snapshot=json.dumps({"rsi_14m": 42.0, "vwap_dist": 0.005}),
        exit_time=datetime(2026, 7, 7, 15, 30),
        generated_at=datetime(2026, 7, 7, 9, 30),
    )
    
    # 2. Insert losing trade
    loss_pred = Prediction(
        id="pred-2",
        symbol="RELIANCE",
        prediction="BUY",
        horizon="INTRADAY",
        confidence=0.65,
        entry_price=2500.0,
        stop_loss=2475.0,
        target_price=2550.0,
        actual_outcome="LOSS",
        actual_return=-0.01,
        is_correct=False,
        feature_snapshot=json.dumps({"rsi_14m": 72.0, "vwap_dist": -0.008}),
        exit_time=datetime(2026, 7, 7, 11, 45),
        generated_at=datetime(2026, 7, 7, 9, 45),
    )
    
    db_session.add(win_pred)
    db_session.add(loss_pred)
    db_session.commit()
    
    # Run post-mortem
    postmortem_id = run_daily_postmortem(db_session, target_date=target_dt)
    assert postmortem_id is not None
    
    # Verify DB insertion and columns
    record = db_session.query(ModelPostmortem).filter(ModelPostmortem.id == postmortem_id).first()
    assert record is not None
    assert record.date == target_dt
    assert record.total_trades == 2
    assert record.total_losses == 1
    assert pytest.approx(record.win_rate) == 0.5
    
    # Validate analysis JSON structure
    analysis = json.loads(record.analysis_json)
    assert "losing_factors" in analysis
    assert "winning_factors" in analysis
    assert "analysis" in analysis
    assert "actionable_warnings" in analysis
    assert "suggested_threshold_adjustments" in analysis
