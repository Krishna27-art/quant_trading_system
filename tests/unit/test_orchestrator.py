"""
Unit Tests for Central Trading Orchestrator
"""

from datetime import datetime

import pytest

from portfolio_execution.config import ExecutionMode, TradingConfig
from portfolio_execution.orchestrator import TickData, TradingOrchestrator


@pytest.fixture
def orchestrator_config():
    """Create basic trading config for tests."""
    return TradingConfig(
        mode=ExecutionMode.BACKTEST,
        instruments=["RELIANCE", "TCS"],
        log_level="INFO",
    )


@pytest.fixture
def orchestrator(orchestrator_config):
    """Create orchestrator instance."""
    import signal
    from unittest.mock import patch

    orig_signal = signal.signal
    try:
        signal.signal = lambda sig, handler: None
        with patch("portfolio_execution.orchestrator.RedisStateStore"):
            orchestrator = TradingOrchestrator(orchestrator_config)
    finally:
        signal.signal = orig_signal
    return orchestrator


def test_orchestrator_initialization(orchestrator):
    assert orchestrator.config.mode == ExecutionMode.BACKTEST
    assert "RELIANCE" in orchestrator.config.instruments
    assert "TCS" in orchestrator.config.instruments
    assert orchestrator.oms is not None
    assert orchestrator.ems is not None


def test_session_initialization(orchestrator):
    previous_day_data = {
        "RELIANCE": {"high": 2500.0, "low": 2400.0, "close": 2450.0},
        "TCS": {"high": 3600.0, "low": 3500.0, "close": 3550.0},
    }

    orchestrator.initialize_session(previous_day_data)
    assert len(orchestrator._state_managers) == 2

    reliance_state = orchestrator._state_managers["RELIANCE"]
    assert reliance_state.levels.pdh == 2500.0
    assert reliance_state.levels.pdl == 2400.0
    assert reliance_state.levels.pdc == 2450.0


def test_tick_processing(orchestrator):
    previous_day_data = {
        "RELIANCE": {"high": 2500.0, "low": 2400.0, "close": 2450.0},
        "TCS": {"high": 3600.0, "low": 3500.0, "close": 3550.0},
    }
    orchestrator.initialize_session(previous_day_data)

    # Process first tick
    tick = TickData(
        symbol="RELIANCE",
        timestamp=datetime.now(),
        open=2460.0,
        high=2470.0,
        low=2455.0,
        close=2465.0,
        volume=1000.0,
    )

    created_orders = orchestrator.on_tick(tick)
    assert orchestrator._tick_count == 1
    assert len(created_orders) == 0  # No alpha models registered to generate signals

    # Verify session state updated
    reliance_state = orchestrator._state_managers["RELIANCE"]
    # Check that high/low are updated
    assert reliance_state.levels.day_high == 2470.0
    assert reliance_state.levels.day_low == 2455.0
