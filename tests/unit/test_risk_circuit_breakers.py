"""
Unit Tests for Portfolio-Level Circuit Breaker Manager
"""

import pytest

from risk_governance.pre_trade.portfolio_drawdown_limits import (
    CircuitAction,
    CircuitBreakerConfig,
    CircuitBreakerManager,
)


@pytest.fixture
def manager():
    """Create a default CircuitBreakerManager instance for testing."""
    config = CircuitBreakerConfig(
        daily_loss_limit_pct=-0.02,
        daily_reduce_threshold_pct=-0.015,
        weekly_loss_limit_pct=-0.05,
        weekly_reduce_threshold_pct=-0.035,
        per_position_loss_limit_pct=-0.10,
        per_position_reduce_pct=-0.07,
        sector_concentration_limit_pct=0.30,
        sector_reduce_threshold_pct=0.25,
        cooldown_minutes=15,
    )
    mgr = CircuitBreakerManager(config)
    mgr.set_start_of_day_nav(10_000_000.0)  # ₹1 Crore SOD NAV
    mgr.set_start_of_week_nav(10_000_000.0)
    return mgr


def test_circuit_breaker_continue_state(manager):
    # Update P&L within safe bounds
    # Daily loss: -1% (NAV = 9,900,000) -> Should continue
    result = manager.evaluate(
        current_nav=9_900_000.0,
        position_values={"RELIANCE": 1_500_000.0},  # 15% concentration
        sector_map={"RELIANCE": "Energy"},
    )

    assert result.action == CircuitAction.CONTINUE
    assert result.reduce_factor == 1.0
    assert len(result.breaches) == 0


def test_circuit_breaker_reduce_state(manager):
    # Daily loss: -1.6% (NAV = 9,840,000) -> Should trigger REDUCE (daily_reduce_threshold_pct is -1.5%)
    result = manager.evaluate(
        current_nav=9_840_000.0,
        position_values={"RELIANCE": 1_000_000.0},
        sector_map={"RELIANCE": "Energy"},
    )

    assert result.action == CircuitAction.REDUCE
    # Reduce factor should scale down
    assert result.reduce_factor < 1.0
    assert any("daily" in b.lower() for b in result.breaches)


def test_circuit_breaker_halt_state(manager):
    # Daily loss: -2.1% (NAV = 9,790,000) -> Should trigger HALT (daily_loss_limit_pct is -2.0%)
    result = manager.evaluate(
        current_nav=9_790_000.0,
        position_values={"RELIANCE": 1_000_000.0},
        sector_map={"RELIANCE": "Energy"},
    )

    assert result.action == CircuitAction.HALT
    assert result.reduce_factor == 0.0
    assert any("daily" in b.lower() for b in result.breaches)


def test_sector_concentration_breach(manager):
    # Sector "Energy" concentration = 3.2M / 10M = 32% (limit is 30%)
    result = manager.evaluate(
        current_nav=10_000_000.0,
        position_values={"RELIANCE": 3_200_000.0},
        sector_map={"RELIANCE": "Energy"},
    )

    assert result.action == CircuitAction.HALT
    assert any("sector" in b.lower() for b in result.breaches)
