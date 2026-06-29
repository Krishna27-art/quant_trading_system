"""
Unit Tests for Unified Execution Interface and Pre-Trade Compliance Checks
"""

import datetime
import uuid
from typing import Any

import pytest

from portfolio_execution.execution.unified_execution import (
    ExecutionMode,
    LiveExecutionEngine,
    Order,
    OrderSide,
    OrderState,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from risk_governance.pre_trade.pre_trade_checks import PreTradeConfig


class MockMarketDataSource:
    """Mock market data source for testing."""

    def __init__(self):
        self.prices = {"RELIANCE": 2400.0, "TCS": 3500.0}
        self.vols = {"RELIANCE": 0.01, "TCS": 0.015}
        self.spreads = {"RELIANCE": 0.10, "TCS": 0.15}

    def get_market_price(self, symbol: str, timestamp: datetime.datetime) -> float | None:
        return self.prices.get(symbol)

    def get_order_book(self, symbol: str, timestamp: datetime.datetime) -> dict[str, Any] | None:
        return None

    def get_volatility(self, symbol: str, timestamp: datetime.datetime) -> float | None:
        return self.vols.get(symbol)

    def get_spread(self, symbol: str, timestamp: datetime.datetime) -> float | None:
        return self.spreads.get(symbol)


class MockBrokerAdapter:
    """Mock broker adapter for testing submission."""

    def __init__(self):
        self.submitted_orders = []
        self.success = True
        self.broker_order_id = "broker_12345"

    def adapt_order_for_broker(self, order_dict: dict) -> dict:
        return order_dict

    def submit_order(self, adapted_order: dict) -> dict:
        self.submitted_orders.append(adapted_order)
        return {
            "success": self.success,
            "broker_order_id": self.broker_order_id,
            "fill_price": adapted_order.get("limit_price") or 2400.0,
        }

    def query_order_status(self, client_order_id: str) -> dict:
        return {"exists": False}


@pytest.fixture
def data_source():
    return MockMarketDataSource()


@pytest.fixture
def broker_adapter():
    return MockBrokerAdapter()


@pytest.fixture
def execution_engine(data_source, broker_adapter):
    # Setup checker config with very tight fat finger cap (₹5 Lakhs cap)
    config = PreTradeConfig(max_order_notional=500000.0, max_qty_pct_of_adv=0.10)
    return LiveExecutionEngine(
        mode=ExecutionMode.LIVE,
        data_source=data_source,
        broker_adapter=broker_adapter,
        pre_trade_config=config,
    )


def test_place_order_success(execution_engine, broker_adapter):
    order = Order(
        order_id=str(uuid.uuid4()),
        symbol="RELIANCE",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.LIMIT,
        limit_price=2400.0,
        time_in_force=TimeInForce.DAY,
        client_order_id="client_id_success_1",
    )

    fills = execution_engine.execute_order(order)

    assert len(fills) == 1
    assert order.status == OrderStatus.FILLED
    assert order.state == OrderState.FILLED
    assert len(broker_adapter.submitted_orders) == 1


def test_place_order_pre_trade_restricted(execution_engine):
    # Set TCS as restricted symbol in pre-trade checker
    execution_engine.pre_trade_checker.load_restricted_list(["TCS"])

    order = Order(
        order_id=str(uuid.uuid4()),
        symbol="TCS",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.LIMIT,
        limit_price=3500.0,
        time_in_force=TimeInForce.DAY,
        client_order_id="client_id_restricted",
    )

    fills = execution_engine.execute_order(order)

    assert len(fills) == 0
    assert order.status == OrderStatus.REJECTED


def test_place_order_pre_trade_notional_cap(execution_engine):
    # Try order with notional = 1000 * 2400 = 2,400,000 > ₹5 Lakhs cap
    order = Order(
        order_id=str(uuid.uuid4()),
        symbol="RELIANCE",
        side=OrderSide.BUY,
        quantity=1000,
        order_type=OrderType.LIMIT,
        limit_price=2400.0,
        time_in_force=TimeInForce.DAY,
        client_order_id="client_id_notional_cap",
    )

    fills = execution_engine.execute_order(order)

    assert len(fills) == 0
    assert order.status == OrderStatus.REJECTED


def test_place_order_deduplication(execution_engine):
    order1 = Order(
        order_id=str(uuid.uuid4()),
        symbol="RELIANCE",
        side=OrderSide.BUY,
        quantity=5,
        order_type=OrderType.LIMIT,
        limit_price=2400.0,
        time_in_force=TimeInForce.DAY,
        client_order_id="dup_client_id",
    )

    fills1 = execution_engine.execute_order(order1)
    assert len(fills1) == 1
    assert order1.status == OrderStatus.FILLED

    # Send another order with the same client_order_id
    order2 = Order(
        order_id=str(uuid.uuid4()),
        symbol="RELIANCE",
        side=OrderSide.BUY,
        quantity=5,
        order_type=OrderType.LIMIT,
        limit_price=2400.0,
        time_in_force=TimeInForce.DAY,
        client_order_id="dup_client_id",
    )

    # Since order1 is in terminal state (FILLED), reuse of client_order_id for retry should pass
    # Let's verify by manually setting order1's state to ACKNOWLEDGED to check duplication blocking
    order1.state = OrderState.ACKNOWLEDGED

    fills2 = execution_engine.execute_order(order2)
    assert len(fills2) == 0
    assert order2.status == OrderStatus.REJECTED
