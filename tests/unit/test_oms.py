"""
Unit Tests for the new Order Management System (OMS)
"""

import pytest

from portfolio_execution.oms import (
    OrderManagementSystem,
    OrderSide,
    OrderStatus,
    OrderType,
)


@pytest.fixture
def oms():
    """Create a default OMS instance for testing."""
    return OrderManagementSystem(
        max_daily_loss_pct=0.02,
        max_order_value_inr=1000000.0,  # ₹10 Lakhs
        max_order_pct_of_adv=0.10,
        max_price_deviation_atr=3.0,
        max_trades_per_day=5,
        max_single_position_pct=0.20,
        max_sector_exposure_pct=0.30,
        nav=5000000.0,  # ₹50 Lakhs NAV
    )


def test_order_creation_and_state_transitions(oms):
    # Create order
    order = oms.create_order(
        symbol="RELIANCE",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100,
        price=2400.0,
        client_order_id="client_id_1",
    )

    assert order.status == OrderStatus.VALIDATED
    assert order.quantity == 100
    assert order.price == 2400.0

    # Send order to broker
    order.update_status(OrderStatus.SENT, "Sent to broker")
    assert order.status == OrderStatus.SENT
    assert order.sent_at is not None

    # Partial fill
    oms.on_fill(order.order_id, 40, 2405.0)
    assert order.status == OrderStatus.PARTIAL_FILL
    assert order.filled_quantity == 40
    assert order.remaining_quantity == 60
    assert order.average_fill_price == 2405.0

    # Full fill
    oms.on_fill(order.order_id, 60, 2410.0)
    assert order.status == OrderStatus.FILLED
    assert order.filled_quantity == 100
    assert order.remaining_quantity == 0
    # Weighted average: (40 * 2405 + 60 * 2410) / 100 = 2408.0
    assert order.average_fill_price == 2408.0
    assert order.filled_at is not None


def test_deduplication(oms):
    order1 = oms.create_order(
        symbol="RELIANCE",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=10,
        price=2400.0,
        client_order_id="dup_client_id_1",
    )
    assert order1.status == OrderStatus.VALIDATED

    # Attempt to submit duplicate
    order2 = oms.create_order(
        symbol="RELIANCE",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=10,
        price=2400.0,
        client_order_id="dup_client_id_1",
    )
    assert order2.status == OrderStatus.REJECTED
    assert "Duplicate order" in order2.rejection_reason


def test_restricted_list(oms):
    oms.set_restricted_symbols({"TCS", "INFY"})

    order = oms.create_order(
        symbol="TCS",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=10,
        price=3500.0,
        client_order_id="client_id_2",
    )
    assert order.status == OrderStatus.REJECTED
    assert "restricted list" in order.rejection_reason


def test_fat_finger_limit_inr(oms):
    # Order value = 500 * 2500 = 1,250,000 > 1,000,000 limit
    order = oms.create_order(
        symbol="RELIANCE",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=500,
        price=2500.0,
        client_order_id="client_id_3",
    )
    assert order.status == OrderStatus.REJECTED
    assert "exceeds cap" in order.rejection_reason


def test_fat_finger_price_deviation(oms):
    # Test ATR deviation > 3.0×
    order = oms.create_order(
        symbol="RELIANCE",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=10,
        price=2550.0,  # last_price = 2400, ATR = 30 -> deviation = (2550 - 2400) / 30 = 5.0× ATR
        client_order_id="client_id_4",
        last_price=2400.0,
        atr=30.0,
    )
    assert order.status == OrderStatus.REJECTED
    assert "PRICE" in order.rejection_reason


def test_adv_participation_adjustment(oms):
    # ADV = 10,000, max percentage = 10% -> max quantity = 1000
    order = oms.create_order(
        symbol="RELIANCE",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=1500,
        price=500.0,  # Value = 750,000 (below 1,000,000)
        client_order_id="client_id_5",
        adv=10000.0,
    )
    # Passed but quantity is adjusted down to 1000
    assert order.status == OrderStatus.VALIDATED
    assert order.quantity == 1000
    assert order.remaining_quantity == 1000


def test_trading_halt(oms):
    oms.halt_trading("High volatility circuit breaker tripped")

    order = oms.create_order(
        symbol="RELIANCE",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=10,
        price=2400.0,
        client_order_id="client_id_6",
    )
    assert order.status == OrderStatus.REJECTED
    assert "Trading halted" in order.rejection_reason


def test_concentration_and_sector_limits(oms):
    # Max position value = 5,000,000 * 0.10 = 500,000 (we'll set limit_value to 0.10 inside the test or use a larger order value below 1M)
    # Let's adjust oms single position limit to 10% to make it 500,000
    oms._max_single_position_pct = 0.10
    # Order value = 300 * 2000 = 600,000 > 500,000
    order = oms.create_order(
        symbol="RELIANCE",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=300,
        price=2000.0,
        client_order_id="client_id_7",
    )
    assert order.status == OrderStatus.REJECTED
    assert "exceeding 10% of NAV" in order.rejection_reason

    # Reset single position limit to 20% (1,000,000) to allow the TCS order (900k) to pass single-position checks
    oms._max_single_position_pct = 0.20

    # Set sector mapping
    oms.set_sector_map({"TCS": "IT", "INFY": "IT"})

    # Max sector IT value = 5,000,000 * 0.30 = 1,500,000
    # First place valid IT order
    order_it = oms.create_order(
        symbol="TCS",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=300,
        price=3000.0,  # Value = 900,000
        client_order_id="client_id_it_1",
    )
    assert order_it.status == OrderStatus.VALIDATED
    # Fill it to create exposure
    oms.on_fill(order_it.order_id, 300, 3000.0)

    # Now attempt another IT order that pushes the sector exposure to 1,800,000 (over 1.5M limit)
    order_it_2 = oms.create_order(
        symbol="INFY",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=600,
        price=1500.0,  # Value = 900,000
        client_order_id="client_id_it_2",
    )
    assert order_it_2.status == OrderStatus.REJECTED
    assert "Sector 'IT' exposure would be" in order_it_2.rejection_reason


def test_position_tracking_after_fills(oms):
    order = oms.create_order(
        symbol="RELIANCE",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100,
        price=2400.0,
        client_order_id="pos_client_id_1",
    )
    oms.on_fill(order.order_id, 100, 2400.0)

    pos = oms.positions["RELIANCE"]
    assert pos.quantity == 100
    assert pos.average_cost == 2400.0

    # Sell half
    order_sell = oms.create_order(
        symbol="RELIANCE",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=50,
        price=2500.0,
        client_order_id="pos_client_id_2",
    )
    oms.on_fill(order_sell.order_id, 50, 2500.0)
    assert pos.quantity == 50
    assert pos.average_cost == 2400.0
    # Realized PnL: (2500 - 2400) * 50 = 5000.0
    assert pos.realized_pnl == 5000.0
