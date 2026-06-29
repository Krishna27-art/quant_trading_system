import time
from enum import Enum


class EventType(Enum):
    TICK = "TICK"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"


class Event:
    """Base class for all events."""

    def __init__(self, type: EventType):
        self.type = type
        self.timestamp = time.time()


class TickEvent(Event):
    """Event triggered by new market data."""

    def __init__(
        self,
        symbol: str,
        price: float,
        volume: int = 0,
        bid: float | None = None,
        ask: float | None = None,
    ):
        super().__init__(EventType.TICK)
        self.symbol = symbol
        self.price = price
        self.volume = volume
        self.bid = bid
        self.ask = ask


class SignalEvent(Event):
    """Event triggered by strategy generating a signal."""

    def __init__(self, strategy_id: str, symbol: str, direction: str, strength: float = 1.0):
        super().__init__(EventType.SIGNAL)
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.direction = direction  # "LONG" or "SHORT"
        self.strength = strength


class OrderEvent(Event):
    """Event triggered to place an order with a broker."""

    def __init__(
        self,
        symbol: str,
        order_type: str,
        quantity: int,
        direction: str,
        price: float | None = None,
    ):
        super().__init__(EventType.ORDER)
        self.symbol = symbol
        self.order_type = order_type  # "MKT" or "LMT"
        self.quantity = quantity
        self.direction = direction  # "BUY" or "SELL"
        self.price = price


class FillEvent(Event):
    """Event triggered when an order is filled by the broker."""

    def __init__(
        self,
        order_id: str,
        symbol: str,
        exchange: str,
        quantity: int,
        direction: str,
        fill_price: float,
        commission: float = 0.0,
    ):
        super().__init__(EventType.FILL)
        self.order_id = order_id
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = quantity
        self.direction = direction
        self.fill_price = fill_price
        self.commission = commission
