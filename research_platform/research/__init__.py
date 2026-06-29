"""
Research Module

Nautilus integration for backtesting, simulation, and event-driven architecture.
Validates slippage, latency, queue position, and fill assumptions before production.
"""

from .backtest.execution_simulator import ExecutionSimulator
from .backtest.fill_validator import FillValidator
from .backtest.latency_model import LatencyModel
from .backtest.nautilus_engine import NautilusBacktestEngine
from .backtest.queue_model import QueuePositionModel
from .backtest.slippage_model import SlippageModel

__all__ = [
    "ExecutionSimulator",
    "NautilusBacktestEngine",
    "SlippageModel",
    "LatencyModel",
    "QueuePositionModel",
    "FillValidator",
]
