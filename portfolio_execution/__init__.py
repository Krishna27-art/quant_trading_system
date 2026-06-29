"""
Trading Engine - Central Orchestrator Package

The brain of the quant platform. Wires together:
data_platform → features → alpha → risk → portfolio → execution
"""

from portfolio_execution.config import ExecutionMode, TradingConfig
from portfolio_execution.ems import ExecutionManagementSystem
from portfolio_execution.oms import ManagedOrder, OrderManagementSystem
from portfolio_execution.state_manager import MarketState, SessionStateManager

__all__ = [
    "TradingConfig",
    "ExecutionMode",
    "SessionStateManager",
    "MarketState",
    "OrderManagementSystem",
    "ManagedOrder",
    "ExecutionManagementSystem",
]
