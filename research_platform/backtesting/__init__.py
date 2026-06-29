"""
Backtesting Engine

Institutional-grade backtesting engine with support for transaction costs,
slippage, market impact, and multiple asset classes.
"""

from .benchmarking import BenchmarkComparator
from .cross_validation import BacktestCrossValidator, CrossValidationMethod
from .engine import BacktestConfig, BacktestingEngine, BacktestResult, Portfolio, Position
from .performance_metrics import PerformanceMetrics
from .results_analysis import ResultsAnalyzer
from .risk_metrics import RiskMetrics
from .transaction_costs import TransactionCostCalculator, TransactionCostModel

__all__ = [
    "BacktestingEngine",
    "BacktestConfig",
    "BacktestResult",
    "Position",
    "Portfolio",
    "PerformanceMetrics",
    "ResultsAnalyzer",
    "RiskMetrics",
    "BenchmarkComparator",
    "BacktestCrossValidator",
    "CrossValidationMethod",
    "TransactionCostCalculator",
    "TransactionCostModel",
]
