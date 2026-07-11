"""
Backtesting Module

Provides comprehensive backtesting capabilities for strategy validation.
Includes historical data building, walk-forward validation, trade simulation,
and performance analysis.

Main Components:
- HistoricalDataBuilder: Build historical datasets
- WalkForwardValidator: Walk-forward validation with rolling windows
- TradeSimulator: Realistic trade simulation with costs
- PerformanceCalculator: Institutional-grade performance metrics
- RegimeAnalyzer: Performance breakdown by market regime
- ConfidenceAnalyzer: Confidence calibration analysis
- FailureAnalyzer: Failure categorization and analysis
"""

from backtesting.historical_builder import (
    HistoricalDataBuilder,
    HistoricalDataConfig,
    build_historical_dataset,
)
from backtesting.walk_forward import (
    WalkForwardValidator,
    WalkForwardConfig,
    WalkForwardWindow,
    create_walk_forward_config,
)
from backtesting.trade_simulator import (
    TradeSimulator,
    Trade,
    TradeConfig,
    ExitReason,
    simulate_backtest,
)
from backtesting.performance_metrics import (
    PerformanceCalculator,
    PerformanceMetrics,
    calculate_performance,
)
from backtesting.regime_analysis import (
    RegimeAnalyzer,
    analyze_by_regime,
)
from backtesting.confidence_analysis import (
    ConfidenceAnalyzer,
    analyze_confidence_calibration,
)
from backtesting.failure_analysis import (
    FailureAnalyzer,
    FailureReason,
    analyze_failures,
)

__all__ = [
    # Historical Data
    "HistoricalDataBuilder",
    "HistoricalDataConfig",
    "build_historical_dataset",
    # Walk-Forward Validation
    "WalkForwardValidator",
    "WalkForwardConfig",
    "WalkForwardWindow",
    "create_walk_forward_config",
    # Trade Simulation
    "TradeSimulator",
    "Trade",
    "TradeConfig",
    "ExitReason",
    "simulate_backtest",
    # Performance Metrics
    "PerformanceCalculator",
    "PerformanceMetrics",
    "calculate_performance",
    # Regime Analysis
    "RegimeAnalyzer",
    "analyze_by_regime",
    # Confidence Analysis
    "ConfidenceAnalyzer",
    "analyze_confidence_calibration",
    # Failure Analysis
    "FailureAnalyzer",
    "FailureReason",
    "analyze_failures",
]
