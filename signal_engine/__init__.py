"""
Signal Engine

The brain of the quant system that converts features into meaningful trading signals.

Pipeline:
Market Data → Feature Laboratory → Signal Generator → Technical Signals
                                                      → Volume Signals
                                                      → Options Signals
                                                      → Fundamental Signals
                                                      → Sentiment Signals
                                                      → Score Every Signal
                                                      → Filter Weak Signals
                                                      → Rank Stocks
                                                      → Top Candidates
                                                      → Prediction Model
"""

from signal_engine.base import (
    Signal,
    SignalCategory,
    SignalDirection,
    SignalSet,
    SignalFilterResult,
    SignalRanking,
    BaseSignalGenerator,
)
from signal_engine.generator import SignalGenerator
from signal_engine.technical import TechnicalSignalGenerator
from signal_engine.volume import VolumeSignalGenerator
from signal_engine.options import OptionsSignalGenerator
from signal_engine.fundamental import FundamentalSignalGenerator
from signal_engine.sentiment import SentimentSignalGenerator
from signal_engine.scoring import SignalScorer, ScoringConfig
from signal_engine.filtering import SignalFilter, FilterRule, MultiSignalConfirmation
from signal_engine.ranking import SignalRanker, RankingCriteria
from signal_engine.performance import SignalPerformanceTracker, SignalPerformance, TradeRecord
from signal_engine.combination import SignalCombinationTester, CombinationRule, CombinationTestResult

__all__ = [
    # Base classes
    "Signal",
    "SignalCategory",
    "SignalDirection",
    "SignalSet",
    "SignalFilterResult",
    "SignalRanking",
    "BaseSignalGenerator",
    # Main generator
    "SignalGenerator",
    # Signal generators
    "TechnicalSignalGenerator",
    "VolumeSignalGenerator",
    "OptionsSignalGenerator",
    "FundamentalSignalGenerator",
    "SentimentSignalGenerator",
    # Engines
    "SignalScorer",
    "ScoringConfig",
    "SignalFilter",
    "FilterRule",
    "MultiSignalConfirmation",
    "SignalRanker",
    "RankingCriteria",
    "SignalPerformanceTracker",
    "SignalPerformance",
    "TradeRecord",
    "SignalCombinationTester",
    "CombinationRule",
    "CombinationTestResult",
]
