"""
Market Regime Detection Module

Provides rule-based market regime classification for Indian markets.
Detects 6 regime types: Strong Bull, Bull, Sideways, Bear, High Volatility, Event Day.

Main Components:
- RegimeFeatureEngine: Computes market-level features
- RegimeRuleEngine: Applies rule-based classification
- MarketRegimeEngine: Main orchestration engine
- RegimeHistoryManager: Storage and retrieval of regime history

Usage:
    from regime import get_regime_engine
    
    engine = get_regime_engine()
    classification = engine.detect_regime()
    print(f"Current regime: {classification.regime.value} (confidence: {classification.confidence}%)")
"""

from regime.market_regime import MarketRegimeEngine, get_regime_engine
from regime.regime_features import RegimeFeatures, RegimeFeatureEngine
from regime.regime_history import RegimeHistoryManager
from regime.regime_rules import RegimeClassification, RegimeRuleEngine, RegimeType

__all__ = [
    "MarketRegimeEngine",
    "get_regime_engine",
    "RegimeFeatures",
    "RegimeFeatureEngine",
    "RegimeHistoryManager",
    "RegimeClassification",
    "RegimeRuleEngine",
    "RegimeType",
]
