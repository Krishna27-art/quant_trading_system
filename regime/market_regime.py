"""
Market Regime Detection Engine

Main engine that orchestrates regime detection:
1. Computes market features
2. Applies rule-based classification
3. Stores regime history
4. Provides regime-aware filtering for signals

This is the primary interface for the regime detection system.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from regime.regime_features import RegimeFeatures, RegimeFeatureEngine
from regime.regime_rules import RegimeClassification, RegimeRuleEngine, RegimeType
from regime.regime_history import RegimeHistoryManager
from utils.logger import get_logger

logger = get_logger("market_regime")


class MarketRegimeEngine:
    """
    Main engine for market regime detection.
    
    Orchestrates feature computation, rule application, and history storage.
    Provides regime-aware signal filtering.
    """
    
    def __init__(self, lookback_days: int = 200):
        """
        Initialize the market regime engine.
        
        Args:
            lookback_days: Number of days for historical feature computation
        """
        self.feature_engine = RegimeFeatureEngine(lookback_days=lookback_days)
        self.rule_engine = RegimeRuleEngine()
        self.history_manager = RegimeHistoryManager()
        self.logger = logger
        
    def detect_regime(self, asof_date: date | None = None) -> RegimeClassification:
        """
        Detect current market regime.
        
        Args:
            asof_date: Date to detect regime for (defaults to today)
            
        Returns:
            RegimeClassification with regime type and confidence
        """
        if asof_date is None:
            asof_date = date.today()
        
        self.logger.info(f"Detecting market regime for {asof_date}")
        
        # Step 1: Compute features
        features = self.feature_engine.compute_all_features(asof_date)
        
        # Step 2: Apply rules
        classification = self.rule_engine.classify(features, asof_date)
        
        # Step 3: Store in history
        try:
            self.history_manager.save_regime(
                date=asof_date,
                classification=classification,
                features=features
            )
            self.logger.info(f"Regime saved to history: {classification.regime.value}")
        except Exception as e:
            self.logger.warning(f"Failed to save regime to history: {e}")
        
        return classification
    
    def get_current_regime(self) -> RegimeClassification:
        """
        Get the most recent regime classification.
        
        Returns:
            RegimeClassification for the latest available date
        """
        return self.history_manager.get_latest_regime()
    
    def get_regime_for_date(self, asof_date: date) -> RegimeClassification | None:
        """
        Get regime classification for a specific date.
        
        Args:
            asof_date: Date to query
            
        Returns:
            RegimeClassification if available, None otherwise
        """
        return self.history_manager.get_regime_for_date(asof_date)
    
    def get_regime_history(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[RegimeClassification]:
        """
        Get regime history for a date range.
        
        Args:
            start_date: Start date (defaults to 30 days ago)
            end_date: End date (defaults to today)
            
        Returns:
            List of RegimeClassification objects
        """
        return self.history_manager.get_regime_history(start_date, end_date)
    
    def filter_signal_by_regime(
        self,
        signal: dict[str, Any],
        signal_type: str,
    ) -> dict[str, Any]:
        """
        Filter or modify a signal based on current regime.
        
        Different strategies perform differently in different regimes.
        This function applies regime-aware filtering.
        
        Args:
            signal: Original signal dict
            signal_type: Type of signal (e.g., 'breakout', 'momentum', 'mean_reversion')
            
        Returns:
            Filtered/modified signal with regime metadata
        """
        current_regime = self.get_current_regime()
        
        # Add regime context to signal
        signal['regime'] = {
            'type': current_regime.regime.value,
            'confidence': current_regime.confidence,
            'timestamp': current_regime.timestamp.isoformat(),
        }
        
        # Apply regime-specific filtering
        signal['regime_filtered'] = self._should_filter_signal(
            signal_type,
            current_regime.regime,
            current_regime.confidence
        )
        
        # Adjust signal confidence based on regime
        if not signal['regime_filtered']:
            signal['adjusted_confidence'] = signal.get('confidence', 0) * self._get_regime_adjustment_factor(
                signal_type,
                current_regime.regime
            )
        else:
            signal['adjusted_confidence'] = 0
        
        # Add explanation
        signal['regime_explanation'] = self._get_regime_explanation(
            signal_type,
            current_regime.regime,
            signal['regime_filtered']
        )
        
        return signal
    
    def _should_filter_signal(
        self,
        signal_type: str,
        regime: RegimeType,
        confidence: float,
    ) -> bool:
        """
        Determine if a signal should be filtered based on regime.
        
        Returns True if signal should be rejected.
        """
        # High volatility - filter most signals except volatility-based strategies
        if regime == RegimeType.HIGH_VOLATILITY:
            if signal_type in ['breakout', 'momentum', 'trend_following']:
                return True  # Reject breakouts in high volatility
        
        # Event day - filter all signals
        if regime == RegimeType.EVENT_DAY:
            return True
        
        # Bear market - filter bullish strategies
        if regime == RegimeType.BEAR:
            if signal_type in ['breakout', 'momentum', 'trend_following']:
                return True  # Bullish strategies fail in bear markets
        
        # Sideways - filter trend-following strategies
        if regime == RegimeType.SIDEWAYS:
            if signal_type in ['trend_following', 'breakout']:
                return True  # Trend strategies fail in sideways markets
        
        # Low confidence regime - be conservative
        if confidence < 60:
            return True
        
        return False
    
    def _get_regime_adjustment_factor(
        self,
        signal_type: str,
        regime: RegimeType,
    ) -> float:
        """
        Get confidence adjustment factor based on regime compatibility.
        
        Returns multiplier between 0.0 and 1.5
        """
        # Strong Bull - boost bullish strategies
        if regime == RegimeType.STRONG_BULL:
            if signal_type in ['breakout', 'momentum', 'trend_following']:
                return 1.3
            elif signal_type == 'mean_reversion':
                return 0.7  # Mean reversion fails in strong trends
        
        # Bull - moderate boost for bullish strategies
        if regime == RegimeType.BULL:
            if signal_type in ['breakout', 'momentum']:
                return 1.15
            elif signal_type == 'mean_reversion':
                return 0.85
        
        # Sideways - boost mean reversion
        if regime == RegimeType.SIDEWAYS:
            if signal_type == 'mean_reversion':
                return 1.25
            elif signal_type in ['trend_following', 'breakout']:
                return 0.5
        
        # Bear - boost bearish strategies
        if regime == RegimeType.BEAR:
            if signal_type in ['short', 'bearish_momentum']:
                return 1.2
            elif signal_type in ['breakout', 'momentum']:
                return 0.4
        
        # High Volatility - reduce all signals
        if regime == RegimeType.HIGH_VOLATILITY:
            return 0.6
        
        return 1.0  # No adjustment
    
    def _get_regime_explanation(
        self,
        signal_type: str,
        regime: RegimeType,
        filtered: bool,
    ) -> str:
        """Generate explanation for regime-based filtering."""
        if filtered:
            if regime == RegimeType.HIGH_VOLATILITY:
                return f"Signal rejected: {signal_type} strategies perform poorly in high volatility regimes"
            elif regime == RegimeType.EVENT_DAY:
                return f"Signal rejected: Event day with abnormal price movement"
            elif regime == RegimeType.BEAR:
                return f"Signal rejected: {signal_type} is a bullish strategy, market is in bear regime"
            elif regime == RegimeType.SIDEWAYS:
                return f"Signal rejected: {signal_type} requires trending market, current regime is sideways"
            else:
                return f"Signal rejected: Low regime confidence or unfavorable conditions"
        else:
            return f"Signal accepted: {signal_type} compatible with {regime.value} regime"
    
    def get_regime_summary(self) -> dict[str, Any]:
        """
        Get a summary of current regime for dashboard display.
        
        Returns:
            Dict with regime information suitable for UI
        """
        classification = self.get_current_regime()
        
        return {
            "regime": classification.regime.value,
            "confidence": classification.confidence,
            "trend_strength": classification.trend_strength,
            "volatility_level": classification.volatility_level,
            "liquidity_status": classification.liquidity_status,
            "matched_rules": classification.matched_rules,
            "timestamp": classification.timestamp.isoformat(),
            "component_scores": {
                "trend": classification.trend_score,
                "volatility": classification.volatility_score,
                "breadth": classification.breadth_score,
                "institutional": classification.institutional_score,
                "liquidity": classification.liquidity_score,
            },
        }
    
    def get_regime_performance_stats(self) -> dict[str, Any]:
        """
        Get performance statistics by regime type.
        
        Returns:
            Dict with prediction accuracy by regime
        """
        return self.history_manager.get_performance_stats()


# Singleton instance for easy access
_regime_engine: MarketRegimeEngine | None = None


def get_regime_engine(lookback_days: int = 200) -> MarketRegimeEngine:
    """
    Get singleton instance of MarketRegimeEngine.
    
    Args:
        lookback_days: Lookback period for feature computation
        
    Returns:
        MarketRegimeEngine instance
    """
    global _regime_engine
    if _regime_engine is None:
        _regime_engine = MarketRegimeEngine(lookback_days=lookback_days)
    return _regime_engine
