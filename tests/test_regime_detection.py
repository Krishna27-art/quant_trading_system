"""
Unit tests for Market Regime Detection module.

Tests:
- Feature computation
- Rule classification
- History storage/retrieval
- Signal filtering
"""

import pytest
from datetime import date, timedelta

from regime import (
    RegimeFeatures,
    RegimeFeatureEngine,
    RegimeRuleEngine,
    RegimeType,
    MarketRegimeEngine,
)


class TestRegimeFeatures:
    """Test regime feature computation."""
    
    def test_regime_features_initialization(self):
        """Test that RegimeFeatures can be initialized."""
        features = RegimeFeatures()
        assert features.nifty_close is None
        assert features.india_vix is None
        assert features.adx is None
    
    def test_regime_features_to_dict(self):
        """Test conversion to dictionary."""
        features = RegimeFeatures()
        features.nifty_close = 22000.0
        features.india_vix = 14.5
        features.adx = 30.0
        
        result = features.to_dict()
        assert result["nifty_close"] == 22000.0
        assert result["india_vix"] == 14.5
        assert result["adx"] == 30.0


class TestRegimeRuleEngine:
    """Test regime rule classification."""
    
    def test_rule_engine_initialization(self):
        """Test that RegimeRuleEngine can be initialized."""
        engine = RegimeRuleEngine()
        assert engine is not None
    
    def test_classify_with_minimal_features(self):
        """Test classification with minimal features."""
        engine = RegimeRuleEngine()
        features = RegimeFeatures()
        
        # Set minimal features
        features.india_vix = 14.0
        features.adx = 18.0
        features.advance_decline_ratio = 1.0
        
        classification = engine.classify(features)
        
        assert classification.regime in [r for r in RegimeType]
        assert 0 <= classification.confidence <= 100
        assert classification.timestamp == date.today()
    
    def test_trend_score_computation(self):
        """Test trend score computation."""
        engine = RegimeRuleEngine()
        features = RegimeFeatures()
        
        # Bullish setup
        features.price_above_ema50 = True
        features.price_above_ema200 = True
        features.ema20_above_ema50 = True
        features.ema20_above_ema200 = True
        features.adx = 35
        features.nifty_pct_change = 1.5
        
        score = engine._compute_trend_score(features)
        assert score > 0  # Should be positive for bullish setup
    
    def test_volatility_score_computation(self):
        """Test volatility score computation."""
        engine = RegimeRuleEngine()
        features = RegimeFeatures()
        
        # High volatility setup
        features.india_vix = 25
        features.atr_pct = 2.0
        features.daily_range_pct = 2.5
        
        score = engine._compute_volatility_score(features)
        assert score > 50  # Should be high for high volatility
    
    def test_high_volatility_detection(self):
        """Test high volatility regime detection."""
        engine = RegimeRuleEngine()
        features = RegimeFeatures()
        
        features.india_vix = 22
        features.atr_pct = 1.8
        features.daily_range_pct = 2.2
        
        volatility_score = engine._compute_volatility_score(features)
        is_high_vol = engine._is_high_volatility(features, volatility_score)
        
        assert is_high_vol is True
    
    def test_strong_bull_detection(self):
        """Test strong bull regime detection."""
        engine = RegimeRuleEngine()
        features = RegimeFeatures()
        
        features.price_above_ema50 = True
        features.ema20_above_ema50 = True
        features.adx = 30
        features.india_vix = 12
        features.fii_buying = True
        features.advance_decline_ratio = 2.0
        
        trend_score = engine._compute_trend_score(features)
        breadth_score = engine._compute_breadth_score(features)
        inst_score = engine._compute_institutional_score(features)
        
        is_strong_bull = engine._is_strong_bull(features, trend_score, breadth_score, inst_score)
        
        assert is_strong_bull is True


class TestMarketRegimeEngine:
    """Test the main market regime engine."""
    
    def test_engine_initialization(self):
        """Test that MarketRegimeEngine can be initialized."""
        engine = MarketRegimeEngine(lookback_days=100)
        assert engine is not None
        assert engine.feature_engine is not None
        assert engine.rule_engine is not None
        assert engine.history_manager is not None
    
    def test_signal_filtering(self):
        """Test signal filtering by regime."""
        engine = MarketRegimeEngine()
        
        # Test breakout signal in high volatility (should be filtered)
        signal = {"type": "breakout", "symbol": "RELIANCE", "confidence": 0.85}
        
        # Mock current regime as high volatility
        from regime.regime_rules import RegimeType, RegimeClassification
        mock_classification = RegimeClassification(
            regime=RegimeType.HIGH_VOLATILITY,
            confidence=85.0,
            timestamp=date.today(),
            trend_score=0.0,
            volatility_score=80.0,
            breadth_score=0.0,
            institutional_score=0.0,
            liquidity_score=50.0,
            matched_rules=["High VIX"],
            trend_strength="Neutral",
            volatility_level="High",
            liquidity_status="Normal",
        )
        
        # Temporarily replace get_current_regime
        original_get = engine.get_current_regime
        engine.get_current_regime = lambda: mock_classification
        
        filtered = engine.filter_signal_by_regime(signal, "breakout")
        
        assert filtered["regime_filtered"] is True
        assert filtered["regime"]["type"] == "High Volatility"
        
        # Restore original method
        engine.get_current_regime = original_get
    
    def test_regime_adjustment_factor(self):
        """Test regime adjustment factor computation."""
        engine = MarketRegimeEngine()
        
        # Strong bull with breakout signal
        factor = engine._get_regime_adjustment_factor("breakout", RegimeType.STRONG_BULL)
        assert factor > 1.0  # Should boost breakout signals
        
        # Strong bull with mean reversion signal
        factor = engine._get_regime_adjustment_factor("mean_reversion", RegimeType.STRONG_BULL)
        assert factor < 1.0  # Should reduce mean reversion signals
        
        # Sideways with mean reversion signal
        factor = engine._get_regime_adjustment_factor("mean_reversion", RegimeType.SIDEWAYS)
        assert factor > 1.0  # Should boost mean reversion in sideways
    
    def test_get_regime_summary(self):
        """Test regime summary generation."""
        engine = MarketRegimeEngine()
        
        summary = engine.get_regime_summary()
        
        assert "regime" in summary
        assert "confidence" in summary
        assert "trend_strength" in summary
        assert "volatility_level" in summary
        assert "liquidity_status" in summary
        assert "component_scores" in summary


class TestRegimeHistory:
    """Test regime history storage and retrieval."""
    
    def test_history_manager_initialization(self):
        """Test that RegimeHistoryManager can be initialized."""
        from regime.regime_history import RegimeHistoryManager
        
        manager = RegimeHistoryManager()
        assert manager is not None
    
    def test_save_and_retrieve_regime(self):
        """Test saving and retrieving regime data."""
        from regime.regime_history import RegimeHistoryManager
        from regime.regime_rules import RegimeType, RegimeClassification
        
        manager = RegimeHistoryManager()
        
        # Create test classification
        test_date = date.today() - timedelta(days=1)
        classification = RegimeClassification(
            regime=RegimeType.BULL,
            confidence=75.0,
            timestamp=test_date,
            trend_score=40.0,
            volatility_score=20.0,
            breadth_score=30.0,
            institutional_score=25.0,
            liquidity_score=60.0,
            matched_rules=["NIFTY > EMA50", "Positive trend"],
            trend_strength="Moderate",
            volatility_level="Normal",
            liquidity_status="Normal",
        )
        
        features = RegimeFeatures()
        features.nifty_close = 22000.0
        features.india_vix = 14.0
        
        # Save
        manager.save_regime(test_date, classification, features)
        
        # Retrieve
        retrieved = manager.get_regime_for_date(test_date)
        
        assert retrieved is not None
        assert retrieved.regime == RegimeType.BULL
        assert retrieved.confidence == 75.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
