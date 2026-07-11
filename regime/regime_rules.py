"""
Market Regime Rules Module

Rule-based classification engine for market regimes.
Implements simple, interpretable rules for 6 regimes:
1. Strong Bull
2. Bull
3. Sideways
4. Bear
5. High Volatility
6. Event Day

Rules are designed to be:
- Simple and interpretable
- Based on well-understood market concepts
- Easy to debug and explain
- Sufficient for most practical trading scenarios
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any

from regime.regime_features import RegimeFeatures
from utils.logger import get_logger

logger = get_logger("regime_rules")


class RegimeType(Enum):
    """Enumeration of market regime types."""
    STRONG_BULL = "Strong Bull"
    BULL = "Bull"
    SIDEWAYS = "Sideways"
    BEAR = "Bear"
    HIGH_VOLATILITY = "High Volatility"
    EVENT_DAY = "Event Day"


@dataclass
class RegimeClassification:
    """Result of regime classification with confidence and metadata."""
    
    regime: RegimeType
    confidence: float  # 0-100
    timestamp: date
    
    # Component scores (for explainability)
    trend_score: float
    volatility_score: float
    breadth_score: float
    institutional_score: float
    liquidity_score: float
    
    # Flags that contributed to decision
    matched_rules: list[str]
    
    # Additional context
    trend_strength: str
    volatility_level: str
    liquidity_status: str
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "regime": self.regime.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "trend_score": self.trend_score,
            "volatility_score": self.volatility_score,
            "breadth_score": self.breadth_score,
            "institutional_score": self.institutional_score,
            "liquidity_score": self.liquidity_score,
            "matched_rules": self.matched_rules,
            "trend_strength": self.trend_strength,
            "volatility_level": self.volatility_level,
            "liquidity_status": self.liquidity_status,
        }


class RegimeRuleEngine:
    """
    Rule-based engine for market regime classification.
    
    Uses simple, interpretable rules based on market features.
    Prioritizes clarity and debuggability over complexity.
    """
    
    def __init__(self):
        """Initialize the rule engine."""
        self.logger = logger
        
    def classify(self, features: RegimeFeatures, asof_date: date | None = None) -> RegimeClassification:
        """
        Classify market regime based on features.
        
        Args:
            features: Computed regime features
            asof_date: Date of classification (defaults to today)
            
        Returns:
            RegimeClassification with regime type and confidence
        """
        if asof_date is None:
            asof_date = date.today()
        
        self.logger.info(f"Classifying regime for {asof_date}")
        
        # Compute component scores
        trend_score = self._compute_trend_score(features)
        volatility_score = self._compute_volatility_score(features)
        breadth_score = self._compute_breadth_score(features)
        institutional_score = self._compute_institutional_score(features)
        liquidity_score = self._compute_liquidity_score(features)
        
        # Apply classification rules in priority order
        classification = self._apply_rules(
            features, asof_date,
            trend_score, volatility_score, breadth_score,
            institutional_score, liquidity_score
        )
        
        self.logger.info(f"Regime classified as {classification.regime.value} with {classification.confidence}% confidence")
        
        return classification
    
    def _compute_trend_score(self, features: RegimeFeatures) -> float:
        """
        Compute trend score (-100 to 100).
        Positive = bullish, Negative = bearish.
        """
        score = 0
        
        # Price vs EMAs (40 points total)
        if features.price_above_ema200:
            score += 15
        if features.price_above_ema50:
            score += 15
        if features.ema20_above_ema50:
            score += 10
        
        # EMA alignment (10 points)
        if features.ema20_above_ema200:
            score += 10
        
        # ADX strength (20 points)
        if features.adx:
            if features.adx > 40:
                score += 20
            elif features.adx > 25:
                score += 15
            elif features.adx > 20:
                score += 10
        
        # NIFTY daily change (30 points)
        if features.nifty_pct_change:
            if features.nifty_pct_change > 1.0:
                score += 30
            elif features.nifty_pct_change > 0.5:
                score += 20
            elif features.nifty_pct_change > 0:
                score += 10
            elif features.nifty_pct_change < -1.0:
                score -= 30
            elif features.nifty_pct_change < -0.5:
                score -= 20
            elif features.nifty_pct_change < 0:
                score -= 10
        
        return max(-100, min(100, score))
    
    def _compute_volatility_score(self, features: RegimeFeatures) -> float:
        """
        Compute volatility score (0 to 100).
        Higher = more volatile.
        """
        score = 0
        
        # India VIX (50 points)
        if features.india_vix:
            if features.india_vix > 30:
                score += 50
            elif features.india_vix > 25:
                score += 40
            elif features.india_vix > 20:
                score += 30
            elif features.india_vix > 15:
                score += 20
            elif features.india_vix < 13:
                score += 0  # Low volatility
            else:
                score += 10
        
        # ATR percentage (30 points)
        if features.atr_pct:
            if features.atr_pct > 2.0:
                score += 30
            elif features.atr_pct > 1.5:
                score += 20
            elif features.atr_pct > 1.0:
                score += 10
        
        # Daily range (20 points)
        if features.daily_range_pct:
            if features.daily_range_pct > 2.0:
                score += 20
            elif features.daily_range_pct > 1.5:
                score += 10
        
        return min(100, score)
    
    def _compute_breadth_score(self, features: RegimeFeatures) -> float:
        """
        Compute breadth score (-100 to 100).
        Positive = healthy breadth, Negative = poor breadth.
        """
        score = 0
        
        # Advance/Decline ratio (60 points)
        if features.advance_decline_ratio:
            if features.advance_decline_ratio > 2.0:
                score += 60
            elif features.advance_decline_ratio > 1.5:
                score += 45
            elif features.advance_decline_ratio > 1.0:
                score += 30
            elif features.advance_decline_ratio < 0.5:
                score -= 60
            elif features.advance_decline_ratio < 0.8:
                score -= 30
        
        # % above EMAs (40 points)
        if features.pct_above_ema50:
            if features.pct_above_ema50 > 0.7:
                score += 20
            elif features.pct_above_ema50 < 0.3:
                score -= 20
        
        if features.pct_above_ema200:
            if features.pct_above_ema200 > 0.6:
                score += 20
            elif features.pct_above_ema200 < 0.2:
                score -= 20
        
        return max(-100, min(100, score))
    
    def _compute_institutional_score(self, features: RegimeFeatures) -> float:
        """
        Compute institutional flow score (-100 to 100).
        Positive = net buying, Negative = net selling.
        """
        score = 0
        
        # FII cash flow (40 points)
        if features.fii_net_cash:
            if features.fii_net_cash > 1000:  # > 1000 Cr
                score += 40
            elif features.fii_net_cash > 500:
                score += 30
            elif features.fii_net_cash > 0:
                score += 20
            elif features.fii_net_cash < -1000:
                score -= 40
            elif features.fii_net_cash < -500:
                score -= 30
            elif features.fii_net_cash < 0:
                score -= 20
        
        # FII futures flow (30 points)
        if features.fii_net_futures:
            if features.fii_net_futures > 500:
                score += 30
            elif features.fii_net_futures > 0:
                score += 15
            elif features.fii_net_futures < -500:
                score -= 30
            elif features.fii_net_futures < 0:
                score -= 15
        
        # DII cash flow (30 points)
        if features.dii_net_cash:
            if features.dii_net_cash > 500:
                score += 30
            elif features.dii_net_cash > 0:
                score += 15
            elif features.dii_net_cash < -500:
                score -= 30
            elif features.dii_net_cash < 0:
                score -= 15
        
        return max(-100, min(100, score))
    
    def _compute_liquidity_score(self, features: RegimeFeatures) -> float:
        """
        Compute liquidity score (0 to 100).
        Higher = better liquidity.
        """
        score = 0
        
        # Volume ratio (70 points)
        if features.volume_ratio:
            if features.volume_ratio > 1.5:
                score += 70
            elif features.volume_ratio > 1.2:
                score += 50
            elif features.volume_ratio > 0.8:
                score += 30
            elif features.volume_ratio < 0.5:
                score += 10
            else:
                score += 20
        
        # Absolute volume (30 points)
        if features.nifty_volume:
            if features.nifty_volume > 100_000_000:  # High volume
                score += 30
            elif features.nifty_volume > 50_000_000:
                score += 20
            elif features.nifty_volume > 25_000_000:
                score += 10
        
        return min(100, score)
    
    def _apply_rules(
        self,
        features: RegimeFeatures,
        asof_date: date,
        trend_score: float,
        volatility_score: float,
        breadth_score: float,
        institutional_score: float,
        liquidity_score: float,
    ) -> RegimeClassification:
        """
        Apply classification rules in priority order.
        
        Rules are evaluated in order - first match wins.
        """
        matched_rules = []
        confidence = 0.0
        regime = RegimeType.SIDEWAYS
        
        # Priority 1: High Volatility (overrides everything)
        if self._is_high_volatility(features, volatility_score):
            regime = RegimeType.HIGH_VOLATILITY
            matched_rules = [
                "India VIX > 20",
                "ATR Rising",
                "Large Daily Ranges",
            ]
            confidence = min(95, 60 + volatility_score * 0.4)
        
        # Priority 2: Event Day (gap detection)
        elif self._is_event_day(features):
            regime = RegimeType.EVENT_DAY
            matched_rules = [
                "Gap Up / Gap Down detected",
                "Abnormal price movement",
            ]
            confidence = 85
        
        # Priority 3: Strong Bull
        elif self._is_strong_bull(features, trend_score, breadth_score, institutional_score):
            regime = RegimeType.STRONG_BULL
            matched_rules = [
                "NIFTY > EMA50",
                "EMA20 > EMA50",
                "ADX > 25",
                "India VIX < 15",
                "FII Buying",
                "Advance/Decline > 1.5",
            ]
            confidence = min(95, 70 + (trend_score + breadth_score + institutional_score) / 300 * 25)
        
        # Priority 4: Bull
        elif self._is_bull(features, trend_score, breadth_score):
            regime = RegimeType.BULL
            matched_rules = [
                "NIFTY > EMA50",
                "Positive trend",
                "Healthy breadth",
            ]
            confidence = min(90, 60 + (trend_score + breadth_score) / 200 * 30)
        
        # Priority 5: Bear
        elif self._is_bear(features, trend_score, breadth_score, institutional_score):
            regime = RegimeType.BEAR
            matched_rules = [
                "NIFTY < EMA200",
                "EMA20 < EMA50",
                "FII Selling",
                "Advance/Decline < 0.8",
                "VIX Rising",
            ]
            confidence = min(90, 60 + (abs(trend_score) + abs(breadth_score) + abs(institutional_score)) / 300 * 30)
        
        # Priority 6: Sideways (default)
        else:
            regime = RegimeType.SIDEWAYS
            matched_rules = [
                "ADX < 20",
                "ATR Low",
                "VIX Stable",
                "EMA20 ≈ EMA50",
                "PCR Neutral",
            ]
            confidence = min(85, 50 + (100 - volatility_score) * 0.35)
        
        return RegimeClassification(
            regime=regime,
            confidence=round(confidence, 1),
            timestamp=asof_date,
            trend_score=round(trend_score, 1),
            volatility_score=round(volatility_score, 1),
            breadth_score=round(breadth_score, 1),
            institutional_score=round(institutional_score, 1),
            liquidity_score=round(liquidity_score, 1),
            matched_rules=matched_rules,
            trend_strength=features.trend_strength,
            volatility_level=features.volatility_level,
            liquidity_status=features.liquidity_status,
        )
    
    def _is_high_volatility(self, features: RegimeFeatures, volatility_score: float) -> bool:
        """Check if market is in high volatility regime."""
        conditions = [
            features.india_vix is not None and features.india_vix > 20,
            features.atr_pct is not None and features.atr_pct > 1.5,
            features.daily_range_pct is not None and features.daily_range_pct > 2.0,
            volatility_score > 50,
        ]
        return sum(conditions) >= 2  # At least 2 conditions must be true
    
    def _is_event_day(self, features: RegimeFeatures) -> bool:
        """Check if today is an event day (gap up/down)."""
        # Event day detection based on abnormal price movement
        if features.nifty_pct_change is None:
            return False
        
        # Gap up/down > 2% or very high volume
        gap_condition = abs(features.nifty_pct_change) > 2.0
        volume_condition = features.volume_ratio and features.volume_ratio > 2.0
        
        return gap_condition or volume_condition
    
    def _is_strong_bull(
        self,
        features: RegimeFeatures,
        trend_score: float,
        breadth_score: float,
        institutional_score: float,
    ) -> bool:
        """Check if market is in strong bull regime."""
        conditions = [
            features.price_above_ema50,
            features.ema20_above_ema50,
            features.adx and features.adx > 25,
            features.india_vix and features.india_vix < 15,
            features.fii_buying,
            features.advance_decline_ratio and features.advance_decline_ratio > 1.5,
            trend_score > 50,
            breadth_score > 30,
            institutional_score > 20,
        ]
        conditions = [bool(c) for c in conditions]
        return sum(conditions) >= 5  # At least 5 conditions must be true
    
    def _is_bull(self, features: RegimeFeatures, trend_score: float, breadth_score: float) -> bool:
        """Check if market is in bull regime."""
        conditions = [
            features.price_above_ema50,
            features.ema20_above_ema50,
            trend_score > 20,
            breadth_score > 10,
            features.nifty_pct_change and features.nifty_pct_change > 0,
        ]
        conditions = [bool(c) for c in conditions]
        return sum(conditions) >= 3
    
    def _is_bear(
        self,
        features: RegimeFeatures,
        trend_score: float,
        breadth_score: float,
        institutional_score: float,
    ) -> bool:
        """Check if market is in bear regime."""
        conditions = [
            not features.price_above_ema200,
            not features.ema20_above_ema50,
            not features.fii_buying,
            features.advance_decline_ratio and features.advance_decline_ratio < 0.8,
            features.india_vix and features.india_vix > 15,
            trend_score < -20,
            breadth_score < -10,
            institutional_score < -20,
        ]
        conditions = [bool(c) for c in conditions]
        return sum(conditions) >= 4
