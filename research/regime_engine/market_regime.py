"""
Market Regime Engine

Classifies market regimes (bull, bear, sideways) based on market conditions.
Essential for understanding when factors work and when they fail.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("research.market_regime")


class MarketRegime(Enum):
    """Market regime classification."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class RegimeClassification:
    """Result of regime classification."""
    regime: MarketRegime
    confidence: float
    market_return: float
    volatility: float
    trend_strength: float
    description: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "regime": self.regime.value,
            "confidence": round(self.confidence, 4),
            "market_return": round(self.market_return, 4),
            "volatility": round(self.volatility, 4),
            "trend_strength": round(self.trend_strength, 4),
            "description": self.description,
        }


class MarketRegimeClassifier:
    """
    Classifies market regimes based on market conditions.
    
    Regimes:
    - Bull: Rising market with positive momentum
    - Bear: Falling market with negative momentum
    - Sideways: Range-bound market
    - High Volatility: Elevated volatility
    - Low Volatility: Low volatility environment
    
    This is critical for understanding factor performance across different market conditions.
    """
    
    def __init__(
        self,
        lookback: int = 60,
        bull_threshold: float = 0.05,
        bear_threshold: float = -0.05,
        volatility_threshold: float = 0.02,
    ):
        """
        Initialize market regime classifier.
        
        Args:
            lookback: Lookback period for regime calculation
            bull_threshold: Monthly return threshold for bull regime
            bear_threshold: Monthly return threshold for bear regime
            volatility_threshold: Volatility threshold for high/low volatility
        """
        self.lookback = lookback
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold
        self.volatility_threshold = volatility_threshold
        self._logger = get_logger("research.market_regime")
    
    def classify(
        self,
        market_prices: pd.Series,
    ) -> RegimeClassification:
        """
        Classify current market regime.
        
        Args:
            market_prices: Series with market index prices (e.g., NIFTY)
            
        Returns:
            RegimeClassification
        """
        if len(market_prices) < self.lookback:
            return RegimeClassification(
                regime=MarketRegime.SIDEWAYS,
                confidence=0.0,
                market_return=0.0,
                volatility=0.0,
                trend_strength=0.0,
                description="Insufficient data for regime classification",
            )
        
        # Calculate recent returns
        recent_prices = market_prices.tail(self.lookback)
        returns = recent_prices.pct_change().dropna()
        
        # Calculate metrics
        market_return = returns.sum()  # Total return over lookback
        volatility = returns.std()
        
        # Calculate trend strength (using moving average slope)
        ma_short = recent_prices.rolling(20).mean().iloc[-1]
        ma_long = recent_prices.rolling(60).mean().iloc[-1]
        trend_strength = (ma_short - ma_long) / ma_long if ma_long != 0 else 0
        
        # Classify regime
        regime = self._determine_regime(market_return, volatility, trend_strength)
        confidence = self._calculate_confidence(market_return, volatility, trend_strength)
        description = self._generate_description(regime, market_return, volatility, trend_strength)
        
        return RegimeClassification(
            regime=regime,
            confidence=confidence,
            market_return=market_return,
            volatility=volatility,
            trend_strength=trend_strength,
            description=description,
        )
    
    def _determine_regime(
        self,
        market_return: float,
        volatility: float,
        trend_strength: float,
    ) -> MarketRegime:
        """Determine market regime based on metrics."""
        # Primary classification based on return
        if market_return >= self.bull_threshold:
            return MarketRegime.BULL
        elif market_return <= self.bear_threshold:
            return MarketRegime.BEAR
        else:
            return MarketRegime.SIDEWAYS
    
    def _calculate_confidence(
        self,
        market_return: float,
        volatility: float,
        trend_strength: float,
    ) -> float:
        """Calculate confidence in regime classification."""
        # Confidence based on how far from thresholds
        if market_return >= self.bull_threshold:
            confidence = min(1.0, (market_return - self.bull_threshold) / self.bull_threshold)
        elif market_return <= self.bear_threshold:
            confidence = min(1.0, (self.bear_threshold - market_return) / abs(self.bear_threshold))
        else:
            # Sideways regime - confidence based on proximity to thresholds
            distance_to_bull = abs(market_return - self.bull_threshold)
            distance_to_bear = abs(market_return - self.bear_threshold)
            min_distance = min(distance_to_bull, distance_to_bear)
            confidence = 1.0 - (min_distance / (self.bull_threshold - self.bear_threshold))
        
        return max(0.0, min(1.0, confidence))
    
    def _generate_description(
        self,
        regime: MarketRegime,
        market_return: float,
        volatility: float,
        trend_strength: float,
    ) -> str:
        """Generate human-readable description of regime."""
        vol_status = "high" if volatility > self.volatility_threshold else "low"
        
        if regime == MarketRegime.BULL:
            return f"Bull market with {market_return:.1%} return, {vol_status} volatility, trend strength {trend_strength:.1%}"
        elif regime == MarketRegime.BEAR:
            return f"Bear market with {market_return:.1%} return, {vol_status} volatility, trend strength {trend_strength:.1%}"
        else:
            return f"Sideways market with {market_return:.1%} return, {vol_status} volatility, trend strength {trend_strength:.1%}"
    
    def classify_historical(
        self,
        market_prices: pd.Series,
    ) -> pd.Series:
        """
        Classify market regime historically (rolling window).
        
        Args:
            market_prices: Series with market index prices
            
        Returns:
            Series with regime classifications
        """
        regimes = []
        
        for i in range(self.lookback, len(market_prices)):
            window_prices = market_prices.iloc[i - self.lookback : i + 1]
            classification = self.classify(window_prices)
            regimes.append(classification.regime.value)
        
        # Create series with same index
        regime_series = pd.Series(regimes, index=market_prices.index[self.lookback:])
        
        return regime_series
    
    def analyze_factor_by_regime(
        self,
        factor_values: pd.Series,
        future_returns: pd.Series,
        market_prices: pd.Series,
    ) -> Dict[str, Dict]:
        """
        Analyze factor performance by market regime.
        
        Args:
            factor_values: Series with factor values
            future_returns: Series with future returns
            market_prices: Series with market index prices
            
        Returns:
            Dictionary mapping regime names to performance metrics
        """
        # Classify historical regimes
        regime_series = self.classify_historical(market_prices)
        
        # Align with factor data
        aligned_factor = factor_values.reindex(regime_series.index)
        aligned_returns = future_returns.reindex(regime_series.index)
        
        # Remove NaN
        valid_mask = aligned_factor.notna() & aligned_returns.notna() & regime_series.notna()
        aligned_factor = aligned_factor[valid_mask]
        aligned_returns = aligned_returns[valid_mask]
        aligned_regimes = regime_series[valid_mask]
        
        # Calculate performance by regime
        from research.factor_tests.information_coefficient import calculate_ic
        
        regime_performance = {}
        
        for regime_name in aligned_regimes.unique():
            regime_mask = aligned_regimes == regime_name
            regime_factor = aligned_factor[regime_mask]
            regime_returns = aligned_returns[regime_mask]
            
            if len(regime_factor) < 10:
                continue
            
            try:
                ic_result = calculate_ic(regime_factor, regime_returns)
                
                regime_performance[regime_name] = {
                    "mean_ic": ic_result.mean_ic,
                    "mean_rank_ic": ic_result.mean_rank_ic,
                    "hit_rate": ic_result.hit_rate,
                    "sample_size": len(regime_factor),
                }
            except Exception as e:
                self._logger.warning(f"Failed to calculate IC for regime {regime_name}: {e}")
                regime_performance[regime_name] = {
                    "mean_ic": 0.0,
                    "mean_rank_ic": 0.0,
                    "hit_rate": 0.0,
                    "sample_size": len(regime_factor),
                }
        
        return regime_performance


def classify_market_regime(
    market_prices: pd.Series,
    lookback: int = 60,
) -> RegimeClassification:
    """
    Convenience function to classify market regime.
    
    Args:
        market_prices: Series with market index prices
        lookback: Lookback period
        
    Returns:
        RegimeClassification
    """
    classifier = MarketRegimeClassifier(lookback=lookback)
    return classifier.classify(market_prices)
