"""
Alpha Regime Adjuster

Adjusts alpha category weights based on market regime.

STEP 5: Regime adjustment

This module connects to the Market Regime Engine and dynamically adjusts
category weights based on the current market conditions.

Example:
- Bear Market: Reduce momentum weights, increase quality weights
- Bull Market: Increase momentum weights, reduce mean reversion weights
- High Volatility: Reduce all weights, increase defensive weights
"""

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, Dict, Optional

from regime.market_regime import MarketRegimeEngine, RegimeType
from utils.logger import get_logger

logger = get_logger("alpha_engine.regime")


@dataclass
class RegimeWeightAdjustment:
    """
    Weight adjustment for a specific regime.
    
    Contains multipliers for each category based on regime characteristics.
    """
    regime: RegimeType
    adjustments: Dict[str, float]  # category -> multiplier
    description: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "regime": self.regime.value,
            "adjustments": {k: round(v, 4) for k, v in self.adjustments.items()},
            "description": self.description,
        }


class AlphaRegimeAdjuster:
    """
    Adjusts alpha category weights based on market regime.
    
    This is the integration point between the Alpha Engine and the
    Market Regime Detection system.
    """
    
    def __init__(self, regime_engine: Optional[MarketRegimeEngine] = None):
        """
        Initialize Alpha Regime Adjuster.
        
        Args:
            regime_engine: Optional MarketRegimeEngine instance.
                         If None, will create a new instance.
        """
        self._logger = logger
        self.regime_engine = regime_engine or MarketRegimeEngine()
        
        # Define regime-specific weight adjustments
        self.regime_adjustments = self._initialize_regime_adjustments()
    
    def get_regime_adjustments(
        self,
        asof_date: Optional[date] = None,
    ) -> RegimeWeightAdjustment:
        """
        Get weight adjustments for the current market regime.
        
        Args:
            asof_date: Date to get regime for (defaults to today)
            
        Returns:
            RegimeWeightAdjustment with category multipliers
        """
        # Get current regime
        classification = self.regime_engine.get_current_regime()
        regime = classification.regime
        
        # Get adjustments for this regime
        adjustment = self.regime_adjustments.get(regime)
        
        if adjustment is None:
            self._logger.warning(f"No adjustments defined for regime {regime.value}, using neutral")
            return RegimeWeightAdjustment(
                regime=regime,
                adjustments=self._get_neutral_adjustments(),
                description=f"No specific adjustments for {regime.value}",
            )
        
        self._logger.info(
            f"Applying regime adjustments for {regime.value}",
            extra={
                "confidence": classification.confidence,
                "adjustments": adjustment.adjustments,
            },
        )
        
        return adjustment
    
    def get_adjusted_weights(
        self,
        base_weights: Dict[str, float],
        asof_date: Optional[date] = None,
    ) -> Dict[str, float]:
        """
        Apply regime adjustments to base weights.
        
        Args:
            base_weights: Base category weights
            asof_date: Date to get regime for (defaults to today)
            
        Returns:
            Adjusted weights
        """
        adjustment = self.get_regime_adjustments(asof_date)
        
        adjusted = {}
        for category, weight in base_weights.items():
            multiplier = adjustment.adjustments.get(category, 1.0)
            adjusted[category] = weight * multiplier
        
        # Normalize to ensure sum = 1.0
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {cat: w / total for cat, w in adjusted.items()}
        
        return adjusted
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """
        Get summary of current regime and its impact on alpha weights.
        
        Returns:
            Dictionary with regime summary and expected adjustments
        """
        classification = self.regime_engine.get_current_regime()
        adjustment = self.get_regime_adjustments()
        
        return {
            "regime": {
                "type": classification.regime.value,
                "confidence": classification.confidence,
                "trend_strength": classification.trend_strength,
                "volatility_level": classification.volatility_level,
            },
            "adjustments": adjustment.to_dict(),
            "impact": self._describe_regime_impact(classification.regime),
        }
    
    def _initialize_regime_adjustments(self) -> Dict[RegimeType, RegimeWeightAdjustment]:
        """
        Initialize regime-specific weight adjustments.
        
        These adjustments are based on research showing that different
        alpha factors perform differently in different market regimes.
        
        Returns:
            Dictionary of RegimeType -> RegimeWeightAdjustment
        """
        return {
            # Strong Bull Market
            RegimeType.STRONG_BULL: RegimeWeightAdjustment(
                regime=RegimeType.STRONG_BULL,
                adjustments={
                    "technical": 1.3,  # Trend following works well
                    "volume": 1.2,
                    "options": 1.2,
                    "sentiment": 1.3,  # Positive sentiment amplified
                    "fundamental": 0.9,  # Less relevant in strong momentum
                    "sector": 1.1,
                    "macro": 0.8,
                },
                description="Strong bull: Boost momentum and sentiment, reduce macro focus",
            ),
            
            # Bull Market
            RegimeType.BULL: RegimeWeightAdjustment(
                regime=RegimeType.BULL,
                adjustments={
                    "technical": 1.15,
                    "volume": 1.1,
                    "options": 1.15,
                    "sentiment": 1.1,
                    "fundamental": 0.95,
                    "sector": 1.05,
                    "macro": 0.9,
                },
                description="Bull: Moderate boost for momentum factors",
            ),
            
            # Sideways/Range-bound Market
            RegimeType.SIDEWAYS: RegimeWeightAdjustment(
                regime=RegimeType.SIDEWAYS,
                adjustments={
                    "technical": 0.6,  # Trend following fails
                    "volume": 0.9,
                    "options": 1.2,  # Range trading strategies work
                    "sentiment": 0.8,
                    "fundamental": 1.1,  # Value matters more
                    "sector": 1.0,
                    "macro": 1.0,
                },
                description="Sideways: Reduce trend weights, boost options and fundamentals",
            ),
            
            # Bear Market
            RegimeType.BEAR: RegimeWeightAdjustment(
                regime=RegimeType.BEAR,
                adjustments={
                    "technical": 0.5,  # Bullish technicals fail
                    "volume": 0.8,
                    "options": 1.3,  # Put options, hedging strategies
                    "sentiment": 0.6,  # Negative sentiment dominates
                    "fundamental": 1.3,  # Quality and value matter
                    "sector": 1.1,  # Defensive sectors
                    "macro": 1.2,  # Macro risks elevated
                },
                description="Bear: Reduce momentum, boost defensive and quality factors",
            ),
            
            # High Volatility
            RegimeType.HIGH_VOLATILITY: RegimeWeightAdjustment(
                regime=RegimeType.HIGH_VOLATILITY,
                adjustments={
                    "technical": 0.7,  # Technical signals less reliable
                    "volume": 1.3,  # Volume spikes important
                    "options": 1.4,  # Volatility trading
                    "sentiment": 0.7,  # Sentiment can be misleading
                    "fundamental": 1.1,  # Focus on quality
                    "sector": 0.9,
                    "macro": 1.1,
                },
                description="High volatility: Focus on volume and options, reduce trend following",
            ),
            
            # Event Day (earnings, policy, etc.)
            RegimeType.EVENT_DAY: RegimeWeightAdjustment(
                regime=RegimeType.EVENT_DAY,
                adjustments={
                    "technical": 0.3,  # Technicals break down
                    "volume": 1.5,  # Volume critical
                    "options": 1.5,  # Options pricing key
                    "sentiment": 0.5,  # Sentiment can be wrong
                    "fundamental": 0.8,  # Fundamentals less relevant intraday
                    "sector": 0.7,
                    "macro": 1.2,  # Macro events drive moves
                },
                description="Event day: Extreme focus on volume and options, reduce other factors",
            ),
        }
    
    def _get_neutral_adjustments(self) -> Dict[str, float]:
        """Get neutral adjustments (no change)."""
        return {
            "technical": 1.0,
            "volume": 1.0,
            "options": 1.0,
            "sentiment": 1.0,
            "fundamental": 1.0,
            "sector": 1.0,
            "macro": 1.0,
        }
    
    def _describe_regime_impact(self, regime: RegimeType) -> str:
        """
        Describe the impact of the regime on alpha strategy.
        
        Args:
            regime: Current regime type
            
        Returns:
            Description of regime impact
        """
        descriptions = {
            RegimeType.STRONG_BULL: (
                "Strong bull market favors momentum and trend-following strategies. "
                "Sentiment and technical signals are highly reliable. "
                "Focus on breakout and continuation patterns."
            ),
            RegimeType.BULL: (
                "Bull market supports momentum strategies with moderate confidence. "
                "Technical and volume signals work well. "
                "Maintain bullish bias but watch for reversals."
            ),
            RegimeType.SIDEWAYS: (
                "Sideways market requires range-trading approach. "
                "Trend-following strategies underperform. "
                "Focus on mean reversion, options strategies, and fundamental value."
            ),
            RegimeType.BEAR: (
                "Bear market requires defensive positioning. "
                "Momentum strategies fail, focus on quality and value. "
                "Consider hedging strategies and defensive sectors."
            ),
            RegimeType.HIGH_VOLATILITY: (
                "High volatility regime requires reduced position sizes. "
                "Focus on volume analysis and options strategies. "
                "Technical signals less reliable, prioritize risk management."
            ),
            RegimeType.EVENT_DAY: (
                "Event day with abnormal market conditions. "
                "Standard signals may break down. "
                "Focus on volume and options flow, reduce exposure."
            ),
        }
        
        return descriptions.get(regime, "No specific regime impact description available.")
    
    def should_filter_signal(
        self,
        signal_type: str,
        asof_date: Optional[date] = None,
    ) -> tuple[bool, str]:
        """
        Determine if a signal type should be filtered based on regime.
        
        Args:
            signal_type: Type of signal (e.g., 'breakout', 'momentum')
            asof_date: Date to check regime for
            
        Returns:
            Tuple of (should_filter, reason)
        """
        classification = self.regime_engine.get_current_regime()
        regime = classification.regime
        
        # Define signal type to category mappings
        signal_categories = {
            "breakout": "technical",
            "momentum": "technical",
            "trend_following": "technical",
            "mean_reversion": "technical",
            "volume_spike": "volume",
            "options_flow": "options",
            "sentiment": "sentiment",
        }
        
        category = signal_categories.get(signal_type, "technical")
        adjustment = self.get_regime_adjustments(asof_date)
        multiplier = adjustment.adjustments.get(category, 1.0)
        
        # Filter if multiplier is very low
        if multiplier < 0.5:
            reason = f"Signal type '{signal_type}' filtered: {adjustment.description}"
            return True, reason
        
        # Filter if regime confidence is low
        if classification.confidence < 60:
            reason = f"Signal filtered: Low regime confidence ({classification.confidence}%)"
            return True, reason
        
        return False, "Signal accepted: Compatible with current regime"
