"""
Condition Builder

Builds condition objects from market context or specifications.
Provides fluent interface for creating conditions.
"""

from typing import Optional, List

from research.interactions.condition_engine.condition import Condition
from research.interactions.market_context.market_state import Trend, Volatility, Liquidity, MarketBreadth, OptionsSentiment
from utils.logger import get_logger

logger = get_logger("research.interactions.condition_engine")


class ConditionBuilder:
    """
    Builds condition objects with fluent interface.
    
    Example:
        builder = ConditionBuilder()
        condition = builder.trend("bull").volatility("low").sector("IT").build()
    """
    
    def __init__(self):
        """Initialize condition builder."""
        self._trend: Optional[str] = None
        self._volatility: Optional[str] = None
        self._sector: Optional[str] = None
        self._liquidity: Optional[str] = None
        self._market_breadth: Optional[str] = None
        self._options_sentiment: Optional[str] = None
        self._timeframe: Optional[str] = None
        self._holding_period: Optional[int] = None
        self._logger = get_logger("research.interactions.condition_engine")
    
    def trend(self, value: str) -> "ConditionBuilder":
        """Set trend condition."""
        self._trend = value
        return self
    
    def volatility(self, value: str) -> "ConditionBuilder":
        """Set volatility condition."""
        self._volatility = value
        return self
    
    def sector(self, value: str) -> "ConditionBuilder":
        """Set sector condition."""
        self._sector = value
        return self
    
    def liquidity(self, value: str) -> "ConditionBuilder":
        """Set liquidity condition."""
        self._liquidity = value
        return self
    
    def market_breadth(self, value: str) -> "ConditionBuilder":
        """Set market breadth condition."""
        self._market_breadth = value
        return self
    
    def options_sentiment(self, value: str) -> "ConditionBuilder":
        """Set options sentiment condition."""
        self._options_sentiment = value
        return self
    
    def timeframe(self, value: str) -> "ConditionBuilder":
        """Set timeframe condition."""
        self._timeframe = value
        return self
    
    def holding_period(self, value: int) -> "ConditionBuilder":
        """Set holding period condition."""
        self._holding_period = value
        return self
    
    def build(self) -> Condition:
        """
        Build the condition.
        
        Returns:
            Condition object
        """
        condition = Condition(
            trend=self._trend,
            volatility=self._volatility,
            sector=self._sector,
            liquidity=self._liquidity,
            market_breadth=self._market_breadth,
            options_sentiment=self._options_sentiment,
            timeframe=self._timeframe,
            holding_period=self._holding_period,
        )
        
        # Validate
        is_valid, errors = condition.validate()
        if not is_valid:
            self._logger.warning(f"Built invalid condition: {errors}")
        
        return condition
    
    def reset(self) -> "ConditionBuilder":
        """Reset builder to initial state."""
        self._trend = None
        self._volatility = None
        self._sector = None
        self._liquidity = None
        self._market_breadth = None
        self._options_sentiment = None
        self._timeframe = None
        self._holding_period = None
        return self


class ConditionFactory:
    """
    Factory for creating common condition patterns.
    """
    
    @staticmethod
    def bull_market() -> Condition:
        """Create bull market condition."""
        return Condition(trend=Trend.BULL.value)
    
    @staticmethod
    def bear_market() -> Condition:
        """Create bear market condition."""
        return Condition(trend=Trend.BEAR.value)
    
    @staticmethod
    def low_volatility() -> Condition:
        """Create low volatility condition."""
        return Condition(volatility=Volatility.LOW.value)
    
    @staticmethod
    def high_volatility() -> Condition:
        """Create high volatility condition."""
        return Condition(volatility=Volatility.HIGH.value)
    
    @staticmethod
    def bull_low_vol() -> Condition:
        """Create bull market with low volatility condition."""
        return Condition(
            trend=Trend.BULL.value,
            volatility=Volatility.LOW.value,
        )
    
    @staticmethod
    def bear_high_vol() -> Condition:
        """Create bear market with high volatility condition."""
        return Condition(
            trend=Trend.BEAR.value,
            volatility=Volatility.HIGH.value,
        )
    
    @staticmethod
    def strong_bull() -> Condition:
        """Create strong bull market condition."""
        return Condition(
            trend=Trend.BULL.value,
            volatility=Volatility.LOW.value,
            market_breadth=MarketBreadth.STRONG.value,
            options_sentiment=OptionsSentiment.BULLISH.value,
        )
    
    @staticmethod
    def strong_bear() -> Condition:
        """Create strong bear market condition."""
        return Condition(
            trend=Trend.BEAR.value,
            volatility=Volatility.HIGH.value,
            market_breadth=MarketBreadth.WEAK.value,
            options_sentiment=OptionsSentiment.BEARISH.value,
        )
    
    @staticmethod
    def from_dict(data: dict) -> Condition:
        """
        Create condition from dictionary.
        
        Args:
            data: Dictionary with condition values
            
        Returns:
            Condition object
        """
        return Condition.deserialize(data)
    
    @staticmethod
    def create_all_combinations(
        trends: List[str],
        volatilities: List[str],
        sectors: Optional[List[str]] = None,
    ) -> List[Condition]:
        """
        Create all combinations of conditions.
        
        Args:
            trends: List of trend values
            volatilities: List of volatility values
            sectors: Optional list of sector values
            
        Returns:
            List of Condition objects
        """
        conditions = []
        
        for trend in trends:
            for volatility in volatilities:
                if sectors:
                    for sector in sectors:
                        conditions.append(Condition(
                            trend=trend,
                            volatility=volatility,
                            sector=sector,
                        ))
                else:
                    conditions.append(Condition(
                        trend=trend,
                        volatility=volatility,
                    ))
        
        return conditions


def build_condition(
    trend: Optional[str] = None,
    volatility: Optional[str] = None,
    sector: Optional[str] = None,
    liquidity: Optional[str] = None,
    market_breadth: Optional[str] = None,
    options_sentiment: Optional[str] = None,
    timeframe: Optional[str] = None,
    holding_period: Optional[int] = None,
) -> Condition:
    """
    Convenience function to build a condition.
    
    Args:
        trend: Trend value
        volatility: Volatility value
        sector: Sector value
        liquidity: Liquidity value
        market_breadth: Market breadth value
        options_sentiment: Options sentiment value
        timeframe: Timeframe value
        holding_period: Holding period in days
        
    Returns:
        Condition object
    """
    return Condition(
        trend=trend,
        volatility=volatility,
        sector=sector,
        liquidity=liquidity,
        market_breadth=market_breadth,
        options_sentiment=options_sentiment,
        timeframe=timeframe,
        holding_period=holding_period,
    )
