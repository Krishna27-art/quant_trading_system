"""
Interaction Builder

Builds factor-condition interaction objects.
Provides fluent interface for creating interactions.
"""

from dataclasses import dataclass
from typing import Optional

from research.interactions.condition_engine.condition import Condition
from utils.logger import get_logger

logger = get_logger("research.interactions.interaction_engine")


@dataclass
class Interaction:
    """Represents a factor-condition interaction."""
    factor_name: str
    condition: Condition
    
    def get_id(self) -> str:
        """Generate unique ID for interaction."""
        condition_str = self.condition.serialize()
        condition_hash = hash(str(condition_str))
        return f"{self.factor_name}_{abs(condition_hash)}"
    
    def get_description(self) -> str:
        """Get human-readable description."""
        return f"{self.factor_name} when {self.condition.get_description()}"


class InteractionBuilder:
    """
    Builds interaction objects with fluent interface.
    
    Example:
        builder = InteractionBuilder()
        interaction = builder.factor("RSI").trend("bull").volatility("low").build()
    """
    
    def __init__(self):
        """Initialize interaction builder."""
        self._factor_name: Optional[str] = None
        self._trend: Optional[str] = None
        self._volatility: Optional[str] = None
        self._sector: Optional[str] = None
        self._liquidity: Optional[str] = None
        self._market_breadth: Optional[str] = None
        self._options_sentiment: Optional[str] = None
        self._timeframe: Optional[str] = None
        self._holding_period: Optional[int] = None
        self._logger = get_logger("research.interactions.interaction_engine")
    
    def factor(self, name: str) -> "InteractionBuilder":
        """Set factor name."""
        self._factor_name = name
        return self
    
    def trend(self, value: str) -> "InteractionBuilder":
        """Set trend condition."""
        self._trend = value
        return self
    
    def volatility(self, value: str) -> "InteractionBuilder":
        """Set volatility condition."""
        self._volatility = value
        return self
    
    def sector(self, value: str) -> "InteractionBuilder":
        """Set sector condition."""
        self._sector = value
        return self
    
    def liquidity(self, value: str) -> "InteractionBuilder":
        """Set liquidity condition."""
        self._liquidity = value
        return self
    
    def market_breadth(self, value: str) -> "InteractionBuilder":
        """Set market breadth condition."""
        self._market_breadth = value
        return self
    
    def options_sentiment(self, value: str) -> "InteractionBuilder":
        """Set options sentiment condition."""
        self._options_sentiment = value
        return self
    
    def timeframe(self, value: str) -> "InteractionBuilder":
        """Set timeframe condition."""
        self._timeframe = value
        return self
    
    def holding_period(self, value: int) -> "InteractionBuilder":
        """Set holding period condition."""
        self._holding_period = value
        return self
    
    def build(self) -> Interaction:
        """
        Build the interaction.
        
        Returns:
            Interaction object
        """
        if self._factor_name is None:
            raise ValueError("Factor name is required")
        
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
        
        return Interaction(
            factor_name=self._factor_name,
            condition=condition,
        )
    
    def reset(self) -> "InteractionBuilder":
        """Reset builder to initial state."""
        self._factor_name = None
        self._trend = None
        self._volatility = None
        self._sector = None
        self._liquidity = None
        self._market_breadth = None
        self._options_sentiment = None
        self._timeframe = None
        self._holding_period = None
        return self


class InteractionFactory:
    """
    Factory for creating common interaction patterns.
    """
    
    @staticmethod
    def bull_market_rsi() -> Interaction:
        """Create RSI in bull market interaction."""
        return Interaction(
            factor_name="RSI",
            condition=Condition(trend="bull"),
        )
    
    @staticmethod
    def low_vol_momentum() -> Interaction:
        """Create momentum in low volatility interaction."""
        return Interaction(
            factor_name="Momentum",
            condition=Condition(volatility="low"),
        )
    
    @staticmethod
    def it_sector_trend() -> Interaction:
        """Create trend factor in IT sector interaction."""
        return Interaction(
            factor_name="Trend",
            condition=Condition(sector="IT"),
        )
    
    @staticmethod
    def strong_bull_options() -> Interaction:
        """Create options sentiment in strong bull market interaction."""
        from research.interactions.condition_engine.condition_builder import ConditionFactory
        return Interaction(
            factor_name="Options_Sentiment",
            condition=ConditionFactory.strong_bull(),
        )
    
    @staticmethod
    def from_dict(data: dict) -> Interaction:
        """
        Create interaction from dictionary.
        
        Args:
            data: Dictionary with interaction data
            
        Returns:
            Interaction object
        """
        condition = Condition.deserialize(data.get("condition", {}))
        return Interaction(
            factor_name=data.get("factor_name"),
            condition=condition,
        )


def build_interaction(
    factor_name: str,
    trend: Optional[str] = None,
    volatility: Optional[str] = None,
    sector: Optional[str] = None,
    liquidity: Optional[str] = None,
    market_breadth: Optional[str] = None,
    options_sentiment: Optional[str] = None,
    timeframe: Optional[str] = None,
    holding_period: Optional[int] = None,
) -> Interaction:
    """
    Convenience function to build an interaction.
    
    Args:
        factor_name: Name of the factor
        trend: Trend value
        volatility: Volatility value
        sector: Sector value
        liquidity: Liquidity value
        market_breadth: Market breadth value
        options_sentiment: Options sentiment value
        timeframe: Timeframe value
        holding_period: Holding period in days
        
    Returns:
        Interaction object
    """
    condition = Condition(
        trend=trend,
        volatility=volatility,
        sector=sector,
        liquidity=liquidity,
        market_breadth=market_breadth,
        options_sentiment=options_sentiment,
        timeframe=timeframe,
        holding_period=holding_period,
    )
    
    return Interaction(
        factor_name=factor_name,
        condition=condition,
    )
