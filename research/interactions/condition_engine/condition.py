"""
Condition

Represents a market condition for factor interaction testing.
Combines multiple market state dimensions into a testable condition.
"""

from dataclasses import dataclass
from typing import Optional

from research.interactions.market_context.market_state import MarketStateValidator
from utils.logger import get_logger

logger = get_logger("research.interactions.condition_engine")


@dataclass
class Condition:
    """
    Market condition for factor interaction testing.
    
    Contains:
    - Trend
    - Volatility
    - Sector
    - Liquidity
    - Market Breadth
    - Options Sentiment
    - Timeframe
    - Holding Period
    """
    trend: Optional[str] = None  # "bull", "bear", "sideways"
    volatility: Optional[str] = None  # "low", "medium", "high"
    sector: Optional[str] = None  # "IT", "Banking", etc.
    liquidity: Optional[str] = None  # "low", "medium", "high"
    market_breadth: Optional[str] = None  # "weak", "neutral", "strong"
    options_sentiment: Optional[str] = None  # "bearish", "neutral", "bullish"
    timeframe: Optional[str] = None  # "1d", "1h", "15m", etc.
    holding_period: Optional[int] = None  # Days
    
    def matches(self, market_context: dict) -> bool:
        """
        Check if condition matches market context.
        
        Args:
            market_context: Dictionary with market state values
            
        Returns:
            True if condition matches context
        """
        # Check each field - if field is None, it's a wildcard (matches anything)
        if self.trend is not None and market_context.get("trend") != self.trend:
            return False
        
        if self.volatility is not None and market_context.get("volatility") != self.volatility:
            return False
        
        if self.liquidity is not None and market_context.get("liquidity") != self.liquidity:
            return False
        
        if self.market_breadth is not None and market_context.get("market_breadth") != self.market_breadth:
            return False
        
        if self.options_sentiment is not None and market_context.get("options_sentiment") != self.options_sentiment:
            return False
        
        # Sector is handled separately (not in standard market context)
        if self.sector is not None and market_context.get("sector") != self.sector:
            return False
        
        return True
    
    def validate(self) -> tuple[bool, list]:
        """
        Validate condition values.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if self.trend is not None and not MarketStateValidator.validate_trend(self.trend):
            errors.append(f"Invalid trend: {self.trend}")
        
        if self.volatility is not None and not MarketStateValidator.validate_volatility(self.volatility):
            errors.append(f"Invalid volatility: {self.volatility}")
        
        if self.liquidity is not None and not MarketStateValidator.validate_liquidity(self.liquidity):
            errors.append(f"Invalid liquidity: {self.liquidity}")
        
        if self.market_breadth is not None and not MarketStateValidator.validate_market_breadth(self.market_breadth):
            errors.append(f"Invalid market_breadth: {self.market_breadth}")
        
        if self.options_sentiment is not None and not MarketStateValidator.validate_options_sentiment(self.options_sentiment):
            errors.append(f"Invalid options_sentiment: {self.options_sentiment}")
        
        if self.holding_period is not None and self.holding_period <= 0:
            errors.append(f"Invalid holding_period: {self.holding_period}")
        
        return len(errors) == 0, errors
    
    def serialize(self) -> dict:
        """
        Serialize condition to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "trend": self.trend,
            "volatility": self.volatility,
            "sector": self.sector,
            "liquidity": self.liquidity,
            "market_breadth": self.market_breadth,
            "options_sentiment": self.options_sentiment,
            "timeframe": self.timeframe,
            "holding_period": self.holding_period,
        }
    
    @classmethod
    def deserialize(cls, data: dict) -> "Condition":
        """
        Deserialize condition from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Condition object
        """
        return cls(
            trend=data.get("trend"),
            volatility=data.get("volatility"),
            sector=data.get("sector"),
            liquidity=data.get("liquidity"),
            market_breadth=data.get("market_breadth"),
            options_sentiment=data.get("options_sentiment"),
            timeframe=data.get("timeframe"),
            holding_period=data.get("holding_period"),
        )
    
    def get_description(self) -> str:
        """
        Get human-readable description of condition.
        
        Returns:
            Description string
        """
        parts = []
        
        if self.trend:
            parts.append(f"Trend={self.trend}")
        if self.volatility:
            parts.append(f"Volatility={self.volatility}")
        if self.sector:
            parts.append(f"Sector={self.sector}")
        if self.liquidity:
            parts.append(f"Liquidity={self.liquidity}")
        if self.market_breadth:
            parts.append(f"Breadth={self.market_breadth}")
        if self.options_sentiment:
            parts.append(f"Options={self.options_sentiment}")
        if self.timeframe:
            parts.append(f"Timeframe={self.timeframe}")
        if self.holding_period:
            parts.append(f"Holding={self.holding_period}d")
        
        if parts:
            return " + ".join(parts)
        else:
            return "Any Condition"
    
    def is_specific(self) -> bool:
        """
        Check if condition has specific requirements (not all wildcards).
        
        Returns:
            True if condition has at least one specific requirement
        """
        return any([
            self.trend is not None,
            self.volatility is not None,
            self.sector is not None,
            self.liquidity is not None,
            self.market_breadth is not None,
            self.options_sentiment is not None,
            self.timeframe is not None,
            self.holding_period is not None,
        ])
