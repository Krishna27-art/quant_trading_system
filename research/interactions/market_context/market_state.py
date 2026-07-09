"""
Market State

Enumerates and validates market state values.
Provides constants for trend, volatility, liquidity, etc.
"""

from enum import Enum
from typing import List

from utils.logger import get_logger

logger = get_logger("research.interactions.market_context")


class Trend(Enum):
    """Market trend states."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    
    @classmethod
    def all(cls) -> List[str]:
        """Get all trend values."""
        return [t.value for t in cls]
    
    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if trend value is valid."""
        return value in cls.all()


class Volatility(Enum):
    """Volatility states."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    
    @classmethod
    def all(cls) -> List[str]:
        """Get all volatility values."""
        return [v.value for v in cls]
    
    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if volatility value is valid."""
        return value in cls.all()


class Liquidity(Enum):
    """Liquidity states."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    
    @classmethod
    def all(cls) -> List[str]:
        """Get all liquidity values."""
        return [l.value for l in cls]
    
    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if liquidity value is valid."""
        return value in cls.all()


class Volume(Enum):
    """Volume states."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    
    @classmethod
    def all(cls) -> List[str]:
        """Get all volume values."""
        return [v.value for v in cls]
    
    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if volume value is valid."""
        return value in cls.all()


class SectorStrength(Enum):
    """Sector strength states."""
    WEAK = "weak"
    NEUTRAL = "neutral"
    STRONG = "strong"
    
    @classmethod
    def all(cls) -> List[str]:
        """Get all sector strength values."""
        return [s.value for s in cls]
    
    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if sector strength value is valid."""
        return value in cls.all()


class MarketBreadth(Enum):
    """Market breadth states."""
    WEAK = "weak"
    NEUTRAL = "neutral"
    STRONG = "strong"
    
    @classmethod
    def all(cls) -> List[str]:
        """Get all market breadth values."""
        return [b.value for b in cls]
    
    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if market breadth value is valid."""
        return value in cls.all()


class OptionsSentiment(Enum):
    """Options sentiment states."""
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    
    @classmethod
    def all(cls) -> List[str]:
        """Get all options sentiment values."""
        return [o.value for o in cls]
    
    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if options sentiment value is valid."""
        return value in cls.all()


class MarketStateValidator:
    """Validates market state values."""
    
    @staticmethod
    def validate_trend(value: str) -> bool:
        """Validate trend value."""
        return Trend.is_valid(value)
    
    @staticmethod
    def validate_volatility(value: str) -> bool:
        """Validate volatility value."""
        return Volatility.is_valid(value)
    
    @staticmethod
    def validate_liquidity(value: str) -> bool:
        """Validate liquidity value."""
        return Liquidity.is_valid(value)
    
    @staticmethod
    def validate_volume(value: str) -> bool:
        """Validate volume value."""
        return Volume.is_valid(value)
    
    @staticmethod
    def validate_sector_strength(value: str) -> bool:
        """Validate sector strength value."""
        return SectorStrength.is_valid(value)
    
    @staticmethod
    def validate_market_breadth(value: str) -> bool:
        """Validate market breadth value."""
        return MarketBreadth.is_valid(value)
    
    @staticmethod
    def validate_options_sentiment(value: str) -> bool:
        """Validate options sentiment value."""
        return OptionsSentiment.is_valid(value)
    
    @staticmethod
    def validate_all(state_dict: dict) -> tuple[bool, List[str]]:
        """
        Validate all market state values.
        
        Args:
            state_dict: Dictionary of state values
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if "trend" in state_dict and not MarketStateValidator.validate_trend(state_dict["trend"]):
            errors.append(f"Invalid trend: {state_dict['trend']}")
        
        if "volatility" in state_dict and not MarketStateValidator.validate_volatility(state_dict["volatility"]):
            errors.append(f"Invalid volatility: {state_dict['volatility']}")
        
        if "liquidity" in state_dict and not MarketStateValidator.validate_liquidity(state_dict["liquidity"]):
            errors.append(f"Invalid liquidity: {state_dict['liquidity']}")
        
        if "volume" in state_dict and not MarketStateValidator.validate_volume(state_dict["volume"]):
            errors.append(f"Invalid volume: {state_dict['volume']}")
        
        if "sector_strength" in state_dict and not MarketStateValidator.validate_sector_strength(state_dict["sector_strength"]):
            errors.append(f"Invalid sector_strength: {state_dict['sector_strength']}")
        
        if "market_breadth" in state_dict and not MarketStateValidator.validate_market_breadth(state_dict["market_breadth"]):
            errors.append(f"Invalid market_breadth: {state_dict['market_breadth']}")
        
        if "options_sentiment" in state_dict and not MarketStateValidator.validate_options_sentiment(state_dict["options_sentiment"]):
            errors.append(f"Invalid options_sentiment: {state_dict['options_sentiment']}")
        
        return len(errors) == 0, errors
