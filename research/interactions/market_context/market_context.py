"""
Market Context

Describes the market state for every timestamp.
Serves as the "weather report" for the market.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from utils.logger import get_logger

logger = get_logger("research.interactions.market_context")


@dataclass
class MarketContext:
    """
    Complete market context at a specific timestamp.
    
    Describes:
    - Trend (Bull/Bear/Sideways)
    - Volatility (Low/Medium/High)
    - Liquidity (Low/Medium/High)
    - Volume (Low/Normal/High)
    - Sector StrengthWeak/Neutral/Strong)
    - Market Breadth (Weak/Neutral/Strong)
    - Options Sentiment (Bearish/Neutral/Bullish)
    """
    timestamp: datetime
    trend: str  # "bull", "bear", "sideways"
    volatility: str  # "low", "medium", "high"
    liquidity: str  # "low", "medium", "high"
    volume: str  # "low", "normal", "high"
    sector_strength: str  # "weak", "neutral", "strong"
    market_breadth: str  # "weak", "neutral", "strong"
    options_sentiment: str  # "bearish", "neutral", "bullish"
    
    # Optional additional context
    vix_level: Optional[float] = None
    nifty_level: Optional[float] = None
    banknifty_level: Optional[float] = None
    advance_decline_ratio: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "trend": self.trend,
            "volatility": self.volatility,
            "liquidity": self.liquidity,
            "volume": self.volume,
            "sector_strength": self.sector_strength,
            "market_breadth": self.market_breadth,
            "options_sentiment": self.options_sentiment,
            "vix_level": self.vix_level,
            "nifty_level": self.nifty_level,
            "banknifty_level": self.banknifty_level,
            "advance_decline_ratio": self.advance_decline_ratio,
        }
    
    def matches_condition(self, condition: dict) -> bool:
        """
        Check if context matches a condition.
        
        Args:
            condition: Dictionary of condition requirements
            
        Returns:
            True if context matches condition
        """
        for key, value in condition.items():
            if hasattr(self, key):
                if getattr(self, key) != value:
                    return False
        return True
    
    def is_bullish(self) -> bool:
        """Check if market is in bullish state."""
        return self.trend == "bull"
    
    def is_bearish(self) -> bool:
        """Check if market is in bearish state."""
        return self.trend == "bear"
    
    def is_high_volatility(self) -> bool:
        """Check if volatility is high."""
        return self.volatility == "high"
    
    def is_low_volatility(self) -> bool:
        """Check if volatility is low."""
        return self.volatility == "low"
    
    def is_strong_sector(self) -> bool:
        """Check if sector strength is strong."""
        return self.sector_strength == "strong"
    
    def is_bullish_options(self) -> bool:
        """Check if options sentiment is bullish."""
        return self.options_sentiment == "bullish"
    
    def get_summary(self) -> str:
        """Get human-readable summary of market context."""
        parts = []
        parts.append(f"Trend: {self.trend.upper()}")
        parts.append(f"Volatility: {self.volatility.upper()}")
        parts.append(f"Liquidity: {self.liquidity.upper()}")
        parts.append(f"Volume: {self.volume.upper()}")
        parts.append(f"Sector: {self.sector_strength.upper()}")
        parts.append(f"Breadth: {self.market_breadth.upper()}")
        parts.append(f"Options: {self.options_sentiment.upper()}")
        return " | ".join(parts)
