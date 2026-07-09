"""
Market Snapshot

Captures and stores market context at a specific timestamp.
Provides methods to create snapshots from market data.
"""

from datetime import datetime
from typing import Optional

import pandas as pd

from research.interactions.market_context.market_context import MarketContext
from research.interactions.market_context.market_state import (
    Trend,
    Volatility,
    Liquidity,
    Volume,
    SectorStrength,
    MarketBreadth,
    OptionsSentiment,
    MarketStateValidator,
)
from utils.logger import get_logger

logger = get_logger("research.interactions.market_context")


class MarketSnapshot:
    """
    Captures and stores market context at a specific timestamp.
    
    Analyzes market data to determine:
    - Trend direction
    - Volatility level
    - Liquidity state
    - Volume level
    - Sector strength
    - Market breadth
    - Options sentiment
    """
    
    def __init__(
        self,
        volatility_thresholds: Optional[dict] = None,
        volume_thresholds: Optional[dict] = None,
    ):
        """
        Initialize market snapshot.
        
        Args:
            volatility_thresholds: Optional thresholds for volatility classification
            volume_thresholds: Optional thresholds for volume classification
        """
        self.volatility_thresholds = volatility_thresholds or {
            "low": 0.15,
            "medium": 0.25,
        }
        self.volume_thresholds = volume_thresholds or {
            "low": 0.8,
            "high": 1.2,
        }
        self._logger = get_logger("research.interactions.market_context")
    
    def create_snapshot(
        self,
        timestamp: datetime,
        nifty_data: pd.DataFrame,
        vix_data: Optional[float] = None,
        advance_decline: Optional[dict] = None,
        options_data: Optional[dict] = None,
        sector_data: Optional[dict] = None,
    ) -> MarketContext:
        """
        Create market context snapshot from market data.
        
        Args:
            timestamp: Timestamp of snapshot
            nifty_data: DataFrame with NIFTY OHLCV data
            vix_data: Optional India VIX level
            advance_decline: Optional dict with advance/decline data
            options_data: Optional dict with options data
            sector_data: Optional dict with sector performance data
            
        Returns:
            MarketContext
        """
        # Determine trend
        trend = self._classify_trend(nifty_data)
        
        # Determine volatility
        volatility = self._classify_volatility(nifty_data, vix_data)
        
        # Determine volume
        volume = self._classify_volume(nifty_data)
        
        # Determine liquidity (simplified)
        liquidity = self._classify_liquidity(nifty_data)
        
        # Determine sector strength
        sector_strength = self._classify_sector_strength(sector_data)
        
        # Determine market breadth
        market_breadth = self._classify_market_breadth(advance_decline)
        
        # Determine options sentiment
        options_sentiment = self._classify_options_sentiment(options_data)
        
        return MarketContext(
            timestamp=timestamp,
            trend=trend,
            volatility=volatility,
            liquidity=liquidity,
            volume=volume,
            sector_strength=sector_strength,
            market_breadth=market_breadth,
            options_sentiment=options_sentiment,
            vix_level=vix_data,
            nifty_level=nifty_data["close"].iloc[-1] if not nifty_data.empty else None,
        )
    
    def _classify_trend(self, data: pd.DataFrame) -> str:
        """Classify market trend from price data."""
        if data.empty or len(data) < 20:
            return Trend.SIDEWAYS.value
        
        close = data["close"]
        
        # Calculate moving averages
        ma_short = close.rolling(10).mean().iloc[-1]
        ma_long = close.rolling(50).mean().iloc[-1]
        current = close.iloc[-1]
        
        # Determine trend
        if current > ma_short > ma_long:
            return Trend.BULL.value
        elif current < ma_short < ma_long:
            return Trend.BEAR.value
        else:
            return Trend.SIDEWAYS.value
    
    def _classify_volatility(self, data: pd.DataFrame, vix: Optional[float]) -> str:
        """Classify volatility level."""
        if vix is not None:
            # Use VIX if available
            if vix < self.volatility_thresholds["low"]:
                return Volatility.LOW.value
            elif vix < self.volatility_thresholds["medium"]:
                return Volatility.MEDIUM.value
            else:
                return Volatility.HIGH.value
        
        # Fallback to historical volatility
        if data.empty or len(data) < 20:
            return Volatility.MEDIUM.value
        
        returns = data["close"].pct_change().dropna()
        volatility = returns.rolling(20).std().iloc[-1]
        
        if volatility < self.volatility_thresholds["low"]:
            return Volatility.LOW.value
        elif volatility < self.volatility_thresholds["medium"]:
            return Volatility.MEDIUM.value
        else:
            return Volatility.HIGH.value
    
    def _classify_volume(self, data: pd.DataFrame) -> str:
        """Classify volume level."""
        if data.empty or len(data) < 20:
            return Volume.NORMAL.value
        
        volume = data["volume"]
        avg_volume = volume.rolling(20).mean().iloc[-1]
        current_volume = volume.iloc[-1]
        
        ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        if ratio < self.volume_thresholds["low"]:
            return Volume.LOW.value
        elif ratio > self.volume_thresholds["high"]:
            return Volume.HIGH.value
        else:
            return Volume.NORMAL.value
    
    def _classify_liquidity(self, data: pd.DataFrame) -> str:
        """Classify liquidity state (simplified)."""
        # For now, use volume as proxy for liquidity
        volume_level = self._classify_volume(data)
        
        if volume_level == Volume.HIGH.value:
            return Liquidity.HIGH.value
        elif volume_level == Volume.LOW.value:
            return Liquidity.LOW.value
        else:
            return Liquidity.MEDIUM.value
    
    def _classify_sector_strength(self, sector_data: Optional[dict]) -> str:
        """Classify sector strength."""
        if sector_data is None:
            return SectorStrength.NEUTRAL.value
        
        # Simplified: use average sector performance
        performances = sector_data.values()
        avg_performance = sum(performances) / len(performances) if performances else 0
        
        if avg_performance > 0.02:
            return SectorStrength.STRONG.value
        elif avg_performance < -0.02:
            return SectorStrength.WEAK.value
        else:
            return SectorStrength.NEUTRAL.value
    
    def _classify_market_breadth(self, advance_decline: Optional[dict]) -> str:
        """Classify market breadth."""
        if advance_decline is None:
            return MarketBreadth.NEUTRAL.value
        
        advances = advance_decline.get("advances", 0)
        declines = advance_decline.get("declines", 0)
        
        if advances + declines == 0:
            return MarketBreadth.NEUTRAL.value
        
        ratio = advances / (advances + declines)
        
        if ratio > 0.6:
            return MarketBreadth.STRONG.value
        elif ratio < 0.4:
            return MarketBreadth.WEAK.value
        else:
            return MarketBreadth.NEUTRAL.value
    
    def _classify_options_sentiment(self, options_data: Optional[dict]) -> str:
        """Classify options sentiment."""
        if options_data is None:
            return OptionsSentiment.NEUTRAL.value
        
        # Simplified: use put-call ratio
        put_call_ratio = options_data.get("put_call_ratio", 1.0)
        
        if put_call_ratio < 0.8:
            return OptionsSentiment.BULLISH.value
        elif put_call_ratio > 1.2:
            return OptionsSentiment.BEARISH.value
        else:
            return OptionsSentiment.NEUTRAL.value


def create_market_snapshot(
    timestamp: datetime,
    nifty_data: pd.DataFrame,
    vix_data: Optional[float] = None,
    advance_decline: Optional[dict] = None,
    options_data: Optional[dict] = None,
    sector_data: Optional[dict] = None,
) -> MarketContext:
    """
    Convenience function to create market snapshot.
    
    Args:
        timestamp: Timestamp of snapshot
        nifty_data: DataFrame with NIFTY OHLCV data
        vix_data: Optional India VIX level
        advance_decline: Optional dict with advance/decline data
        options_data: Optional dict with options data
        sector_data: Optional dict with sector performance data
        
    Returns:
        MarketContext
    """
    snapshot = MarketSnapshot()
    return snapshot.create_snapshot(
        timestamp,
        nifty_data,
        vix_data,
        advance_decline,
        options_data,
        sector_data,
    )
