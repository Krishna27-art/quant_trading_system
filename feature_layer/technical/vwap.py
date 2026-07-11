"""
VWAP (Volume Weighted Average Price)

Trading benchmark that gives the average price a security has traded at throughout the day, based on both volume and price.
"""

import pandas as pd
import numpy as np
from datetime import datetime

from feature_layer.base_feature import (
    BaseFeature,
    FeatureMetadata,
    FeatureResult,
    FeatureCategory,
    Timeframe
)


class VWAP(BaseFeature):
    """
    Volume Weighted Average Price (VWAP).
    
    VWAP is the average price a security has traded at throughout the day,
    based on both volume and price. It is important because it provides
    a trading benchmark that represents the true average price paid for shares.
    
    Calculation:
    VWAP = Cumulative(Price * Volume) / Cumulative(Volume)
    where Price is typically the typical price: (High + Low + Close) / 3
    
    VWAP Distance feature measures how far the current price is from VWAP,
    expressed as a percentage.
    """
    
    def _define_metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            feature_name="VWAP",
            description=(
                "Volume Weighted Average Price (VWAP) - Trading benchmark that represents "
                "the true average price paid for shares. Calculated as the cumulative sum of "
                "price times volume divided by cumulative volume. Used by institutional traders "
                "to assess execution quality and identify fair value."
            ),
            category=FeatureCategory.TECHNICAL,
            timeframe=Timeframe.DAY_1,
            required_columns=["High", "Low", "Close", "Volume"],
            output_range="Price range",
            version="1.0",
            author="system",
            computation_method=(
                "Typical Price = (High + Low + Close) / 3\n"
                "VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)"
            ),
            assumptions=(
                "1. Volume-weighted price represents true average cost\n"
                "2. Price tends to revert to VWAP over time\n"
                "3. VWAP acts as dynamic support/resistance"
            ),
            limitations=(
                "1. Only meaningful for intraday analysis (resets daily)\n"
                "2. Less useful for low-volume stocks\n"
                "3. Can be skewed by large block trades"
            ),
            references="Standard institutional trading literature"
        )
    
    def compute(self, data: pd.DataFrame) -> FeatureResult:
        """
        Compute VWAP.
        
        Args:
            data: DataFrame with 'High', 'Low', 'Close', 'Volume' columns
            
        Returns:
            FeatureResult with VWAP values
        """
        import time
        start_time = time.time()
        
        high = data['High']
        low = data['Low']
        close = data['Close']
        volume = data['Volume']
        
        # Calculate typical price
        typical_price = (high + low + close) / 3
        
        # Calculate VWAP
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        computation_time = (time.time() - start_time) * 1000
        
        return FeatureResult(
            feature_name=self.metadata.feature_name,
            values=vwap,
            metadata=self.metadata,
            computation_time_ms=computation_time,
            warnings=[]
        )


class VWAPDistance(BaseFeature):
    """
    VWAP Distance - Percentage distance from VWAP.
    
    Measures how far the current price is from VWAP as a percentage.
    Positive values indicate price above VWAP, negative indicates below.
    """
    
    def _define_metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            feature_name="VWAP_Distance",
            description=(
                "VWAP Distance - Percentage distance of current price from VWAP. "
                "Positive values indicate price trading above VWAP (bullish), "
                "negative values indicate price trading below VWAP (bearish). "
                "Used to identify overextended moves and potential mean reversion."
            ),
            category=FeatureCategory.TECHNICAL,
            timeframe=Timeframe.DAY_1,
            required_columns=["High", "Low", "Close", "Volume"],
            output_range="Percentage",
            version="1.0",
            author="system",
            computation_method=(
                "VWAP_Distance = ((Close - VWAP) / VWAP) * 100"
            ),
            assumptions="Price tends to revert to VWAP mean",
            limitations="Less effective in strong trending markets",
            references="Institutional trading literature"
        )
    
    def compute(self, data: pd.DataFrame) -> FeatureResult:
        """
        Compute VWAP Distance.
        
        Args:
            data: DataFrame with 'High', 'Low', 'Close', 'Volume' columns
            
        Returns:
            FeatureResult with VWAP Distance values
        """
        import time
        start_time = time.time()
        
        high = data['High']
        low = data['Low']
        close = data['Close']
        volume = data['Volume']
        
        # Calculate VWAP
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        # Calculate distance percentage
        vwap_distance = ((close - vwap) / vwap) * 100
        
        computation_time = (time.time() - start_time) * 1000
        
        return FeatureResult(
            feature_name=self.metadata.feature_name,
            values=vwap_distance,
            metadata=self.metadata,
            computation_time_ms=computation_time,
            warnings=[]
        )
