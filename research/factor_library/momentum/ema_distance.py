"""
EMA Distance Momentum Factor

Exponential Moving Average distance factor.
Measures price deviation from EMA for trend following.
"""

import pandas as pd
import numpy as np

from research.factor_library.base_factor import BaseFactor


class EMADistanceFactor(BaseFactor):
    """
    EMA Distance momentum factor.
    
    Measures how far the current price is from its exponential moving average.
    Positive values indicate price above EMA (bullish), negative below (bearish).
    """
    
    @property
    def name(self) -> str:
        return "ema_distance"
    
    @property
    def category(self) -> str:
        return "momentum"
    
    @property
    def version(self) -> str:
        return "1.0"
    
    @property
    def author(self) -> str:
        return "Krishna"
    
    @property
    def timeframe(self) -> str:
        return "SWING"
    
    @property
    def lookback(self) -> int:
        return 20
    
    @property
    def prediction_horizon(self) -> int:
        return 5
    
    @property
    def required_columns(self) -> list:
        return ["close"]
    
    @property
    def uses_future_data(self) -> bool:
        return False
    
    @property
    def description(self) -> str:
        return "EMA distance factor measuring price deviation from 20-period EMA for trend following."
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute EMA distance factor.
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            Series with normalized EMA distance values
        """
        close = df["close"]
        
        # Calculate 20-period EMA
        ema = close.ewm(span=20, adjust=False).mean()
        
        # Calculate distance from EMA
        distance = (close - ema) / ema
        
        # Normalize by recent volatility
        volatility = close.pct_change().rolling(20).std()
        normalized_distance = distance / volatility.replace(0, np.nan)
        
        # Clip extreme values
        normalized_distance = normalized_distance.clip(-5, 5)
        
        return normalized_distance
