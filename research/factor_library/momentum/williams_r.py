"""
Williams %R Momentum Factor

Williams %R momentum factor measuring overbought/oversold conditions.
Similar to Stochastic but with inverted scale.
"""

import pandas as pd
import numpy as np

from research.factor_library.base_factor import BaseFactor


class WilliamsRFactor(BaseFactor):
    """
    Williams %R momentum factor.
    
    Measures the current closing price relative to the high-low range
    over a lookback period. Similar to Stochastic but with inverted scale
    (0 to -100 instead of 0 to 100).
    """
    
    @property
    def name(self) -> str:
        return "williams_r"
    
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
        return 14
    
    @property
    def prediction_horizon(self) -> int:
        return 5
    
    @property
    def required_columns(self) -> list:
        return ["high", "low", "close"]
    
    @property
    def uses_future_data(self) -> bool:
        return False
    
    @property
    def description(self) -> str:
        return "Williams %R momentum factor measuring overbought/oversold conditions with inverted scale."
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Williams %R momentum factor.
        
        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            
        Returns:
            Series with Williams %R values (normalized to -1 to 1)
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        # Calculate rolling high and low
        rolling_high = high.rolling(self.lookback).max()
        rolling_low = low.rolling(self.lookback).min()
        
        # Calculate Williams %R
        williams_r = ((rolling_high - close) / (rolling_high - rolling_low).replace(0, np.nan)) * -100
        
        # Normalize to -1 to 1 range (-50 becomes 0)
        normalized_wr = (williams_r + 50) / 50
        
        return normalized_wr
