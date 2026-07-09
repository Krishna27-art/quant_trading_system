"""
VWAP Distance Factor

Volume Weighted Average Price distance factor.
Measures how far current price is from VWAP, indicating overbought/oversold conditions.
"""

import pandas as pd
import numpy as np

from research.factor_library.base_factor import BaseFactor


class VWAPDistanceFactor(BaseFactor):
    """
    VWAP (Volume Weighted Average Price) distance factor.
    
    Measures the distance between current price and VWAP.
    Price above VWAP indicates bullishness, below indicates bearishness.
    """
    
    @property
    def name(self) -> str:
        return "vwap_distance"
    
    @property
    def category(self) -> str:
        return "volume"
    
    @property
    def version(self) -> str:
        return "1.0"
    
    @property
    def author(self) -> str:
        return "Krishna"
    
    @property
    def timeframe(self) -> str:
        return "INTRADAY"
    
    @property
    def lookback(self) -> int:
        return 20
    
    @property
    def prediction_horizon(self) -> int:
        return 5
    
    @property
    def required_columns(self) -> list:
        return ["high", "low", "close", "volume"]
    
    @property
    def uses_future_data(self) -> bool:
        return False
    
    @property
    def description(self) -> str:
        return "VWAP distance factor measuring how far current price is from volume-weighted average price."
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute VWAP distance factor.
        
        Args:
            df: DataFrame with 'high', 'low', 'close', 'volume' columns
            
        Returns:
            Series with VWAP distance values
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]
        volume = df["volume"]
        
        # Calculate typical price
        typical_price = (high + low + close) / 3
        
        # Calculate VWAP (cumulative)
        cumulative_tp_volume = (typical_price * volume).cumsum()
        cumulative_volume = volume.cumsum()
        vwap = cumulative_tp_volume / cumulative_volume.replace(0, np.nan)
        
        # Calculate distance from VWAP
        vwap_distance = (close - vwap) / vwap
        
        # Normalize by recent volatility
        volatility = close.pct_change().rolling(20).std()
        normalized_distance = vwap_distance / volatility.replace(0, np.nan)
        
        # Clip extreme values
        normalized_distance = normalized_distance.clip(-5, 5)
        
        return normalized_distance
