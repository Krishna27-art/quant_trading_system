"""
MACD Momentum Factor

Moving Average Convergence Divergence momentum factor.
Measures trend strength and momentum changes.
"""

import pandas as pd
import numpy as np

from research.factor_library.base_factor import BaseFactor


class MACDFactor(BaseFactor):
    """
    MACD (Moving Average Convergence Divergence) momentum factor.
    
    MACD shows the relationship between two moving averages of prices.
    The MACD is calculated by subtracting the 26-period EMA from the 12-period EMA.
    """
    
    @property
    def name(self) -> str:
        return "macd_momentum"
    
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
        return 26
    
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
        return "MACD momentum factor measuring trend strength and momentum changes."
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute MACD momentum factor.
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            Series with MACD histogram values
        """
        close = df["close"]
        
        # Calculate EMAs
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = ema_12 - ema_26
        
        # Calculate signal line (9-period EMA of MACD)
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        # Calculate MACD histogram
        macd_histogram = macd_line - signal_line
        
        # Normalize by recent volatility for stationarity
        volatility = close.pct_change().rolling(20).std()
        normalized_histogram = macd_histogram / volatility.replace(0, np.nan)
        
        # Clip extreme values
        normalized_histogram = normalized_histogram.clip(-5, 5)
        
        return normalized_histogram
