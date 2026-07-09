"""
Price Momentum Factor

Simple price momentum factor based on historical returns.
Measures the rate of price change over a specified period.
"""

import pandas as pd
import numpy as np

from research.factor_library.base_factor import BaseFactor


class PriceMomentumFactor(BaseFactor):
    """
    Simple price momentum factor.
    
    Measures the rate of price change over a specified period.
    This is the most basic momentum factor but still widely used.
    """
    
    @property
    def name(self) -> str:
        return "price_momentum"
    
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
        return "Simple price momentum factor measuring rate of price change over specified period."
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute price momentum factor.
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            Series with price momentum values
        """
        close = df["close"]
        
        # Calculate price change over lookback period
        price_change = close.pct_change(self.lookback)
        
        # Normalize by volatility for stationarity
        volatility = close.pct_change().rolling(20).std()
        normalized_momentum = price_change / volatility.replace(0, np.nan)
        
        # Clip extreme values
        normalized_momentum = normalized_momentum.clip(-5, 5)
        
        return normalized_momentum
