"""
Stochastic Momentum Factor

Stochastic oscillator momentum factor.
Measures momentum relative to price range over a specified period.
"""

import pandas as pd
import numpy as np

from research.factor_library.base_factor import BaseFactor


class StochasticMomentumFactor(BaseFactor):
    """
    Stochastic momentum factor.
    
    Measures the current price relative to the high-low range over a lookback period.
    Values near 100 indicate overbought, near 0 indicate oversold.
    """
    
    @property
    def name(self) -> str:
        return "stochastic_momentum"
    
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
        return "Stochastic oscillator momentum factor measuring price relative to high-low range."
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute stochastic momentum factor.
        
        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            
        Returns:
            Series with stochastic momentum values (normalized to -1 to 1)
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        # Calculate rolling high and low
        rolling_high = high.rolling(self.lookback).max()
        rolling_low = low.rolling(self.lookback).min()
        
        # Calculate %K (fast stochastic)
        k_percent = ((close - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)) * 100
        
        # Calculate %D (slow stochastic - 3-period SMA of %K)
        d_percent = k_percent.rolling(3).mean()
        
        # Normalize to -1 to 1 range (50 becomes 0)
        normalized_stoch = (d_percent - 50) / 50
        
        return normalized_stoch
