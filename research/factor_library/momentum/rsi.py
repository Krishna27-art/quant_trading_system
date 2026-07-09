"""
RSI Momentum Factor

Relative Strength Index momentum factor.
Measures overbought/oversold conditions and potential reversals.
"""

import pandas as pd
import numpy as np

from research.factor_library.base_factor import BaseFactor


class RSIFactor(BaseFactor):
    """
    RSI (Relative Strength Index) momentum factor.
    
    RSI measures the magnitude of recent price changes to evaluate overbought
    or oversold conditions. Values above 70 indicate overbought, below 30 indicate
    oversold.
    """
    
    @property
    def name(self) -> str:
        return "rsi_momentum"
    
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
        return ["close"]
    
    @property
    def uses_future_data(self) -> bool:
        return False
    
    @property
    def description(self) -> str:
        return "RSI momentum factor measuring overbought/oversold conditions for potential reversals."
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute RSI momentum factor.
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            Series with RSI values (normalized to -1 to 1 range)
        """
        close = df["close"]
        
        # Calculate price changes
        delta = close.diff()
        
        # Separate gains and losses
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        # Calculate average gain and loss over lookback period
        avg_gain = gain.rolling(window=self.lookback).mean()
        avg_loss = loss.rolling(window=self.lookback).mean()
        
        # Calculate RSI
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        # Normalize to -1 to 1 range (50 becomes 0)
        normalized_rsi = (rsi - 50) / 50
        
        return normalized_rsi
