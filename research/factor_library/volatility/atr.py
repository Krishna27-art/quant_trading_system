"""
ATR Volatility Factor

Average True Range volatility factor.
Measures market volatility and potential range expansion.
"""

import pandas as pd
import numpy as np

from research.factor_library.base_factor import BaseFactor


class ATRFactor(BaseFactor):
    """
    ATR (Average True Range) volatility factor.
    
    ATR measures market volatility by calculating the average of true ranges.
    High ATR indicates high volatility, low ATR indicates low volatility.
    """
    
    @property
    def name(self) -> str:
        return "atr_volatility"
    
    @property
    def category(self) -> str:
        return "volatility"
    
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
        return "ATR volatility factor measuring market volatility and potential range expansion."
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute ATR volatility factor.
        
        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            
        Returns:
            Series with normalized ATR values
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        # Calculate True Range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(self.lookback).mean()
        
        # Normalize by price for stationarity
        normalized_atr = atr / close
        
        # Z-score normalization
        atr_zscore = (normalized_atr - normalized_atr.rolling(60).mean()) / normalized_atr.rolling(60).std()
        
        # Clip extreme values
        atr_zscore = atr_zscore.clip(-5, 5)
        
        return atr_zscore
