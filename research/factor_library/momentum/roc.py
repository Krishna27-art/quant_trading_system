"""
ROC (Rate of Change) Momentum Factor

Rate of Change momentum factor measuring percentage price change.
Simple but effective momentum indicator.
"""

import pandas as pd
import numpy as np

from research.factor_library.base_factor import BaseFactor


class ROCFactor(BaseFactor):
    """
    Rate of Change (ROC) momentum factor.
    
    Measures the percentage change in price over a specified period.
    Similar to price momentum but expressed as percentage.
    """
    
    @property
    def name(self) -> str:
        return "roc_momentum"
    
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
        return "Rate of Change momentum factor measuring percentage price change over specified period."
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute ROC momentum factor.
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            Series with ROC values
        """
        close = df["close"]
        
        # Calculate ROC as percentage change
        roc = ((close - close.shift(self.lookback)) / close.shift(self.lookback)) * 100
        
        # Normalize by historical standard deviation
        roc_std = roc.rolling(60).std()
        normalized_roc = roc / roc_std.replace(0, np.nan)
        
        # Clip extreme values
        normalized_roc = normalized_roc.clip(-5, 5)
        
        return normalized_roc
