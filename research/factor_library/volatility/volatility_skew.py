"""
Volatility Skew Factor

Volatility skew factor measuring asymmetry in returns distribution.
Identifies tail risk and potential crash scenarios.
"""

import pandas as pd
import numpy as np

from research.factor_library.base_factor import BaseFactor


class VolatilitySkewFactor(BaseFactor):
    """
    Volatility skew factor.
    
    Measures the asymmetry in the returns distribution.
    Negative skew indicates more frequent large negative returns (crash risk).
    Positive skew indicates more frequent large positive returns.
    """
    
    @property
    def name(self) -> str:
        return "volatility_skew"
    
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
        return "Volatility skew factor measuring asymmetry in returns distribution to identify tail risk and crash scenarios."
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute volatility skew factor.
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            Series with volatility skew values
        """
        close = df["close"]
        
        # Calculate returns
        returns = close.pct_change()
        
        # Calculate rolling skewness
        rolling_skew = returns.rolling(self.lookback).skew()
        
        # Normalize by historical distribution
        skew_zscore = (rolling_skew - rolling_skew.rolling(60).mean()) / rolling_skew.rolling(60).std()
        
        # Clip extreme values
        skew_zscore = skew_zscore.clip(-5, 5)
        
        return skew_zscore
