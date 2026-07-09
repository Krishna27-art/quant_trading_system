"""
Volatility Kurtosis Factor

Volatility kurtosis factor measuring tail thickness of returns distribution.
Identifies fat-tail risk and extreme event probability.
"""

import pandas as pd
import numpy as np

from research.factor_library.base_factor import BaseFactor


class VolatilityKurtosisFactor(BaseFactor):
    """
    Volatility kurtosis factor.
    
    Measures the thickness of the tails in the returns distribution.
    High kurtosis indicates fat tails (more extreme events than normal distribution).
    Low kurtosis indicates thin tails (fewer extreme events).
    """
    
    @property
    def name(self) -> str:
        return "volatility_kurtosis"
    
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
        return "Volatility kurtosis factor measuring tail thickness of returns distribution to identify fat-tail risk and extreme event probability."
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute volatility kurtosis factor.
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            Series with volatility kurtosis values
        """
        close = df["close"]
        
        # Calculate returns
        returns = close.pct_change()
        
        # Calculate rolling kurtosis
        rolling_kurtosis = returns.rolling(self.lookback).kurt()
        
        # Normalize by historical distribution
        kurtosis_zscore = (rolling_kurtosis - rolling_kurtosis.rolling(60).mean()) / rolling_kurtosis.rolling(60).std()
        
        # Clip extreme values
        kurtosis_zscore = kurtosis_zscore.clip(-5, 5)
        
        return kurtosis_zscore
