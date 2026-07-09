"""
Parkinson Volatility Factor

Parkinson volatility estimator using high-low range.
More efficient than standard deviation for capturing volatility.
"""

import pandas as pd
import numpy as np

from research.factor_library.base_factor import BaseFactor


class ParkinsonVolatilityFactor(BaseFactor):
    """
    Parkinson volatility factor.
    
    Uses the high-low range to estimate volatility, which is more efficient
    than standard deviation at capturing true volatility, especially for
    continuous price processes.
    """
    
    @property
    def name(self) -> str:
        return "parkinson_volatility"
    
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
        return ["high", "low"]
    
    @property
    def uses_future_data(self) -> bool:
        return False
    
    @property
    def description(self) -> str:
        return "Parkinson volatility estimator using high-low range for more efficient volatility measurement."
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Parkinson volatility factor.
        
        Args:
            df: DataFrame with 'high' and 'low' columns
            
        Returns:
            Series with Parkinson volatility values
        """
        high = df["high"]
        low = df["low"]
        
        # Calculate log high-low ratio
        hl_ratio = np.log(high / low)
        
        # Calculate Parkinson volatility estimator
        parkinson_vol = np.sqrt((hl_ratio ** 2) / (4 * np.log(2)))
        
        # Smooth with rolling window
        parkinson_vol_smooth = parkinson_vol.rolling(self.lookback).mean()
        
        # Normalize by price
        close = df["close"] if "close" in df.columns else (high + low) / 2
        normalized_vol = parkinson_vol_smooth / close
        
        # Z-score normalization
        vol_zscore = (normalized_vol - normalized_vol.rolling(60).mean()) / normalized_vol.rolling(60).std()
        
        # Clip extreme values
        vol_zscore = vol_zscore.clip(-5, 5)
        
        return vol_zscore
