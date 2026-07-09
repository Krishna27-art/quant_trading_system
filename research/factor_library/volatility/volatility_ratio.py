"""
Volatility Ratio Factor

Volatility ratio factor comparing short-term to long-term volatility.
Identifies volatility expansion and contraction regimes.
"""

import pandas as pd
import numpy as np

from research.factor_library.base_factor import BaseFactor


class VolatilityRatioFactor(BaseFactor):
    """
    Volatility ratio factor.
    
    Compares short-term volatility to long-term volatility.
    Values > 1 indicate volatility expansion, < 1 indicate contraction.
    """
    
    @property
    def name(self) -> str:
        return "volatility_ratio"
    
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
        return "Volatility ratio factor comparing short-term to long-term volatility to identify expansion/contraction regimes."
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute volatility ratio factor.
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            Series with volatility ratio values
        """
        close = df["close"]
        
        # Calculate returns
        returns = close.pct_change()
        
        # Calculate short-term volatility (5-day)
        short_vol = returns.rolling(5).std()
        
        # Calculate long-term volatility (20-day)
        long_vol = returns.rolling(20).std()
        
        # Calculate ratio
        vol_ratio = short_vol / long_vol.replace(0, np.nan)
        
        # Log transform for normalization
        log_ratio = np.log(vol_ratio)
        
        # Z-score normalization
        vol_ratio_zscore = (log_ratio - log_ratio.rolling(60).mean()) / log_ratio.rolling(60).std()
        
        # Clip extreme values
        vol_ratio_zscore = vol_ratio_zscore.clip(-5, 5)
        
        return vol_ratio_zscore
