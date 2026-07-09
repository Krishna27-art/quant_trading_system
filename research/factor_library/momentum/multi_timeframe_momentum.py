"""
Multi-Timeframe Momentum Factor

Combines momentum signals from multiple timeframes.
Captures trend strength across different time horizons.
"""

import pandas as pd
import numpy as np

from research.factor_library.base_factor import BaseFactor


class MultiTimeframeMomentumFactor(BaseFactor):
    """
    Multi-timeframe momentum factor.
    
    Combines momentum signals from short, medium, and long timeframes.
    This captures trend strength across different time horizons and can identify
    when momentum is consistent across multiple scales.
    """
    
    @property
    def name(self) -> str:
        return "multi_timeframe_momentum"
    
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
        return 60
    
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
        return "Multi-timeframe momentum factor combining signals from short, medium, and long timeframes."
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute multi-timeframe momentum factor.
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            Series with multi-timeframe momentum values
        """
        close = df["close"]
        
        # Calculate momentum at different timeframes
        # Short-term (10-day)
        short_momentum = close.pct_change(10)
        
        # Medium-term (20-day)
        medium_momentum = close.pct_change(20)
        
        # Long-term (60-day)
        long_momentum = close.pct_change(60)
        
        # Normalize each by its rolling volatility
        short_vol = close.pct_change().rolling(10).std()
        medium_vol = close.pct_change().rolling(20).std()
        long_vol = close.pct_change().rolling(60).std()
        
        short_norm = short_momentum / short_vol.replace(0, np.nan)
        medium_norm = medium_momentum / medium_vol.replace(0, np.nan)
        long_norm = long_momentum / long_vol.replace(0, np.nan)
        
        # Combine with weights (more weight to medium-term)
        combined = 0.25 * short_norm + 0.5 * medium_norm + 0.25 * long_norm
        
        # Clip extreme values
        combined = combined.clip(-5, 5)
        
        return combined
