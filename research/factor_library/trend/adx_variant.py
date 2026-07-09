"""
ADX Variant Factor

Average Directional Index variant for trend strength measurement.
Improved version with smoother calculations and better normalization.
"""

import pandas as pd
import numpy as np

from research.factor_library.base_factor import BaseFactor


class ADXVariantFactor(BaseFactor):
    """
    ADX (Average Directional Index) variant factor.
    
    Measures the strength of a trend regardless of direction.
    This variant uses smoothed calculations for better signal quality.
    """
    
    @property
    def name(self) -> str:
        return "adx_variant"
    
    @property
    def category(self) -> str:
        return "trend"
    
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
        return "ADX variant factor measuring trend strength with smoothed calculations for better signal quality."
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute ADX variant factor.
        
        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            
        Returns:
            Series with ADX variant values (normalized to 0-1)
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
        
        # Calculate directional movements
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        # Calculate +DM and -DM
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        # Smooth using Wilder's smoothing (exponential with alpha = 1/n)
        alpha = 1.0 / self.lookback
        atr = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm_smooth / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm_smooth / atr.replace(0, np.nan))
        
        # Calculate DX
        di_diff = (plus_di - minus_di).abs()
        di_sum = plus_di + minus_di
        dx = 100 * (di_diff / di_sum.replace(0, np.nan))
        
        # Calculate ADX
        adx = dx.ewm(alpha=alpha, adjust=False).mean()
        
        # Normalize to 0-1 range
        normalized_adx = adx / 100
        
        return normalized_adx
