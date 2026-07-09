"""
Volume Imbalance Factor

Volume imbalance factor measuring buying vs. selling pressure.
Compares up volume to down volume to identify directional pressure.
"""

import pandas as pd
import numpy as np

from research.factor_library.base_factor import BaseFactor


class VolumeImbalanceFactor(BaseFactor):
    """
    Volume imbalance factor.
    
    Measures the imbalance between buying volume (up days) and selling volume (down days).
    Positive values indicate buying pressure, negative indicate selling pressure.
    """
    
    @property
    def name(self) -> str:
        return "volume_imbalance"
    
    @property
    def category(self) -> str:
        return "volume"
    
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
        return ["close", "volume"]
    
    @property
    def uses_future_data(self) -> bool:
        return False
    
    @property
    def description(self) -> str:
        return "Volume imbalance factor measuring buying vs. selling pressure by comparing up volume to down volume."
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute volume imbalance factor.
        
        Args:
            df: DataFrame with 'close' and 'volume' columns
            
        Returns:
            Series with volume imbalance values
        """
        close = df["close"]
        volume = df["volume"]
        
        # Calculate price change
        price_change = close.diff()
        
        # Separate up and down volume
        up_volume = volume.where(price_change > 0, 0)
        down_volume = volume.where(price_change < 0, 0)
        
        # Calculate rolling sums
        up_volume_sum = up_volume.rolling(self.lookback).sum()
        down_volume_sum = down_volume.rolling(self.lookback).sum()
        
        # Calculate volume imbalance ratio
        total_volume = up_volume_sum + down_volume_sum
        imbalance_ratio = (up_volume_sum - down_volume_sum) / total_volume.replace(0, np.nan)
        
        # Z-score normalization
        imbalance_zscore = (imbalance_ratio - imbalance_ratio.rolling(60).mean()) / imbalance_ratio.rolling(60).std()
        
        # Clip extreme values
        imbalance_zscore = imbalance_zscore.clip(-5, 5)
        
        return imbalance_zscore
