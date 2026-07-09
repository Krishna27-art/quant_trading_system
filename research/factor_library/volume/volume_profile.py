"""
Volume Profile Factor

Volume profile factor analyzing volume distribution across price levels.
Identifies support/resistance levels based on volume concentration.
"""

import pandas as pd
import numpy as np

from research.factor_library.base_factor import BaseFactor


class VolumeProfileFactor(BaseFactor):
    """
    Volume profile factor.
    
    Analyzes the distribution of volume across price levels over the lookback period.
    Identifies whether current price is near high-volume areas (support/resistance).
    """
    
    @property
    def name(self) -> str:
        return "volume_profile"
    
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
        return "Volume profile factor analyzing volume distribution across price levels to identify support/resistance."
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute volume profile factor.
        
        Args:
            df: DataFrame with 'close' and 'volume' columns
            
        Returns:
            Series with volume profile values
        """
        close = df["close"]
        volume = df["volume"]
        
        # Calculate price percentiles for volume profile buckets
        profile_signal = pd.Series(index=close.index, dtype=float)
        
        for i in range(self.lookback, len(close)):
            window_close = close.iloc[i - self.lookback : i + 1]
            window_volume = volume.iloc[i - self.lookback : i + 1]
            
            # Create price buckets (percentiles)
            price_buckets = np.percentile(window_close, [20, 40, 60, 80])
            
            # Calculate volume in each bucket
            current_price = close.iloc[i]
            
            # Determine which bucket current price is in
            if current_price <= price_buckets[0]:
                bucket_volume = window_volume[window_close <= price_buckets[0]].sum()
                total_volume = window_volume.sum()
            elif current_price <= price_buckets[1]:
                bucket_volume = window_volume[
                    (window_close > price_buckets[0]) & (window_close <= price_buckets[1])
                ].sum()
                total_volume = window_volume.sum()
            elif current_price <= price_buckets[2]:
                bucket_volume = window_volume[
                    (window_close > price_buckets[1]) & (window_close <= price_buckets[2])
                ].sum()
                total_volume = window_volume.sum()
            elif current_price <= price_buckets[3]:
                bucket_volume = window_volume[
                    (window_close > price_buckets[2]) & (window_close <= price_buckets[3])
                ].sum()
                total_volume = window_volume.sum()
            else:
                bucket_volume = window_volume[window_close > price_buckets[3]].sum()
                total_volume = window_volume.sum()
            
            # Calculate volume concentration in current bucket
            volume_concentration = bucket_volume / total_volume if total_volume > 0 else 0
            
            # High concentration indicates near support/resistance
            profile_signal.iloc[i] = volume_concentration
        
        # Z-score normalization
        profile_zscore = (profile_signal - profile_signal.rolling(60).mean()) / profile_signal.rolling(60).std()
        
        # Clip extreme values
        profile_zscore = profile_zscore.clip(-5, 5)
        
        return profile_zscore
