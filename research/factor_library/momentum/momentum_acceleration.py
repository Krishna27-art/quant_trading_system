"""
Momentum Acceleration Factor

Momentum acceleration factor measuring the rate of change of momentum.
Identifies accelerating or decelerating trends.
"""

import pandas as pd
import numpy as np

from research.factor_library.base_factor import BaseFactor


class MomentumAccelerationFactor(BaseFactor):
    """
    Momentum acceleration factor.
    
    Measures the second derivative of price (acceleration of momentum).
    Positive acceleration indicates strengthening trend, negative indicates weakening.
    """
    
    @property
    def name(self) -> str:
        return "momentum_acceleration"
    
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
        return "Momentum acceleration factor measuring the rate of change of momentum to identify strengthening/weakening trends."
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute momentum acceleration factor.
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            Series with momentum acceleration values
        """
        close = df["close"]
        
        # Calculate first derivative (momentum)
        momentum = close.pct_change(self.lookback)
        
        # Calculate second derivative (acceleration)
        acceleration = momentum.diff()
        
        # Normalize by volatility
        volatility = close.pct_change().rolling(20).std()
        normalized_acceleration = acceleration / volatility.replace(0, np.nan)
        
        # Clip extreme values
        normalized_acceleration = normalized_acceleration.clip(-5, 5)
        
        return normalized_acceleration
