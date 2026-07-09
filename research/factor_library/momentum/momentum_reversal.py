"""
Momentum Reversal Factor

Momentum reversal factor identifying potential reversals after extreme momentum.
Captures the idea that extreme momentum often reverses.
"""

import pandas as pd
import numpy as np

from research.factor_library.base_factor import BaseFactor


class MomentumReversalFactor(BaseFactor):
    """
    Momentum reversal factor.
    
    Identifies potential reversals after extreme momentum readings.
    Based on the mean reversion principle - extreme momentum tends to reverse.
    """
    
    @property
    def name(self) -> str:
        return "momentum_reversal"
    
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
        return "Momentum reversal factor identifying potential reversals after extreme momentum readings."
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute momentum reversal factor.
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            Series with momentum reversal values
        """
        close = df["close"]
        
        # Calculate momentum
        momentum = close.pct_change(self.lookback)
        
        # Calculate z-score of momentum
        momentum_mean = momentum.rolling(60).mean()
        momentum_std = momentum.rolling(60).std()
        momentum_zscore = (momentum - momentum_mean) / momentum_std.replace(0, np.nan)
        
        # Reversal signal: negative of extreme z-score
        # If momentum is extremely positive (z > 2), signal is negative (expect reversal)
        # If momentum is extremely negative (z < -2), signal is positive (expect reversal)
        reversal_signal = -momentum_zscore
        
        # Clip to reasonable range
        reversal_signal = reversal_signal.clip(-3, 3)
        
        return reversal_signal
