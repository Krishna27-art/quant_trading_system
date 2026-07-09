"""
Parabolic SAR Trend Factor

Parabolic Stop and Reverse trend following factor.
Identifies trend direction and potential reversal points.
"""

import pandas as pd
import numpy as np

from research.factor_library.base_factor import BaseFactor


class ParabolicSARFactor(BaseFactor):
    """
    Parabolic SAR (Stop and Reverse) trend factor.
    
    Trend-following indicator that sets trailing stop-loss levels.
    When price crosses the SAR, it indicates a potential trend reversal.
    """
    
    @property
    def name(self) -> str:
        return "parabolic_sar"
    
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
        return 20
    
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
        return "Parabolic SAR trend factor identifying trend direction and potential reversal points."
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Parabolic SAR factor.
        
        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            
        Returns:
            Series with Parabolic SAR signal values
        """
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        
        n = len(close)
        sar = np.zeros(n)
        ep = np.zeros(n)  # Extreme point
        af = np.zeros(n)  # Acceleration factor
        trend = np.zeros(n)  # 1 for uptrend, -1 for downtrend
        
        # Initialize
        af[0] = 0.02
        max_af = 0.2
        if n > 1:
            trend[0] = 1 if close[0] > close[1] else -1
        else:
            trend[0] = 1
        
        for i in range(1, n):
            if trend[i-1] == 1:  # Uptrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                
                # Check for trend reversal
                if low[i] < sar[i]:
                    trend[i] = -1
                    sar[i] = ep[i-1]
                    ep[i] = low[i]
                    af[i] = 0.02
                else:
                    trend[i] = 1
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af[i] = min(af[i-1] + 0.02, max_af)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
                    
                    # Ensure SAR is below low
                    sar[i] = min(sar[i], low[i-1], low[i])
            
            else:  # Downtrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                
                # Check for trend reversal
                if high[i] > sar[i]:
                    trend[i] = 1
                    sar[i] = ep[i-1]
                    ep[i] = high[i]
                    af[i] = 0.02
                else:
                    trend[i] = -1
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af[i] = min(af[i-1] + 0.02, max_af)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
                    
                    # Ensure SAR is above high
                    sar[i] = max(sar[i], high[i-1], high[i])
        
        # Convert to signal: distance from price to SAR, normalized
        signal = (close - sar) / close
        
        return pd.Series(signal, index=df.index)
