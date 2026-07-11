"""
RSI (Relative Strength Index)

Momentum oscillator that measures the speed and change of price movements.
"""

import pandas as pd
import numpy as np
from datetime import datetime

from feature_layer.base_feature import (
    BaseFeature,
    FeatureMetadata,
    FeatureResult,
    FeatureCategory,
    Timeframe
)


class RSI(BaseFeature):
    """
    Relative Strength Index (RSI).
    
    RSI is a momentum oscillator that measures the speed and change of price movements.
    It oscillates between 0 and 100. Traditionally, RSI is considered overbought when
    above 70 and oversold when below 30.
    
    Calculation:
    1. Calculate price changes (delta)
    2. Separate gains and losses
    3. Calculate average gain and average loss over N periods
    4. Calculate RS = Average Gain / Average Loss
    5. RSI = 100 - (100 / (1 + RS))
    """
    
    def _define_metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            feature_name="RSI14",
            description=(
                "Relative Strength Index (RSI) - Momentum oscillator that measures "
                "the speed and change of price movements. RSI oscillates between 0 and 100. "
                "Values above 70 indicate overbought conditions, values below 30 indicate oversold."
            ),
            category=FeatureCategory.TECHNICAL,
            timeframe=Timeframe.DAY_1,
            required_columns=["Close"],
            output_range="0-100",
            version="1.0",
            author="system",
            computation_method=(
                "Wilder's Smoothing: Calculate average gain and loss over 14 periods, "
                "then compute RSI = 100 - (100 / (1 + RS)) where RS = AvgGain / AvgLoss"
            ),
            assumptions=(
                "1. Price trends persist in the short term\n"
                "2. Overbought/oversold conditions lead to reversals\n"
                "3. 14-period window captures relevant momentum"
            ),
            limitations=(
                "1. Can remain overbought/oversold for extended periods in strong trends\n"
                "2. False signals in ranging markets\n"
                "3. Lagging indicator - reacts after price moves"
            ),
            references="J. Welles Wilder Jr., New Concepts in Technical Trading Systems (1978)"
        )
    
    def compute(self, data: pd.DataFrame) -> FeatureResult:
        """
        Compute RSI.
        
        Args:
            data: DataFrame with 'Close' column
            
        Returns:
            FeatureResult with RSI values
        """
        import time
        start_time = time.time()
        
        close = data['Close']
        period = 14
        
        # Calculate price changes
        delta = close.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss using Wilder's smoothing
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        # Use exponential smoothing for Wilder's method
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        computation_time = (time.time() - start_time) * 1000
        
        return FeatureResult(
            feature_name=self.metadata.feature_name,
            values=rsi,
            metadata=self.metadata,
            computation_time_ms=computation_time,
            warnings=[]
        )
