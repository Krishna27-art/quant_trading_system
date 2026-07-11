"""
ATR (Average True Range)

Volatility indicator that measures market volatility by decomposing the entire range of an asset price.
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


class ATR(BaseFeature):
    """
    Average True Range (ATR).
    
    ATR is a volatility indicator that measures the volatility of price changes.
    It does not indicate price direction, only the degree of price volatility.
    
    Calculation:
    1. True Range = max(High - Low, |High - Previous Close|, |Low - Previous Close|)
    2. ATR = 14-period moving average of True Range (typically using Wilder's smoothing)
    
    Higher ATR values indicate higher volatility, lower values indicate lower volatility.
    """
    
    def _define_metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            feature_name="ATR14",
            description=(
                "Average True Range (ATR) - Volatility indicator that measures market volatility. "
                "ATR does not indicate price direction, only the degree of price volatility. "
                "Higher ATR values indicate higher volatility, useful for setting stop-loss levels "
                "and position sizing."
            ),
            category=FeatureCategory.TECHNICAL,
            timeframe=Timeframe.DAY_1,
            required_columns=["High", "Low", "Close"],
            output_range="0+",
            version="1.0",
            author="system",
            computation_method=(
                "True Range = max(High - Low, |High - Previous Close|, |Low - Previous Close|)\n"
                "ATR = 14-period EMA of True Range (Wilder's smoothing)"
            ),
            assumptions=(
                "1. Volatility tends to cluster\n"
                "2. High volatility periods are followed by high volatility\n"
                "3. True range captures gaps and limit moves"
            ),
            limitations=(
                "1. Does not indicate price direction\n"
                "2. Lagging indicator - reacts after volatility changes\n"
                "3. Can be skewed by single extreme price moves"
            ),
            references="J. Welles Wilder Jr., New Concepts in Technical Trading Systems (1978)"
        )
    
    def compute(self, data: pd.DataFrame) -> FeatureResult:
        """
        Compute ATR.
        
        Args:
            data: DataFrame with 'High', 'Low', 'Close' columns
            
        Returns:
            FeatureResult with ATR values
        """
        import time
        start_time = time.time()
        
        high = data['High']
        low = data['Low']
        close = data['Close']
        period = 14
        
        # Calculate True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR using Wilder's smoothing (EMA)
        atr = true_range.ewm(alpha=1/period, adjust=False).mean()
        
        computation_time = (time.time() - start_time) * 1000
        
        return FeatureResult(
            feature_name=self.metadata.feature_name,
            values=atr,
            metadata=self.metadata,
            computation_time_ms=computation_time,
            warnings=[]
        )
