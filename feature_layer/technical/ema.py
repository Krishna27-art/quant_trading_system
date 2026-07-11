"""
EMA (Exponential Moving Average)

Moving average that places greater weight on more recent data points.
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


class EMA(BaseFeature):
    """
    Exponential Moving Average (EMA).
    
    EMA is a type of moving average that places greater weight and significance
    on the most recent data points. This makes it more responsive to new information
    compared to a simple moving average (SMA).
    
    Calculation:
    EMA = (Close - Previous EMA) * (2 / (period + 1)) + Previous EMA
    
    This implementation uses the standard pandas ewm() function.
    """
    
    def __init__(self, period: int = 20):
        """
        Initialize EMA with specified period.
        
        Args:
            period: Number of periods for EMA calculation (default: 20)
        """
        self.period = period
        super().__init__()
    
    def _define_metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            feature_name=f"EMA{self.period}",
            description=(
                f"Exponential Moving Average (EMA{self.period}) - Moving average that places "
                f"greater weight on more recent data points. More responsive to new information "
                f"compared to simple moving average. Commonly used for trend identification and "
                f"dynamic support/resistance levels."
            ),
            category=FeatureCategory.TECHNICAL,
            timeframe=Timeframe.DAY_1,
            required_columns=["Close"],
            output_range="Price range",
            version="1.0",
            author="system",
            computation_method=(
                f"EMA = (Close - Previous EMA) * (2 / ({self.period} + 1)) + Previous EMA\n"
                f"Using exponential weighting with alpha = 2 / ({self.period} + 1)"
            ),
            assumptions=(
                "1. Recent price action is more relevant than older data\n"
                "2. Trends persist and can be identified by moving averages\n"
                "3. EMA reduces lag compared to SMA"
            ),
            limitations=(
                "1. Still a lagging indicator\n"
                "2. Can be whipsawed in ranging markets\n"
                "3. Period selection is subjective and affects results"
            ),
            references="Standard technical analysis literature"
        )
    
    def compute(self, data: pd.DataFrame) -> FeatureResult:
        """
        Compute EMA.
        
        Args:
            data: DataFrame with 'Close' column
            
        Returns:
            FeatureResult with EMA values
        """
        import time
        start_time = time.time()
        
        close = data['Close']
        
        # Calculate EMA
        ema = close.ewm(span=self.period, adjust=False).mean()
        
        computation_time = (time.time() - start_time) * 1000
        
        return FeatureResult(
            feature_name=self.metadata.feature_name,
            values=ema,
            metadata=self.metadata,
            computation_time_ms=computation_time,
            warnings=[]
        )


class EMA9(EMA):
    """9-period EMA - commonly used for short-term trends."""
    def __init__(self):
        super().__init__(period=9)


class EMA20(EMA):
    """20-period EMA - commonly used for medium-term trends."""
    def __init__(self):
        super().__init__(period=20)


class EMA50(EMA):
    """50-period EMA - commonly used for intermediate-term trends."""
    def __init__(self):
        super().__init__(period=50)


class EMA200(EMA):
    """200-period EMA - commonly used for long-term trends."""
    def __init__(self):
        super().__init__(period=200)
