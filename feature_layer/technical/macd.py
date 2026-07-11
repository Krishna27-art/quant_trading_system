"""
MACD (Moving Average Convergence Divergence)

Trend-following momentum indicator that shows the relationship between two moving averages.
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


class MACD(BaseFeature):
    """
    Moving Average Convergence Divergence (MACD).
    
    MACD is a trend-following momentum indicator that shows the relationship between
    two moving averages of a security's price. The MACD is calculated by subtracting
    the 26-period EMA from the 12-period EMA.
    
    Calculation:
    1. Calculate 12-period EMA
    2. Calculate 26-period EMA
    3. MACD Line = 12-period EMA - 26-period EMA
    4. Signal Line = 9-period EMA of MACD Line
    5. Histogram = MACD Line - Signal Line
    
    This feature returns the MACD Line value.
    """
    
    def _define_metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            feature_name="MACD",
            description=(
                "Moving Average Convergence Divergence (MACD) - Trend-following momentum indicator. "
                "Shows the relationship between two moving averages of price. "
                "MACD Line = 12-period EMA - 26-period EMA. "
                "Positive MACD indicates upward momentum, negative indicates downward momentum."
            ),
            category=FeatureCategory.TECHNICAL,
            timeframe=Timeframe.DAY_1,
            required_columns=["Close"],
            output_range="Unbounded",
            version="1.0",
            author="system",
            computation_method=(
                "MACD Line = EMA(Close, 12) - EMA(Close, 26)\n"
                "Signal Line = EMA(MACD, 9)\n"
                "Histogram = MACD - Signal"
            ),
            assumptions=(
                "1. Moving averages smooth price data and reveal trends\n"
                "2. Convergence/divergence of MAs indicates momentum changes\n"
                "3. Crossovers signal trend reversals"
            ),
            limitations=(
                "1. Lagging indicator - reacts after price moves\n"
                "2. Whipsaws in ranging markets\n"
                "3. Standard parameters (12, 26, 9) may not suit all timeframes"
            ),
            references="Gerald Appel, The Moving Average Convergence Divergence Trading Method (1979)"
        )
    
    def compute(self, data: pd.DataFrame) -> FeatureResult:
        """
        Compute MACD Line.
        
        Args:
            data: DataFrame with 'Close' column
            
        Returns:
            FeatureResult with MACD Line values
        """
        import time
        start_time = time.time()
        
        close = data['Close']
        
        # Calculate EMAs
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        
        # Calculate MACD Line
        macd_line = ema_12 - ema_26
        
        computation_time = (time.time() - start_time) * 1000
        
        return FeatureResult(
            feature_name=self.metadata.feature_name,
            values=macd_line,
            metadata=self.metadata,
            computation_time_ms=computation_time,
            warnings=[]
        )


class MACDSignal(BaseFeature):
    """MACD Signal Line (9-period EMA of MACD)."""
    
    def _define_metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            feature_name="MACD_Signal",
            description="MACD Signal Line - 9-period EMA of the MACD Line. Used to generate trading signals when it crosses the MACD Line.",
            category=FeatureCategory.TECHNICAL,
            timeframe=Timeframe.DAY_1,
            required_columns=["Close"],
            output_range="Unbounded",
            version="1.0",
            author="system",
            computation_method="Signal Line = EMA(MACD, 9)",
            assumptions="Signal line smooths MACD to reduce false signals",
            limitations="Lagging indicator, reacts after MACD moves",
            references="Gerald Appel"
        )
    
    def compute(self, data: pd.DataFrame) -> FeatureResult:
        import time
        start_time = time.time()
        
        close = data['Close']
        
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        
        # Calculate Signal Line
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        computation_time = (time.time() - start_time) * 1000
        
        return FeatureResult(
            feature_name=self.metadata.feature_name,
            values=signal_line,
            metadata=self.metadata,
            computation_time_ms=computation_time,
            warnings=[]
        )


class MACDHistogram(BaseFeature):
    """MACD Histogram (MACD Line - Signal Line)."""
    
    def _define_metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            feature_name="MACD_Histogram",
            description="MACD Histogram - Difference between MACD Line and Signal Line. Shows momentum strength and potential reversals.",
            category=FeatureCategory.TECHNICAL,
            timeframe=Timeframe.DAY_1,
            required_columns=["Close"],
            output_range="Unbounded",
            version="1.0",
            author="system",
            computation_method="Histogram = MACD Line - Signal Line",
            assumptions="Histogram expansion indicates strengthening momentum",
            limitations="Can give false signals in choppy markets",
            references="Gerald Appel"
        )
    
    def compute(self, data: pd.DataFrame) -> FeatureResult:
        import time
        start_time = time.time()
        
        close = data['Close']
        
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        # Calculate Histogram
        histogram = macd_line - signal_line
        
        computation_time = (time.time() - start_time) * 1000
        
        return FeatureResult(
            feature_name=self.metadata.feature_name,
            values=histogram,
            metadata=self.metadata,
            computation_time_ms=computation_time,
            warnings=[]
        )
