"""
Linear Regression Slope Trend Factor

Linear regression slope factor measuring trend direction and steepness.
Uses linear regression on price over lookback period to quantify trend.
"""

import pandas as pd
import numpy as np
from scipy import stats

from research.factor_library.base_factor import BaseFactor


class LinearRegressionSlopeFactor(BaseFactor):
    """
    Linear regression slope trend factor.
    
    Calculates the slope of a linear regression line fitted to prices
    over the lookback period. Positive slope indicates uptrend, negative
    indicates downtrend. The magnitude indicates trend steepness.
    """
    
    @property
    def name(self) -> str:
        return "linear_regression_slope"
    
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
        return ["close"]
    
    @property
    def uses_future_data(self) -> bool:
        return False
    
    @property
    def description(self) -> str:
        return "Linear regression slope factor measuring trend direction and steepness using regression on price history."
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute linear regression slope factor.
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            Series with linear regression slope values
        """
        close = df["close"]
        
        # Calculate rolling linear regression slope
        slopes = pd.Series(index=close.index, dtype=float)
        
        for i in range(self.lookback, len(close)):
            window = close.iloc[i - self.lookback : i + 1]
            x = np.arange(len(window))
            y = window.values
            
            # Calculate linear regression slope
            slope, _, _, _, _ = stats.linregress(x, y)
            
            # Normalize by price for stationarity
            normalized_slope = slope / window.mean()
            
            slopes.iloc[i] = normalized_slope
        
        # Z-score normalization
        slopes_zscore = (slopes - slopes.rolling(60).mean()) / slopes.rolling(60).std()
        
        # Clip extreme values
        slopes_zscore = slopes_zscore.clip(-5, 5)
        
        return slopes_zscore
