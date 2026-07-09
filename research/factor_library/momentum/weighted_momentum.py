"""
Weighted Momentum Factor

Weighted momentum factor giving more weight to recent returns.
Captures the idea that recent momentum is more important than older momentum.
"""

import pandas as pd
import numpy as np

from research.factor_library.base_factor import BaseFactor


class WeightedMomentumFactor(BaseFactor):
    """
    Weighted momentum factor.
    
    Calculates momentum with exponential weights, giving more importance
    to recent returns. This captures the idea that recent momentum is
    more predictive than older momentum.
    """
    
    @property
    def name(self) -> str:
        return "weighted_momentum"
    
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
        return "Weighted momentum factor with exponential weights giving more importance to recent returns."
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute weighted momentum factor.
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            Series with weighted momentum values
        """
        close = df["close"]
        
        # Calculate daily returns
        returns = close.pct_change()
        
        # Calculate exponentially weighted returns (more weight to recent)
        # Using span of 10 for half-life of ~7 days
        weighted_returns = returns.ewm(span=10, adjust=False).mean()
        
        # Cumulative weighted returns over lookback
        cumulative_weighted = weighted_returns.rolling(self.lookback).sum()
        
        # Normalize by volatility
        volatility = returns.rolling(20).std()
        normalized_weighted = cumulative_weighted / volatility.replace(0, np.nan)
        
        # Clip extreme values
        normalized_weighted = normalized_weighted.clip(-5, 5)
        
        return normalized_weighted
