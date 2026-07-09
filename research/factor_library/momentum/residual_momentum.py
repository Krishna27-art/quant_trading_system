"""
Residual Momentum Factor

Residual momentum after removing market beta.
Measures stock-specific momentum independent of market movements.
"""

import pandas as pd
import numpy as np

from research.factor_library.base_factor import BaseFactor


class ResidualMomentumFactor(BaseFactor):
    """
    Residual momentum factor.
    
    Calculates momentum after removing market beta exposure.
    This isolates stock-specific alpha from market beta.
    """
    
    @property
    def name(self) -> str:
        return "residual_momentum"
    
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
        return 60
    
    @property
    def prediction_horizon(self) -> int:
        return 5
    
    @property
    def required_columns(self) -> list:
        return ["close", "market_close"]
    
    @property
    def uses_future_data(self) -> bool:
        return False
    
    @property
    def description(self) -> str:
        return "Residual momentum factor measuring stock-specific momentum after removing market beta."
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute residual momentum factor.
        
        Args:
            df: DataFrame with 'close' and 'market_close' columns
            
        Returns:
            Series with residual momentum values
        """
        close = df["close"]
        market_close = df["market_close"]
        
        # Calculate returns
        stock_returns = close.pct_change()
        market_returns = market_close.pct_change()
        
        # Calculate rolling beta (60-day window)
        rolling_cov = stock_returns.rolling(self.lookback).cov(market_returns)
        market_var = market_returns.rolling(self.lookback).var()
        beta = rolling_cov / market_var.replace(0, np.nan)
        
        # Calculate expected returns based on market
        expected_returns = beta * market_returns
        
        # Calculate residual returns (stock-specific)
        residual_returns = stock_returns - expected_returns
        
        # Calculate cumulative residual returns over lookback
        cumulative_residual = residual_returns.rolling(self.lookback).sum()
        
        # Normalize by volatility
        residual_vol = residual_returns.rolling(self.lookback).std()
        normalized_residual = cumulative_residual / residual_vol.replace(0, np.nan)
        
        # Clip extreme values
        normalized_residual = normalized_residual.clip(-5, 5)
        
        return normalized_residual
