"""
Relative Momentum Factor

Relative momentum comparing stock performance to benchmark.
Measures stock's momentum relative to market index.
"""

import pandas as pd
import numpy as np

from research.factor_library.base_factor import BaseFactor


class RelativeMomentumFactor(BaseFactor):
    """
    Relative momentum factor.
    
    Compares stock's momentum to benchmark (market index) momentum.
    Positive values indicate stock outperforming market, negative indicate underperforming.
    """
    
    @property
    def name(self) -> str:
        return "relative_momentum"
    
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
        return ["close", "benchmark_close"]
    
    @property
    def uses_future_data(self) -> bool:
        return False
    
    @property
    def description(self) -> str:
        return "Relative momentum factor comparing stock performance to benchmark for market-relative signals."
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute relative momentum factor.
        
        Args:
            df: DataFrame with 'close' and 'benchmark_close' columns
            
        Returns:
            Series with relative momentum values
        """
        close = df["close"]
        benchmark_close = df["benchmark_close"]
        
        # Calculate returns for stock and benchmark
        stock_returns = close.pct_change(self.lookback)
        benchmark_returns = benchmark_close.pct_change(self.lookback)
        
        # Calculate relative returns (stock - benchmark)
        relative_returns = stock_returns - benchmark_returns
        
        # Normalize by benchmark volatility
        benchmark_vol = benchmark_returns.rolling(20).std()
        normalized_relative = relative_returns / benchmark_vol.replace(0, np.nan)
        
        # Clip extreme values
        normalized_relative = normalized_relative.clip(-5, 5)
        
        return normalized_relative
