"""
Trend Persistence Factor

Trend persistence factor measuring how long a trend has been in place.
Identifies established trends vs. recent trend changes.
"""

import pandas as pd
import numpy as np

from research.factor_library.base_factor import BaseFactor


class TrendPersistenceFactor(BaseFactor):
    """
    Trend persistence factor.
    
    Measures how long a trend has been in place by counting consecutive
    days with positive/negative returns. Longer persistence indicates
    more established trends.
    """
    
    @property
    def name(self) -> str:
        return "trend_persistence"
    
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
        return "Trend persistence factor measuring how long a trend has been in place by counting consecutive directional moves."
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute trend persistence factor.
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            Series with trend persistence values
        """
        close = df["close"]
        
        # Calculate daily returns
        returns = close.pct_change()
        
        # Determine direction (1 for positive, -1 for negative, 0 for zero)
        direction = np.sign(returns).fillna(0)
        
        # Calculate consecutive runs of same direction
        persistence = pd.Series(index=close.index, dtype=float)
        current_run = 0
        current_direction = 0
        
        for i in range(len(direction)):
            if i == 0:
                current_run = 0
                current_direction = 0
            else:
                if direction.iloc[i] == current_direction and current_direction != 0:
                    current_run += 1
                elif direction.iloc[i] != 0:
                    current_run = 1
                    current_direction = direction.iloc[i]
                else:
                    current_run = 0
                    current_direction = 0
            
            persistence.iloc[i] = current_run * current_direction
        
        # Normalize by lookback period
        normalized_persistence = persistence / self.lookback
        
        return normalized_persistence
