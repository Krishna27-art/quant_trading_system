import pandas as pd
import numpy as np
from typing import List

class DecileTokenizer:
    """
    Transforms continuous features into quantized decile tokens (0-9) 
    using an expanding window to prevent look-ahead bias.
    This hardens models against distribution shifts and absolute scale changes.
    """
    
    def __init__(self, min_periods: int = 20):
        self.min_periods = min_periods
        
    def transform(self, series: pd.Series) -> pd.Series:
        """
        Transforms a continuous pandas Series into 0-9 tokens.
        Uses expanding rank to ensure we only use past data to define bin thresholds.
        """
        # Calculate expanding rank (percentile)
        # We use a custom expanding apply to get the rank of the current value
        # relative to all past values.
        expanding_rank = series.expanding(min_periods=self.min_periods).apply(
            lambda x: (pd.Series(x).rank(pct=True).iloc[-1]), raw=False
        )
        
        # Convert percentiles (0.0 to 1.0) to deciles (0 to 9)
        # Note: 1.0 percentile will floor to 10, so we clip at 9
        tokens = (expanding_rank * 10).apply(np.floor).clip(0, 9)
        
        return tokens
        
    def transform_dataframe(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Applies tokenization to multiple columns in a DataFrame.
        """
        df_out = df.copy()
        for col in columns:
            df_out[f"{col}_token"] = self.transform(df_out[col])
        return df_out
