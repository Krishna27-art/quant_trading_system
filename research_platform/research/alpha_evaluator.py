"""
Alpha Evaluation Metrics.
Vectorized computation of Information Coefficient (IC), Rank IC, and Quantile Spread Returns.
Proves the statistical edge of generated signals.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from utils.logger import get_logger

logger = get_logger(__name__)


class AlphaEvaluator:
    def __init__(self, quantiles: int = 5):
        self.quantiles = quantiles

    def calculate_ic(
        self, predictions_df: pd.DataFrame, forward_returns_df: pd.DataFrame
    ) -> dict[str, float]:
        """
        Calculates Pearson IC and Spearman Rank IC.
        predictions_df: DataFrame with 'symbol' as columns and 'timestamp' as index.
        forward_returns_df: DataFrame with exact same shape representing actual forward returns.
        """
        # Align dataframes exactly
        aligned_preds, aligned_rets = predictions_df.align(forward_returns_df, join="inner")

        # Flatten for cross-sectional + time-series pooled correlation, or we can do daily cross-sectional
        # Here we compute the average cross-sectional Rank IC (the standard institutional metric)

        daily_rank_ic = []
        daily_pearson_ic = []

        for dt in aligned_preds.index:
            preds = aligned_preds.loc[dt].dropna()
            rets = aligned_rets.loc[dt].dropna()

            # Intersect symbols available for this datetime
            common_symbols = preds.index.intersection(rets.index)
            if len(common_symbols) > 5:  # Need minimum degrees of freedom
                p = preds[common_symbols].values
                r = rets[common_symbols].values

                # Spearman Rank IC
                rank_ic, _ = spearmanr(p, r)
                if not np.isnan(rank_ic):
                    daily_rank_ic.append(rank_ic)

                # Pearson IC
                pearson_ic = np.corrcoef(p, r)[0, 1]
                if not np.isnan(pearson_ic):
                    daily_pearson_ic.append(pearson_ic)

        return {
            "mean_rank_ic": float(np.mean(daily_rank_ic)),
            "mean_pearson_ic": float(np.mean(daily_pearson_ic)),
            "ic_ir": (
                float(np.mean(daily_rank_ic) / np.std(daily_rank_ic))
                if np.std(daily_rank_ic) > 0
                else 0.0
            ),
        }

    def simulate_quantile_spread(
        self, predictions_df: pd.DataFrame, forward_returns_df: pd.DataFrame
    ) -> tuple[float, pd.Series]:
        """
        Simulates going Long the top quantile (e.g. top 20%) and Short the bottom quantile (e.g. bottom 20%).
        Returns the annualized spread return and the cumulative return series.
        """
        aligned_preds, aligned_rets = predictions_df.align(forward_returns_df, join="inner")
        spread_returns = []

        for dt in aligned_preds.index:
            preds = aligned_preds.loc[dt].dropna()
            rets = aligned_rets.loc[dt].dropna()

            common_symbols = preds.index.intersection(rets.index)
            if len(common_symbols) < self.quantiles:
                spread_returns.append(0.0)
                continue

            p = preds[common_symbols]
            r = rets[common_symbols]

            # Assign quantiles
            try:
                q_labels = pd.qcut(p, self.quantiles, labels=False, duplicates="drop")
                top_q = q_labels == (self.quantiles - 1)
                bottom_q = q_labels == 0

                long_ret = r[top_q].mean() if top_q.any() else 0.0
                short_ret = r[bottom_q].mean() if bottom_q.any() else 0.0

                # Market neutral spread return
                spread = long_ret - short_ret
                spread_returns.append(spread)
            except ValueError:
                # Can happen if all predictions are identical
                spread_returns.append(0.0)

        spread_series = pd.Series(spread_returns, index=aligned_preds.index)
        cum_ret = (1 + spread_series).cumprod()
        annualized = (
            spread_series.mean() * 252
        )  # Assuming daily for annualization factor, adjust if intraday

        return float(annualized), cum_ret
