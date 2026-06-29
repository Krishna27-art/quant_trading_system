"""
Walk-Forward IC Decay Report Generator

Rigorous walk-forward validation for generated signals.
Computes Spearman Rank Information Coefficient (IC) across multiple forward horizons.
Identifies the precise half-life of S1 Cross-Sectional signals.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from utils.logger import get_logger

logger = get_logger("ic_decay")


class WalkForwardICDecayReport:
    """
    Evaluates alpha models by computing Rank IC over multiple horizons.
    Helps verify if the signal's half-life matches execution-aware decay expectations.
    """

    def __init__(self, horizons: list[int] = None):
        self.horizons = horizons or [1, 2, 3, 5, 10, 20]

    def compute_cross_sectional_ic(
        self, alpha_scores: pd.Series, forward_returns: pd.DataFrame
    ) -> dict[int, float]:
        """
        Compute Rank IC for a single cross-section (e.g., one day).
        alpha_scores: Series of predicted alpha scores for N stocks.
        forward_returns: DataFrame where columns are 'return_t{h}' corresponding to horizons.

        Returns dictionary of {horizon: IC_value}
        """
        ic_results = {}

        # Align data to ensure we are comparing exactly the same stocks
        aligned = pd.concat([alpha_scores.rename("alpha"), forward_returns], axis=1).dropna()

        if len(aligned) < 5:  # Need minimum number of samples for meaningful rank correlation
            return dict.fromkeys(self.horizons, np.nan)

        for h in self.horizons:
            col_name = f"return_t{h}"
            # Support vol-adjusted label names if present
            if col_name not in aligned.columns:
                col_name = f"vol_adj_return_t{h}"

            if col_name in aligned.columns:
                # Calculate Spearman Rank Correlation
                ic, p_val = spearmanr(aligned["alpha"], aligned[col_name])
                ic_results[h] = ic
            else:
                ic_results[h] = np.nan

        return ic_results

    def generate_report(
        self, panel_alpha: pd.Series, panel_forward_returns: pd.DataFrame, date_column: str = "date"
    ) -> pd.DataFrame:
        """
        Generate a walk-forward report over a panel dataset.
        panel_alpha and panel_forward_returns should have the date_column in their index (MultiIndex or reset).
        """
        logger.info(f"Generating Walk-Forward IC Decay Report for horizons: {self.horizons}")

        # Ensure we have date available
        if isinstance(panel_alpha.index, pd.MultiIndex):
            # Extract date level
            dates = panel_alpha.index.get_level_values(date_column).unique()
        else:
            dates = panel_alpha.index.unique()  # Assuming datetime index

        daily_ics = []

        # Iterate walk-forward day by day
        for d in dates:
            try:
                # Extract cross-section for the day
                if isinstance(panel_alpha.index, pd.MultiIndex):
                    cross_section_alpha = panel_alpha.xs(d, level=date_column)
                    cross_section_fwd = panel_forward_returns.xs(d, level=date_column)
                else:
                    cross_section_alpha = panel_alpha.loc[d]
                    cross_section_fwd = panel_forward_returns.loc[d]

                ic_dict = self.compute_cross_sectional_ic(cross_section_alpha, cross_section_fwd)
                ic_dict[date_column] = d
                daily_ics.append(ic_dict)
            except Exception as e:
                logger.debug(f"Could not compute IC for date {d}: {e}")

        # Aggregate results
        df_ic = pd.DataFrame(daily_ics).set_index(date_column)

        # Compute mean IC across all days (Walk-forward average)
        mean_ic = df_ic.mean()
        # Compute Information Ratio (Mean IC / Std Dev of IC)
        ir = df_ic.mean() / df_ic.std() * np.sqrt(252)  # Annualized

        report = pd.DataFrame(
            {
                "Mean_IC": mean_ic,
                "Annualized_IR": ir,
                "Hit_Rate": (df_ic > 0).mean(),  # % of days with positive IC
            }
        )

        logger.info("Walk-Forward IC Decay Summary", extra={"data": report.to_string()})

        # Find half-life (when IC drops to half its maximum value)
        max_ic = report["Mean_IC"].max()
        half_ic_threshold = max_ic / 2.0

        half_life_horizon = None
        for h in self.horizons:
            if report.loc[h, "Mean_IC"] < half_ic_threshold:
                half_life_horizon = h
                break

        if half_life_horizon:
            logger.info(f"Estimated Alpha Half-Life: ~{half_life_horizon} days")
        else:
            logger.info("Estimated Alpha Half-Life: > Max Horizon")

        return report
