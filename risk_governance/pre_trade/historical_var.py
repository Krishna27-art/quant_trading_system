"""
Historical Value-at-Risk (VaR) Engine.
"""

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class HistoricalVaR:
    def __init__(self, confidence_level: float = 0.99, lookback_window: int = 252):
        self.confidence_level = confidence_level
        self.lookback_window = lookback_window

    def compute_portfolio_var(
        self, portfolio_weights: dict[str, float], historical_returns: np.ndarray
    ) -> float:
        """
        historical_returns: (T, N) array of historical returns for N assets.
        Returns the Historical VaR for the portfolio.
        """
        weights = np.array(list(portfolio_weights.values()))

        # Calculate historical portfolio returns
        portfolio_simulated_returns = np.dot(historical_returns, weights)

        # Sort returns worst to best
        sorted_returns = np.sort(portfolio_simulated_returns)

        # Find the percentile index
        percentile_idx = int((1.0 - self.confidence_level) * len(sorted_returns))

        # VaR is the absolute value of the loss at the given percentile
        var_loss = sorted_returns[percentile_idx]

        return abs(var_loss) if var_loss < 0 else 0.0
