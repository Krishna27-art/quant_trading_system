"""
Signal Decay and Alpha Half-Life Engine

Analyzes the decay rate of alpha signals and computes the optimal holding period.
Provides dynamic weighting inversely proportional to signal decay.
Uses vectorized cross-sectional Spearman Rank IC.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SignalDecayConfig:
    """Configuration for signal decay analysis."""

    max_lag_days: int = 10
    min_periods_for_fit: int = 5
    retirement_halflife_threshold: float = 1.0  # Retire if halflife < 1 day
    ic_threshold: float = 0.02


class AlphaDecayAnalyzer:
    """
    Analyzes historical signal and return series to compute decay metrics.
    """

    def __init__(self, config: SignalDecayConfig = SignalDecayConfig()):
        self.config = config

    def measure_halflife(self, signal_series: pd.Series, return_series: pd.Series) -> float:
        """
        Estimate the decay half-life of a signal.
        Expects multi-index (date, symbol) series.
        """
        lags = list(range(1, self.config.max_lag_days + 1))
        decay_curve = self.compute_decay_curve(signal_series, return_series, lags)

        valid_lags = []
        valid_ics = []

        for lag, ic in decay_curve.items():
            if ic > 0.001:  # Only fit positive ICs
                valid_lags.append(lag)
                valid_ics.append(np.log(ic))

        if len(valid_lags) < self.config.min_periods_for_fit:
            return np.nan

        # Linear fit on log(IC)
        # log(IC(t)) = log(IC(0)) - lambda * t
        slope, _ = np.polyfit(valid_lags, valid_ics, 1)

        if slope >= 0:
            return np.inf  # Signal is not decaying

        halflife = -np.log(2) / slope
        return halflife

    def compute_decay_curve(
        self, signal_series: pd.Series, return_series: pd.Series, lags: list[int]
    ) -> dict[int, float]:
        """
        Compute Information Coefficient at various forward lags.
        Fully vectorized implementation using pandas cross-sectional ranking.
        """
        decay_curve = {}
        if not isinstance(signal_series.index, pd.MultiIndex):
            return dict.fromkeys(lags, 0.0)

        sig_df = signal_series.unstack()
        ret_df = return_series.unstack()

        # Cross-sectional rank and center
        sig_ranked = sig_df.rank(axis=1)
        sig_centered = sig_ranked.sub(sig_ranked.mean(axis=1), axis=0)
        sig_var = sig_centered.pow(2).sum(axis=1)

        for lag in lags:
            shifted_ret = ret_df.shift(-lag)

            # Cross-sectional rank and center for returns
            ret_ranked = shifted_ret.rank(axis=1)
            ret_centered = ret_ranked.sub(ret_ranked.mean(axis=1), axis=0)
            ret_var = ret_centered.pow(2).sum(axis=1)

            # Vectorized Covariance and Correlation
            cov = (sig_centered * ret_centered).sum(axis=1)

            # Avoid division by zero
            denom = np.sqrt(sig_var * ret_var)
            denom = denom.replace(0, np.nan)

            daily_ics = cov / denom

            # Drop invalid days
            daily_ics = daily_ics.replace([np.inf, -np.inf], np.nan).dropna()

            if not daily_ics.empty:
                decay_curve[lag] = daily_ics.mean()
            else:
                decay_curve[lag] = 0.0

        return decay_curve

    def optimal_holding_period(self, halflife: float) -> int:
        """
        Recommend optimal holding period based on signal half-life.
        Rule of thumb: Hold until signal decays to ~25% of original strength (2 half-lives).
        """
        if np.isnan(halflife) or np.isinf(halflife):
            return self.config.max_lag_days

        optimal = int(np.ceil(halflife * 2))
        return min(optimal, self.config.max_lag_days)

    def is_signal_alive(self, halflife: float, current_ic: float) -> bool:
        """Check if signal is still viable."""
        if current_ic < self.config.ic_threshold:
            return False
        return not (not np.isnan(halflife) and halflife < self.config.retirement_halflife_threshold)


class DecayAwareSignalCombiner:
    """
    Combines signals, weighting them based on their decay profiles.
    """

    def __init__(self, analyzer: AlphaDecayAnalyzer = AlphaDecayAnalyzer()):
        self.analyzer = analyzer
        self.model_halflives: dict[str, float] = {}

    def update_model_halflife(self, model_name: str, halflife: float):
        """Store the computed half-life for a model."""
        self.model_halflives[model_name] = halflife

    def get_decay_weights(self, active_models: list[str]) -> dict[str, float]:
        """
        Calculate weights inversely proportional to decay rate (i.e. directly proportional to halflife).
        Longer halflife = higher weight.
        """
        weights = {}
        total_hl = 0.0

        for model in active_models:
            hl = self.model_halflives.get(model, 5.0)  # Default to 5 days if unknown
            if np.isnan(hl) or np.isinf(hl):
                hl = 10.0  # Cap infinite halflife
            hl = max(0.1, hl)  # Floor to prevent div zero
            weights[model] = hl
            total_hl += hl

        if total_hl > 0:
            weights = {k: v / total_hl for k, v in weights.items()}
        else:
            weights = {k: 1.0 / len(active_models) for k in active_models}

        return weights


class IntraTradeDecayTracker:
    """
    Tracks the real-time decay of a signal after a position is entered.
    Helps the execution and portfolio layers to exit a trade early if the
    signal decays faster than the expected half-life.
    """

    def __init__(self, exit_decay_threshold: float = 0.25):
        """
        exit_decay_threshold: Fraction of original signal strength at which we exit.
                              e.g., 0.25 means exit when signal is 25% of entry strength (2 half-lives).
        """
        self.exit_decay_threshold = exit_decay_threshold
        self.active_trades: dict[str, dict] = {}

    def register_trade(
        self, trade_id: str, symbol: str, entry_signal_value: float, expected_halflife_days: float
    ):
        """Register a new trade for tracking."""
        self.active_trades[trade_id] = {
            "symbol": symbol,
            "entry_signal_value": entry_signal_value,
            "expected_halflife_days": expected_halflife_days,
            "days_held": 0,
        }

    def update_and_check_exits(self, current_signals: pd.Series) -> list[str]:
        """
        Update the holding period for all active trades and check if current signal
        has decayed below the threshold. Returns a list of trade_ids to exit.
        """
        exits = []

        for trade_id, data in list(self.active_trades.items()):
            data["days_held"] += 1
            symbol = data["symbol"]

            if symbol not in current_signals:
                continue

            current_signal = current_signals[symbol]
            entry_signal = data["entry_signal_value"]

            # Check if signal flipped sign (immediate exit)
            if np.sign(current_signal) != np.sign(entry_signal):
                exits.append(trade_id)
                del self.active_trades[trade_id]
                continue

            # Check if signal magnitude decayed below threshold
            current_magnitude = abs(current_signal)
            entry_magnitude = abs(entry_signal)

            if current_magnitude < entry_magnitude * self.exit_decay_threshold:
                exits.append(trade_id)
                del self.active_trades[trade_id]
                continue

            # Time-based stop using expected halflife
            # If held for more than 3 half-lives, force exit
            if data["days_held"] > data["expected_halflife_days"] * 3:
                exits.append(trade_id)
                del self.active_trades[trade_id]

        return exits
