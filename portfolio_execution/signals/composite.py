"""
Composite Alpha Blending and Decay Tracking Engine — Indian Markets

This module implements:
1. CompositeAlphaModel: Combines multiple individual alpha signals using dynamic
   or static Information Coefficient (IC) weights. Supports winsorization,
   z-score normalization, sector-neutralization, and dynamic regime blending.
2. SignalDecayTracker: Measures signal decay half-life and automatically retires
   signals with decayed predictive power or negative IC IRs.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from portfolio_execution.signals.base import AlphaModel, SignalDirection, SignalNorm
from utils.logger import get_logger

logger = get_logger(__name__)


class SignalDecayTracker:
    """
    Tracks historical performance and predictive decay of alpha models.
    Measures the decay half-life of signals and handles automatic retirement of
    underperforming models (IC Information Ratio below threshold or negative).
    """

    def __init__(
        self,
        min_periods: int = 20,
        ic_ir_threshold: float = 0.1,
        halflife_threshold_days: float = 5.0,
    ):
        self.min_periods = min_periods
        self.ic_ir_threshold = ic_ir_threshold
        self.halflife_threshold_days = halflife_threshold_days
        self.performance_history: dict[str, list[dict]] = {}

    def record_performance(
        self,
        model_name: str,
        timestamp: pd.Timestamp,
        signal: pd.Series,
        forward_returns: pd.Series,
    ) -> None:
        """
        Record the performance of an alpha model for a single period.
        """
        common_idx = signal.index.intersection(forward_returns.index)
        if len(common_idx) < 5:
            return

        sig = signal.loc[common_idx]
        ret = forward_returns.loc[common_idx]

        rank_ic, _ = spearmanr(sig, ret)
        if np.isnan(rank_ic):
            rank_ic = 0.0

        lags = [1, 2, 3, 5, 10]
        decay_ics = {}
        for lag in lags:
            shifted_ret = forward_returns.shift(-lag).loc[common_idx]
            common_lag_idx = sig.index.intersection(shifted_ret.dropna().index)
            if len(common_lag_idx) >= 5:
                lag_ic, _ = spearmanr(sig.loc[common_lag_idx], shifted_ret.loc[common_lag_idx])
                decay_ics[f"ic_lag_{lag}"] = 0.0 if np.isnan(lag_ic) else lag_ic
            else:
                decay_ics[f"ic_lag_{lag}"] = 0.0

        if model_name not in self.performance_history:
            self.performance_history[model_name] = []

        record = {"timestamp": timestamp, "ic": rank_ic, **decay_ics}
        self.performance_history[model_name].append(record)

    def calculate_decay_metrics(self, model_name: str) -> dict[str, float]:
        """
        Computes the mean IC, IC Information Ratio, and decay half-life.
        """
        history = self.performance_history.get(model_name, [])
        if len(history) < self.min_periods:
            return {
                "mean_ic": 0.0,
                "ic_ir": 0.0,
                "decay_halflife": np.nan,
                "should_retire": 0.0,
            }

        df = pd.DataFrame(history)
        ic_col = df["ic"]
        mean_ic = ic_col.mean()
        std_ic = ic_col.std()
        ic_ir = mean_ic / std_ic if std_ic > 0 else 0.0

        lags = [1, 2, 3, 5, 10]
        lag_means = []
        for lag in lags:
            col_name = f"ic_lag_{lag}"
            if col_name in df.columns:
                lag_means.append(abs(df[col_name].mean()))
            else:
                lag_means.append(0.0)

        valid_lags = []
        valid_vals = []
        for lag, val in zip(lags, lag_means, strict=False):
            if val > 0.001:
                valid_lags.append(lag)
                valid_vals.append(np.log(val))

        halflife = np.nan
        if len(valid_lags) >= 2:
            slope, _ = np.polyfit(valid_lags, valid_vals, 1)
            if slope < 0:
                halflife = -np.log(2) / slope

        should_retire = 0.0
        if ic_ir < self.ic_ir_threshold or (
            not np.isnan(halflife) and halflife < self.halflife_threshold_days
        ):
            should_retire = 1.0
            logger.warning(
                "Alpha model %s flagged for retirement | IC IR: %.3f, Half-Life: %.1f days",
                model_name,
                ic_ir,
                halflife,
            )

        return {
            "mean_ic": mean_ic,
            "ic_ir": ic_ir,
            "decay_halflife": halflife,
            "should_retire": should_retire,
        }


class CompositeAlphaModel(AlphaModel):
    """
    Blends multiple alpha models into a single portfolio-level alpha score.
    Supports dynamic weighting using rolling Information Coefficients,
    as well as Dynamic Regime Blending.
    """

    def __init__(
        self,
        models: list[AlphaModel],
        decay_tracker: SignalDecayTracker | None = None,
        dynamic_weights: bool = True,
        regime_weights: dict[str, dict[str, float]] | None = None,
        lookback: int = 60,
        norm: SignalNorm = SignalNorm.ZSCORE,
        **kwargs,
    ):
        super().__init__(
            name="composite_alpha",
            lookback=lookback,
            norm=norm,
            direction=SignalDirection.LONG_SHORT,
            **kwargs,
        )
        self.models = models
        self.decay_tracker = decay_tracker or SignalDecayTracker()
        self.dynamic_weights = dynamic_weights
        self.regime_weights = regime_weights or {}
        self.static_weights: dict[str, float] = {model.name: 1.0 / len(models) for model in models}

    def _compute_raw_signal(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Aggregate signals from all active child models.
        Optionally uses 'regime' from kwargs to apply Dynamic Regime Blending.
        """
        signals = {}
        active_models = []

        current_regime = kwargs.get("regime", "neutral")

        for model in self.models:
            metrics = self.decay_tracker.calculate_decay_metrics(model.name)
            if metrics.get("should_retire", 0.0) > 0.5:
                logger.info("Skipping retired alpha model: %s", model.name)
                continue

            sig_obj = model.generate(data, **kwargs)
            if sig_obj is not None and not sig_obj.signal.empty:
                signals[model.name] = sig_obj.signal
                active_models.append(model)

        if not signals:
            logger.warning("No active alpha signals generated for composite blending.")
            return pd.Series(dtype=float)

        sig_df = pd.DataFrame(signals)

        # Dynamic Regime Blending
        base_weights = {}
        if current_regime in self.regime_weights:
            base_weights = self.regime_weights[current_regime]
        else:
            base_weights = self.static_weights

        weights = {}
        if self.dynamic_weights and active_models:
            total_ir = 0.0
            for model in active_models:
                metrics = self.decay_tracker.calculate_decay_metrics(model.name)
                ir = max(0.01, metrics.get("ic_ir", 0.1))  # Floor weight at 0.01

                # Apply regime base multiplier
                regime_multiplier = base_weights.get(model.name, 1.0)
                final_weight = ir * regime_multiplier

                weights[model.name] = final_weight
                total_ir += final_weight

            if total_ir > 0:
                weights = {k: v / total_ir for k, v in weights.items()}
            else:
                weights = {model.name: 1.0 / len(active_models) for model in active_models}
        else:
            total_static = sum(base_weights.get(m.name, 0.0) for m in active_models)
            if total_static > 0:
                weights = {
                    m.name: base_weights.get(m.name, 0.0) / total_static for m in active_models
                }
            else:
                weights = {m.name: 1.0 / len(active_models) for m in active_models}

        blended = pd.Series(0.0, index=sig_df.index)
        for col in sig_df.columns:
            blended += sig_df[col].fillna(0.0) * weights.get(col, 0.0)

        valid_mask = sig_df.notna().any(axis=1)
        return blended.loc[valid_mask]
