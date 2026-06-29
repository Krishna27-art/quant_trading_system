"""
Cross-Asset Correlation Signal Engine

Generates alpha signals by looking at correlations across different asset classes
and macro indicators (Index futures basis, Sector rotation, FX correlation, Global markets).
"""

import pandas as pd

from portfolio_execution.signals.base import AlphaModel, SignalDirection, SignalNorm
from utils.logger import get_logger

logger = get_logger(__name__)


class IndexFuturesBasisAlpha(AlphaModel):
    """
    Nifty/Bank Nifty futures basis mean-reversion.
    Basis = (Futures - Spot) / Spot
    """

    def __init__(self, lookback: int = 20, zscore_threshold: float = 1.5, **kwargs):
        super().__init__(
            name="index_futures_basis",
            direction=SignalDirection.LONG_SHORT,
            norm=SignalNorm.ZSCORE,
            lookback=lookback,
            **kwargs,
        )
        self.zscore_threshold = zscore_threshold

    def _compute_raw_signal(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Expects 'futures_price' and 'spot_price' in data.
        """
        if "futures_price" not in data.columns or "spot_price" not in data.columns:
            return pd.Series(index=data.index, dtype=float)

        basis = (data["futures_price"] - data["spot_price"]) / data["spot_price"]

        # In a real model, we would calculate rolling z-score of the basis
        # Here we assume 'basis_zscore' is pre-calculated or we proxy it
        if "basis_zscore" in data.columns:
            z_basis = data["basis_zscore"]
        else:
            # Fallback naive implementation (requires rolling window which isn't
            # fully available in a single cross-sectional snapshot without history)
            # We'll just return negative basis to imply mean-reversion
            z_basis = basis * 1000  # scale up

        # Mean reversion: if basis is highly positive (contango), expect it to revert (short)
        # If basis is highly negative (backwardation), expect it to revert (long)
        raw_signal = -z_basis
        return raw_signal


class SectorRotationAlpha(AlphaModel):
    """
    Sector momentum rotation signals.
    """

    def __init__(self, sector_momentum_lookback: int = 20, **kwargs):
        super().__init__(
            name="sector_rotation",
            lookback=kwargs.pop("lookback", 20),
            direction=SignalDirection.LONG_SHORT,
            norm=SignalNorm.ZSCORE,
            **kwargs,
        )
        self.sector_momentum_lookback = sector_momentum_lookback

    def _compute_raw_signal(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Expects 'sector_return_1m' in data.
        """
        if "sector_return_1m" not in data.columns:
            return pd.Series(index=data.index, dtype=float)

        # Simple sector momentum - buy stocks in sectors with strong recent returns
        raw_signal = data["sector_return_1m"]
        return raw_signal


class FXEquityCorrelationAlpha(AlphaModel):
    """
    USD/INR correlation signals.
    IT and Pharma benefit from INR depreciation (USD/INR up).
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="fx_equity_correlation",
            lookback=kwargs.pop("lookback", 20),
            direction=SignalDirection.LONG_SHORT,
            norm=SignalNorm.ZSCORE,
            **kwargs,
        )

    def _compute_raw_signal(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Expects 'usd_inr_return' and 'fx_sensitivity' in data.
        """
        if "usd_inr_return" not in data.columns or "fx_sensitivity" not in data.columns:
            return pd.Series(index=data.index, dtype=float)

        # Signal = USD_INR_Return * Stock_FX_Sensitivity
        # E.g., if USD goes up (+1%) and IT stock has sensitivity (+0.5), signal is positive
        usd_inr_ret = data["usd_inr_return"]
        sensitivity = data["fx_sensitivity"]

        raw_signal = usd_inr_ret * sensitivity
        return raw_signal


class GlobalCorrelationAlpha(AlphaModel):
    """
    Global market correlation signals.
    Uses S&P 500 overnight returns to predict Nifty opening moves.
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="global_correlation",
            lookback=kwargs.pop("lookback", 20),
            direction=SignalDirection.LONG_SHORT,
            norm=SignalNorm.ZSCORE,
            **kwargs,
        )

    def _compute_raw_signal(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Expects 'sp500_overnight_return' and 'global_beta' in data.
        """
        if "sp500_overnight_return" not in data.columns or "global_beta" not in data.columns:
            return pd.Series(index=data.index, dtype=float)

        sp500_ret = data["sp500_overnight_return"]
        beta = data["global_beta"]

        raw_signal = sp500_ret * beta
        return raw_signal
