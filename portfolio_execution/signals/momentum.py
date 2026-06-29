"""
Momentum Alpha Models — Indian Markets

Implements:
1. CrossSectionalMomentum: Classic 12-1 month momentum with sector neutralisation.
2. TimeSeriesMomentum: Single-asset TSMOM with volatility scaling.
3. DualMomentum: Combines both for a robust momentum signal.

All signals output standardized z-scores or ranks via the AlphaModel base.
"""

import numpy as np
import pandas as pd

from portfolio_execution.signals.base import AlphaModel, SignalDirection, SignalNorm
from utils.logger import get_logger

logger = get_logger(__name__)


class CrossSectionalMomentum(AlphaModel):
    """
    12-month return with 1-month skip (12-1 momentum).

    Computes cumulative return over [t-skip-lookback, t-skip] and ranks
    the cross-section.  The skip period avoids short-term reversal.
    Optionally sector-neutralised via `generate(sector_map=...)`.

    Parameters
    ----------
    lookback : int
        Momentum look-back in trading days (default 252 ≈ 12 months).
    skip : int
        Recent days to skip to avoid reversal (default 21 ≈ 1 month).
    vol_adj : bool
        If True, divide raw momentum by realised volatility.
    """

    def __init__(
        self,
        lookback: int = 252,
        skip: int = 21,
        vol_adj: bool = True,
        norm: SignalNorm = SignalNorm.ZSCORE,
        **kwargs,
    ):
        super().__init__(
            name="cs_momentum_12_1",
            lookback=lookback,
            norm=norm,
            direction=SignalDirection.LONG_SHORT,
            **kwargs,
        )
        self.skip = skip
        self.vol_adj = vol_adj

    def _compute_raw_signal(
        self,
        data: pd.DataFrame,
        **kwargs,
    ) -> pd.Series:
        """
        Expects `data` as a wide DataFrame: index = dates, columns = symbols,
        values = adjusted close prices.
        """
        if data.shape[0] < self.lookback + self.skip:
            logger.warning(
                "Insufficient history for momentum",
                extra={"rows": data.shape[0], "required": self.lookback + self.skip},
            )
            return pd.Series(dtype=float)

        # Cumulative return from t-(lookback+skip) to t-skip
        end_prices = data.iloc[-self.skip - 1]  # price at t-skip
        start_prices = data.iloc[-(self.lookback + self.skip)]  # price at t-lookback-skip

        mom = (end_prices / start_prices) - 1.0

        if self.vol_adj:
            # Annualised realised vol over the full lookback window
            log_ret = np.log(data / data.shift(1))
            window = log_ret.iloc[-(self.lookback + self.skip) : -self.skip]
            vol = window.std() * np.sqrt(252)
            vol = vol.replace(0.0, np.nan)
            mom = mom / vol

        return mom.dropna()


class TimeSeriesMomentum(AlphaModel):
    """
    Time-series momentum (TSMOM) with volatility scaling.

    For each asset, compute sign(cumulative return) × (1 / realised_vol)
    so that high-volatility assets get smaller positions.

    Parameters
    ----------
    lookback : int
        Look-back in trading days (default 126 ≈ 6 months).
    vol_lookback : int
        Days for realised vol estimation (default 63 ≈ 3 months).
    vol_target : float
        Annualised volatility target for scaling (default 0.15 = 15%).
    """

    def __init__(
        self,
        lookback: int = 126,
        vol_lookback: int = 63,
        vol_target: float = 0.15,
        norm: SignalNorm = SignalNorm.ZSCORE,
        **kwargs,
    ):
        super().__init__(
            name="ts_momentum",
            lookback=lookback,
            norm=norm,
            direction=SignalDirection.LONG_SHORT,
            **kwargs,
        )
        self.vol_lookback = vol_lookback
        self.vol_target = vol_target

    def _compute_raw_signal(
        self,
        data: pd.DataFrame,
        **kwargs,
    ) -> pd.Series:
        """
        data : wide DataFrame (dates × symbols) of adjusted close.
        """
        required = max(self.lookback, self.vol_lookback) + 1
        if data.shape[0] < required:
            return pd.Series(dtype=float)

        log_ret = np.log(data / data.shift(1))

        # Cumulative return over lookback
        cum_ret = log_ret.iloc[-self.lookback :].sum()

        # Realised vol (annualised)
        recent_vol = log_ret.iloc[-self.vol_lookback :].std() * np.sqrt(252)
        recent_vol = recent_vol.replace(0.0, np.nan)

        # TSMOM signal = sign(r) × vol_target / realised_vol
        signal = np.sign(cum_ret) * (self.vol_target / recent_vol)

        return signal.dropna()


class DualMomentum(AlphaModel):
    """
    Dual momentum: blend of cross-sectional and time-series momentum.

    Signal = w_cs × CS_MOM + w_ts × TS_MOM

    Both sub-signals are z-scored before blending.  The resulting composite
    is re-z-scored.

    Parameters
    ----------
    cs_lookback : int
        Cross-sectional momentum look-back (trading days).
    ts_lookback : int
        Time-series momentum look-back (trading days).
    cs_weight : float
        Weight on cross-sectional signal (0–1).
    ts_weight : float
        Weight on time-series signal (0–1).
    """

    def __init__(
        self,
        cs_lookback: int = 252,
        cs_skip: int = 21,
        ts_lookback: int = 126,
        cs_weight: float = 0.6,
        ts_weight: float = 0.4,
        norm: SignalNorm = SignalNorm.ZSCORE,
        **kwargs,
    ):
        super().__init__(
            name="dual_momentum",
            lookback=max(cs_lookback + cs_skip, ts_lookback),
            norm=norm,
            direction=SignalDirection.LONG_SHORT,
            **kwargs,
        )
        self.cs_model = CrossSectionalMomentum(
            lookback=cs_lookback,
            skip=cs_skip,
            norm=SignalNorm.ZSCORE,
        )
        self.ts_model = TimeSeriesMomentum(
            lookback=ts_lookback,
            norm=SignalNorm.ZSCORE,
        )
        self.cs_weight = cs_weight
        self.ts_weight = ts_weight

    def _compute_raw_signal(
        self,
        data: pd.DataFrame,
        **kwargs,
    ) -> pd.Series:
        cs_raw = self.cs_model._compute_raw_signal(data, **kwargs)
        ts_raw = self.ts_model._compute_raw_signal(data, **kwargs)

        if cs_raw.empty or ts_raw.empty:
            return cs_raw if not cs_raw.empty else ts_raw

        # z-score each sub-signal independently
        cs_z = self._zscore(cs_raw)
        ts_z = self._zscore(ts_raw)

        common = cs_z.dropna().index.intersection(ts_z.dropna().index)
        if len(common) < 2:
            return pd.Series(dtype=float)

        blended = self.cs_weight * cs_z.loc[common] + self.ts_weight * ts_z.loc[common]
        return blended


class SectorRelativeMomentum(AlphaModel):
    """
    Momentum relative to sector median.

    Computes each stock's momentum minus its sector median momentum,
    capturing intra-sector relative strength.

    Parameters
    ----------
    lookback : int
        Momentum look-back in trading days.
    skip : int
        Recent days to skip.
    """

    def __init__(
        self,
        lookback: int = 126,
        skip: int = 21,
        norm: SignalNorm = SignalNorm.ZSCORE,
        **kwargs,
    ):
        super().__init__(
            name="sector_relative_momentum",
            lookback=lookback,
            norm=norm,
            direction=SignalDirection.LONG_SHORT,
            **kwargs,
        )
        self.skip = skip

    def _compute_raw_signal(
        self,
        data: pd.DataFrame,
        sector_map: pd.Series | None = None,
        **kwargs,
    ) -> pd.Series:
        """
        data : wide DataFrame (dates × symbols) of adjusted close.
        sector_map : pd.Series mapping symbol → sector.
        """
        if data.shape[0] < self.lookback + self.skip:
            return pd.Series(dtype=float)

        end_prices = data.iloc[-self.skip - 1]
        start_prices = data.iloc[-(self.lookback + self.skip)]
        mom = (end_prices / start_prices) - 1.0
        mom = mom.dropna()

        if sector_map is None:
            return mom

        # Compute sector-relative momentum
        common = mom.index.intersection(sector_map.index)
        mom = mom.loc[common]
        sectors = sector_map.loc[common]

        sector_median = mom.groupby(sectors).transform("median")
        return mom - sector_median


class MomentumAcceleration(AlphaModel):
    """
    Momentum acceleration: second derivative of cumulative returns.

    Detects whether momentum is accelerating or decelerating by comparing
    recent momentum to prior momentum over two consecutive windows.

    Signal = MOM(short) − MOM(long) where both are normalised.
    """

    def __init__(
        self,
        short_lookback: int = 63,
        long_lookback: int = 126,
        skip: int = 5,
        norm: SignalNorm = SignalNorm.ZSCORE,
        **kwargs,
    ):
        super().__init__(
            name="momentum_acceleration",
            lookback=long_lookback + skip,
            norm=norm,
            direction=SignalDirection.LONG_SHORT,
            **kwargs,
        )
        self.short_lookback = short_lookback
        self.long_lookback = long_lookback
        self.skip = skip

    def _compute_raw_signal(
        self,
        data: pd.DataFrame,
        **kwargs,
    ) -> pd.Series:
        required = self.long_lookback + self.skip + 1
        if data.shape[0] < required:
            return pd.Series(dtype=float)

        # Short-window momentum (recent)
        short_end = data.iloc[-self.skip - 1]
        short_start = data.iloc[-(self.short_lookback + self.skip)]
        mom_short = (short_end / short_start) - 1.0

        # Long-window momentum
        long_end = data.iloc[-self.skip - 1]
        long_start = data.iloc[-(self.long_lookback + self.skip)]
        mom_long = (long_end / long_start) - 1.0

        # Normalize each independently before differencing
        common = mom_short.dropna().index.intersection(mom_long.dropna().index)
        if len(common) < 5:
            return pd.Series(dtype=float)

        ms = self._zscore(mom_short.loc[common])
        ml = self._zscore(mom_long.loc[common])

        # Acceleration = short normalised momentum − long normalised momentum
        return ms - ml
