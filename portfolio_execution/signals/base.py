"""
Alpha Model Base Class

Abstract base class providing a standardized interface for all alpha signal
generators. Handles signal normalization (z-score, rank), winsorization,
information coefficient computation, and decay tracking.

Indian market conventions:
- Symbols follow NSE/BSE naming (e.g., RELIANCE, TCS)
- Trading hours 09:15–15:30 IST
- T+1 settlement
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from utils.logger import get_logger

logger = get_logger(__name__)


class SignalNorm(Enum):
    """Signal normalization methods."""

    ZSCORE = "zscore"
    RANK = "rank"
    MINMAX = "minmax"
    RAW = "raw"


class SignalDirection(Enum):
    """Expected direction of alpha signal vs forward returns."""

    LONG = 1
    SHORT = -1
    LONG_SHORT = 0


@dataclass
class AlphaSignal:
    """Container for a single cross-sectional alpha signal snapshot."""

    timestamp: pd.Timestamp
    signal: pd.Series  # symbol → signal value
    raw_signal: pd.Series  # pre-normalization values
    direction: SignalDirection = SignalDirection.LONG_SHORT
    metadata: dict = field(default_factory=dict)

    @property
    def n_assets(self) -> int:
        return self.signal.dropna().shape[0]

    @property
    def coverage(self) -> float:
        total = self.signal.shape[0]
        return self.signal.dropna().shape[0] / total if total > 0 else 0.0

    def top_n(self, n: int = 10) -> pd.Series:
        return self.signal.nlargest(n)

    def bottom_n(self, n: int = 10) -> pd.Series:
        return self.signal.nsmallest(n)


@dataclass
class AlphaPerformance:
    """Performance metrics for an alpha model over a backtest window."""

    ic_mean: float = 0.0
    ic_std: float = 0.0
    ic_ir: float = 0.0  # IC Information Ratio = mean(IC) / std(IC)
    rank_ic_mean: float = 0.0
    rank_ic_std: float = 0.0
    rank_ic_ir: float = 0.0
    turnover_mean: float = 0.0
    decay_halflife: float = np.nan
    hit_rate: float = 0.0  # fraction of periods with positive IC
    n_periods: int = 0


class AlphaModel(ABC):
    """
    Abstract base class for all alpha signal generators.

    Subclasses must implement:
        - _compute_raw_signal(data, **kwargs) -> pd.Series

    The base class handles:
        - Winsorization, z-scoring, ranking
        - Sector/group neutralization
        - IC and Rank-IC computation
        - Signal decay measurement
        - Turnover tracking
    """

    def __init__(
        self,
        name: str,
        lookback: int,
        holding_period: int = 1,
        norm: SignalNorm = SignalNorm.ZSCORE,
        winsorize_std: float = 3.0,
        direction: SignalDirection = SignalDirection.LONG_SHORT,
        min_history: int = 60,
    ):
        self.name = name
        self.lookback = lookback
        self.holding_period = holding_period
        self.norm = norm
        self.winsorize_std = winsorize_std
        self.direction = direction
        self.min_history = min_history

        self._signal_history: list[AlphaSignal] = []
        self._ic_series: list[tuple[pd.Timestamp, float]] = []
        self._rank_ic_series: list[tuple[pd.Timestamp, float]] = []
        self._logger = get_logger(f"alpha.{name}")

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @abstractmethod
    def _compute_raw_signal(
        self,
        data: pd.DataFrame,
        **kwargs,
    ) -> pd.Series:
        """
        Compute the raw (un-normalised) cross-sectional signal.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data with MultiIndex (date, symbol) or wide-format
            (dates × symbols).  Columns include at minimum:
            open, high, low, close, volume.

        Returns
        -------
        pd.Series
            Index = symbol, values = raw signal.
        """
        ...

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        data: pd.DataFrame,
        timestamp: pd.Timestamp | None = None,
        sector_map: pd.Series | None = None,
        **kwargs,
    ) -> AlphaSignal:
        """
        Full pipeline: compute raw signal → winsorize → normalise → neutralise.

        Parameters
        ----------
        data : pd.DataFrame
            Price/volume panel.
        timestamp : pd.Timestamp, optional
            Signal generation timestamp (defaults to latest date in data).
        sector_map : pd.Series, optional
            symbol → sector for cross-sectional neutralization.

        Returns
        -------
        AlphaSignal
        """
        raw = self._compute_raw_signal(data, **kwargs)
        if raw.dropna().shape[0] < 2:
            self._logger.info(
                "Insufficient non-NaN signals",
                extra={"n_valid": int(raw.dropna().shape[0])},
            )
            return AlphaSignal(
                timestamp=timestamp or pd.Timestamp.now(tz="Asia/Kolkata"),
                signal=raw,
                raw_signal=raw,
                direction=self.direction,
            )

        processed = self._winsorize(raw)
        processed = self._normalize(processed)

        if sector_map is not None:
            processed = self._sector_neutralize(processed, sector_map)

        ts = timestamp or pd.Timestamp.now(tz="Asia/Kolkata")
        sig = AlphaSignal(
            timestamp=ts,
            signal=processed,
            raw_signal=raw,
            direction=self.direction,
            metadata={"model": self.name, "norm": self.norm.value},
        )
        self._signal_history.append(sig)
        self._logger.info(
            "Signal generated",
            extra={"n_assets": sig.n_assets, "coverage": f"{sig.coverage:.2%}"},
        )
        return sig

    def generate_signals(self, state_managers: dict[str, Any]) -> list[Any]:
        """
        Orchestrator-compatible entry point.
        Converts the state managers' 1-minute bar history into wide pandas DataFrames,
        runs the signal generation logic cross-sectionally, and wraps the outputs as TradeSignal objects.
        """
        closes_dict = {}
        ohlcv_dict = {}

        required_len = self.lookback + 10

        for sym, state_mgr in state_managers.items():
            candles = state_mgr.candles_1m
            if len(candles) < required_len:
                continue

            dates = [pd.Timestamp(c.timestamp) for c in candles]
            closes = [c.close for c in candles]
            s = pd.Series(closes, index=dates)
            closes_dict[sym] = s[~s.index.duplicated(keep="last")]

            df_ohlcv = pd.DataFrame(
                [
                    {
                        "open": c.open,
                        "high": c.high,
                        "low": c.low,
                        "close": c.close,
                        "volume": c.volume,
                    }
                    for c in candles
                ],
                index=dates,
            )
            df_ohlcv = df_ohlcv[~df_ohlcv.index.duplicated(keep="last")]
            ohlcv_dict[sym] = df_ohlcv

        if not closes_dict:
            return []

        # Build wide prices with point-in-time alignment only.
        # Missing values at the start stay missing and are excluded from signals.
        df_prices = pd.DataFrame(closes_dict).sort_index().ffill()

        try:
            # Generate signal snapshot
            last_timestamp = df_prices.index[-1]
            sig_obj = self.generate(
                data=df_prices,
                ohlcv=ohlcv_dict,
                timestamp=last_timestamp,
                state_managers=state_managers,
            )
            if sig_obj is None or sig_obj.signal.empty:
                return []

            from portfolio_execution.orchestrator import build_signal

            signals = []
            for sym, val in sig_obj.signal.items():
                if pd.isna(val) or val == 0:
                    continue

                state_mgr = state_managers.get(sym)
                if not state_mgr:
                    continue

                direction = "long" if val > 0 else "short"
                curr_price = df_prices[sym].iloc[-1]
                if pd.isna(curr_price):
                    continue
                curr_price = float(curr_price)

                market_data = {
                    "ltp": curr_price,
                    "adv_20d": getattr(state_mgr, "get_adv", lambda: 1000000)() or 1000000,
                }

                # Default OBI for now, the real OBI stream is captured at EMS level
                obi_features = {"spread_bps": 2.0}

                inst_sig = build_signal(
                    symbol=sym,
                    side=direction.upper(),
                    model_score=float(val),
                    obi_features=obi_features,
                    market_data=market_data,
                )

                signals.append(inst_sig)

            return signals
        except Exception as e:
            self._logger.error(f"Error in generate_signals wrapper: {e}", exc_info=True)
            return []

    def compute_ic(
        self,
        signal: pd.Series,
        forward_returns: pd.Series,
        timestamp: pd.Timestamp | None = None,
    ) -> tuple[float, float]:
        """
        Compute Pearson IC and Spearman Rank-IC between signal and forward
        returns. Both series must share the same symbol index.

        Returns
        -------
        (ic, rank_ic) : Tuple[float, float]
        """
        common = signal.dropna().index.intersection(forward_returns.dropna().index)
        if len(common) < 10:
            return np.nan, np.nan

        s = signal.loc[common].values
        r = forward_returns.loc[common].values

        ic = np.corrcoef(s, r)[0, 1]
        rank_ic = sp_stats.spearmanr(s, r).statistic

        ts = timestamp or pd.Timestamp.now(tz="Asia/Kolkata")
        self._ic_series.append((ts, ic))
        self._rank_ic_series.append((ts, rank_ic))
        return float(ic), float(rank_ic)

    def performance_summary(self) -> AlphaPerformance:
        """Aggregate IC/Rank-IC history into a performance summary."""
        if not self._ic_series:
            return AlphaPerformance()

        ics = np.array([v for _, v in self._ic_series if np.isfinite(v)])
        rics = np.array([v for _, v in self._rank_ic_series if np.isfinite(v)])

        ic_mean = float(np.mean(ics)) if len(ics) else 0.0
        ic_std = float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0
        ic_ir = ic_mean / ic_std if ic_std > 1e-9 else 0.0

        ric_mean = float(np.mean(rics)) if len(rics) else 0.0
        ric_std = float(np.std(rics, ddof=1)) if len(rics) > 1 else 0.0
        ric_ir = ric_mean / ric_std if ric_std > 1e-9 else 0.0

        hit = float(np.mean(ics > 0)) if len(ics) else 0.0
        turnover = self._compute_avg_turnover()
        halflife = self._estimate_decay_halflife()

        return AlphaPerformance(
            ic_mean=ic_mean,
            ic_std=ic_std,
            ic_ir=ic_ir,
            rank_ic_mean=ric_mean,
            rank_ic_std=ric_std,
            rank_ic_ir=ric_ir,
            turnover_mean=turnover,
            decay_halflife=halflife,
            hit_rate=hit,
            n_periods=len(ics),
        )

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------
    def _winsorize(self, s: pd.Series) -> pd.Series:
        """Winsorize signal at ±winsorize_std standard deviations."""
        if isinstance(s, pd.DataFrame):
            mu = s.mean(axis=1)
            sigma = s.std(axis=1)
            lower = mu - self.winsorize_std * sigma
            upper = mu + self.winsorize_std * sigma
            return s.clip(lower=lower, upper=upper, axis=0)

        mu = s.mean()
        sigma = s.std()
        if sigma < 1e-12:
            return s
        lower = mu - self.winsorize_std * sigma
        upper = mu + self.winsorize_std * sigma
        return s.clip(lower=lower, upper=upper)

    def _normalize(self, s: pd.Series) -> pd.Series:
        """Apply chosen normalisation."""
        if self.norm == SignalNorm.ZSCORE:
            return self._zscore(s)
        elif self.norm == SignalNorm.RANK:
            return self._rank_normalize(s)
        elif self.norm == SignalNorm.MINMAX:
            return self._minmax(s)
        return s

    @staticmethod
    def _zscore(s: pd.Series) -> pd.Series:
        """Cross-sectional z-score."""
        if isinstance(s, pd.DataFrame):
            mu = s.mean(axis=1)
            sigma = s.std(axis=1)
            # Avoid division by zero
            return s.sub(mu, axis=0).div(sigma.replace(0, np.nan), axis=0).fillna(0.0)

        mu = s.mean()
        sigma = s.std()
        if sigma < 1e-12:
            return s * 0.0
        return (s - mu) / sigma

    @staticmethod
    def _rank_normalize(s: pd.Series) -> pd.Series:
        """Rank-normalise to uniform [−1, +1]."""
        if isinstance(s, pd.DataFrame):
            ranked = s.rank(pct=True, axis=1, method="average")
        else:
            ranked = s.rank(pct=True, method="average")
        return 2.0 * ranked - 1.0

    @staticmethod
    def _minmax(s: pd.Series) -> pd.Series:
        """Min-max normalise to [0, 1]."""
        if isinstance(s, pd.DataFrame):
            lo = s.min(axis=1)
            hi = s.max(axis=1)
            rng = hi - lo
            return s.sub(lo, axis=0).div(rng.replace(0, np.nan), axis=0).fillna(0.0)

        lo, hi = s.min(), s.max()
        rng = hi - lo
        if rng < 1e-12:
            return s * 0.0
        return (s - lo) / rng

    @staticmethod
    def _sector_neutralize(signal: pd.Series, sector_map: pd.Series) -> pd.Series:
        """
        Demean signal within each sector so that signal is
        cross-sectionally neutral with respect to sector exposure.
        """
        if isinstance(signal, pd.DataFrame):
            common = signal.columns.intersection(sector_map.index)
            sig = signal[common].copy()
            sec = sector_map.loc[common]
            # Groupby columns is done by transposing
            group_mean = sig.T.groupby(sec).transform("mean").T
            return sig - group_mean

        common = signal.index.intersection(sector_map.index)
        sig = signal.loc[common].copy()
        sec = sector_map.loc[common]
        group_mean = sig.groupby(sec).transform("mean")
        return sig - group_mean

    # ------------------------------------------------------------------
    # Turnover and decay helpers
    # ------------------------------------------------------------------
    def _compute_avg_turnover(self) -> float:
        """Average rank-based turnover across consecutive signals."""
        if len(self._signal_history) < 2:
            return 0.0

        turnovers: list[float] = []
        for i in range(1, len(self._signal_history)):
            prev = self._signal_history[i - 1].signal
            curr = self._signal_history[i].signal
            common = prev.dropna().index.intersection(curr.dropna().index)
            if len(common) < 2:
                continue
            prev_r = prev.loc[common].rank(pct=True)
            curr_r = curr.loc[common].rank(pct=True)
            turnover = (curr_r - prev_r).abs().mean()
            turnovers.append(float(turnover))

        return float(np.mean(turnovers)) if turnovers else 0.0

    def _estimate_decay_halflife(self) -> float:
        """
        Estimate signal decay half-life from autocorrelation of the
        IC series.  Fits an exponential decay model.
        """
        if len(self._ic_series) < 10:
            return np.nan

        ics = pd.Series(
            [v for _, v in self._ic_series],
            index=[t for t, _ in self._ic_series],
        ).dropna()

        if len(ics) < 10:
            return np.nan

        max_lag = min(20, len(ics) // 2)
        autocorrs = np.array([ics.autocorr(lag=lag) for lag in range(1, max_lag + 1)])
        # Find first lag where autocorrelation drops below 0.5
        below_half = np.where(autocorrs < 0.5)[0]
        if len(below_half) == 0:
            return float(max_lag)  # very persistent signal
        return float(below_half[0] + 1)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Serialise model configuration to dictionary."""
        return {
            "name": self.name,
            "lookback": self.lookback,
            "holding_period": self.holding_period,
            "norm": self.norm.value,
            "winsorize_std": self.winsorize_std,
            "direction": self.direction.value,
            "min_history": self.min_history,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"lookback={self.lookback}, hold={self.holding_period})"
        )
