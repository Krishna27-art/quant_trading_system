"""
Microstructure Alpha Models — Indian Markets (NSE/BSE)

Implements:
1. OrderFlowImbalance: Buy/sell volume imbalance using tick-rule classification.
2. VolumeWeightedPressure: VWAP-deviation pressure with volume clustering.
3. AmihudIlliquidity: Amihud (2002) illiquidity ratio — price impact per unit volume.
4. KyleLambda: Kyle's lambda (price impact coefficient) from intraday data.

Designed for Indian market microstructure where NSE provides trade-level
data with buy/sell flags when available, or tick-rule classification as
fallback.
"""

import numpy as np
import pandas as pd

from portfolio_execution.signals.base import AlphaModel, SignalDirection, SignalNorm
from utils.logger import get_logger

logger = get_logger(__name__)


class OrderFlowImbalance(AlphaModel):
    """
    Order flow imbalance (OFI): net buying pressure.

    For each stock, classify trades as buyer- or seller-initiated
    (using tick rule if explicit side is unavailable), then compute:

        OFI = (buy_volume − sell_volume) / total_volume

    The cross-sectional z-score of OFI predicts short-term returns
    (informed flow leads price).

    Parameters
    ----------
    lookback : int
        Number of bars (periods) to aggregate OFI over.
    decay_factor : float
        Exponential decay for weighting recent bars more heavily.
        1.0 = equal weight, <1.0 = more weight on recent.
    """

    def __init__(
        self,
        lookback: int = 20,
        decay_factor: float = 0.94,
        norm: SignalNorm = SignalNorm.ZSCORE,
        **kwargs,
    ):
        super().__init__(
            name="order_flow_imbalance",
            lookback=lookback,
            norm=norm,
            direction=SignalDirection.LONG_SHORT,
            **kwargs,
        )
        self.decay_factor = decay_factor

    @staticmethod
    def _tick_rule_classify(
        prices: pd.Series,
        volumes: pd.Series,
    ) -> pd.DataFrame:
        """
        Classify volume as buy- or sell-initiated using the tick rule.

        Tick rule: if price_t > price_{t-1} → buy; else → sell.
        Zero-tick: use previous classification.

        Returns DataFrame with columns: buy_volume, sell_volume.
        """
        price_diff = prices.diff()

        # Forward-fill zero ticks with last non-zero direction
        direction = np.sign(price_diff)
        direction = direction.replace(0, np.nan).ffill().fillna(1.0)

        buy_vol = volumes.where(direction > 0, 0.0)
        sell_vol = volumes.where(direction < 0, 0.0)

        return pd.DataFrame(
            {
                "buy_volume": buy_vol,
                "sell_volume": sell_vol,
            },
            index=prices.index,
        )

    def _compute_raw_signal(
        self,
        data: pd.DataFrame,
        prices: pd.DataFrame | None = None,
        volumes: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        """
        Compute OFI signal.

        If `prices` and `volumes` are provided (wide DataFrames), use
        tick-rule classification.  Otherwise, expects `data` to have
        columns: symbol, close, volume (long format with date index),
        or a wide DataFrame of close prices with a corresponding
        `volumes` argument.
        """
        if prices is None:
            prices = data
        if volumes is None:
            if "volume" in data.columns:
                volumes = (
                    data["volume"].unstack() if isinstance(data.index, pd.MultiIndex) else data
                )
            else:
                # Assume data is wide-format close; need volume from kwargs
                logger.warning("No volume data provided for OFI")
                return pd.Series(dtype=float)

        if prices.shape[0] < self.lookback:
            return pd.Series(dtype=float)

        p = prices.iloc[-self.lookback :]
        v = volumes.iloc[-self.lookback :]

        # filter out symbols with too many NaNs
        valid_cols = p.columns[p.isna().sum() <= self.lookback * 0.3]
        p = p[valid_cols]
        v = v[valid_cols]

        price_diff = p.diff()
        direction = np.sign(price_diff).replace(0, np.nan).ffill().fillna(1.0)

        buy_vol = v.where(direction > 0, 0.0)
        sell_vol = v.where(direction < 0, 0.0)

        total_vol = buy_vol + sell_vol
        total_vol = total_vol.replace(0, np.nan)

        imbalance = (buy_vol - sell_vol) / total_vol

        # valid mask
        valid_mask = imbalance.notna()

        weights = np.array([self.decay_factor**i for i in range(self.lookback - 1, -1, -1)])
        weights = weights / weights.sum()

        # w_matrix needs to be broadcasted to valid_mask shape
        w_matrix = np.tile(weights[:, np.newaxis], (1, imbalance.shape[1]))
        w_matrix = np.where(valid_mask, w_matrix, 0.0)
        w_sum = w_matrix.sum(axis=0)

        # Filter symbols with insufficient valid points
        insufficient = valid_mask.sum(axis=0) < self.lookback * 0.5
        w_sum[insufficient | (w_sum < 1e-10)] = np.nan

        weighted_imbalance = (imbalance.fillna(0.0) * w_matrix).sum(axis=0) / w_sum

        return pd.Series(weighted_imbalance, index=p.columns).dropna()


class VolumeWeightedPressure(AlphaModel):
    """
    VWAP deviation pressure: how far the current price is from VWAP,
    weighted by volume concentration.

    Signal = −(close − VWAP) / ATR × volume_zscore

    Negative deviation with high volume → buying pressure → go long.

    Parameters
    ----------
    lookback : int
        Window for VWAP and ATR computation.
    atr_lookback : int
        Window for Average True Range.
    """

    def __init__(
        self,
        lookback: int = 20,
        atr_lookback: int = 14,
        norm: SignalNorm = SignalNorm.ZSCORE,
        **kwargs,
    ):
        super().__init__(
            name="vwap_pressure",
            lookback=lookback,
            norm=norm,
            direction=SignalDirection.LONG_SHORT,
            **kwargs,
        )
        self.atr_lookback = atr_lookback

    @staticmethod
    def _compute_vwap(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> float:
        """Compute VWAP = Σ(typical_price × volume) / Σ(volume)."""
        typical = (high + low + close) / 3.0
        total_vol = volume.sum()
        if total_vol < 1:
            return float(close.iloc[-1])
        return float((typical * volume).sum() / total_vol)

    @staticmethod
    def _compute_atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int,
    ) -> float:
        """Average True Range over a window."""
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return float(tr.iloc[-period:].mean())

    def _compute_raw_signal(
        self,
        data: pd.DataFrame,
        ohlcv: dict[str, pd.DataFrame] | None = None,
        **kwargs,
    ) -> pd.Series:
        """
        Compute VWAP pressure signal.

        Parameters
        ----------
        data : pd.DataFrame
            If ohlcv is None, expects MultiIndex (date, symbol) with
            columns: open, high, low, close, volume.
            If ohlcv is provided, data is ignored.
        ohlcv : dict
            Mapping symbol → DataFrame with columns (open, high, low,
            close, volume) indexed by date.
        """
        if ohlcv is None:
            if isinstance(data.index, pd.MultiIndex):
                symbols = data.index.get_level_values(1).unique()
                ohlcv = {}
                for sym in symbols:
                    try:
                        ohlcv[sym] = data.xs(sym, level=1)
                    except KeyError:
                        continue
            else:
                logger.warning("VolumeWeightedPressure needs OHLCV data")
                return pd.Series(dtype=float)

        signals = {}
        for sym, df in ohlcv.items():
            if len(df) < max(self.lookback, self.atr_lookback) + 1:
                continue

            h = df["high"].iloc[-self.lookback :]
            low = df["low"].iloc[-self.lookback :]
            c = df["close"].iloc[-self.lookback :]
            v = df["volume"].iloc[-self.lookback :]

            vwap = self._compute_vwap(h, low, c, v)
            atr = self._compute_atr(
                df["high"],
                df["low"],
                df["close"],
                self.atr_lookback,
            )

            if atr < 1e-6:
                continue

            current_close = float(c.iloc[-1])

            # Volume z-score: how concentrated is today's volume?
            vol_mean = v.mean()
            vol_std = v.std()
            vol_z = 0.0 if vol_std < 1 else (float(v.iloc[-1]) - vol_mean) / vol_std

            # Pressure = −deviation × volume_z
            deviation = (current_close - vwap) / atr
            pressure = -deviation * max(vol_z, 0.0)
            signals[sym] = pressure

        return pd.Series(signals, dtype=float)


class AmihudIlliquidity(AlphaModel):
    """
    Amihud (2002) illiquidity ratio.

    ILLIQ = (1/N) Σ |r_t| / volume_t

    High illiquidity → price moves easily per unit volume → carries a
    premium (illiquidity premium).  Signal direction: stocks becoming
    more liquid (falling ILLIQ) may be underpriced.

    Parameters
    ----------
    lookback : int
        Window for averaging illiquidity (trading days).
    change_lookback : int
        Window for computing change in illiquidity (for momentum in
        liquidity).
    """

    def __init__(
        self,
        lookback: int = 21,
        change_lookback: int = 63,
        norm: SignalNorm = SignalNorm.ZSCORE,
        **kwargs,
    ):
        super().__init__(
            name="amihud_illiquidity",
            lookback=lookback,
            norm=norm,
            direction=SignalDirection.LONG_SHORT,
            **kwargs,
        )
        self.change_lookback = change_lookback

    def _compute_raw_signal(
        self,
        data: pd.DataFrame,
        volumes: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        """
        data : wide DataFrame (dates × symbols) of adjusted close prices.
        volumes : wide DataFrame (dates × symbols) of traded volume
                  (in rupees or shares × price).
        """
        if volumes is None:
            logger.warning("AmihudIlliquidity requires volume data")
            return pd.Series(dtype=float)

        required = max(self.lookback, self.change_lookback) + 5
        if data.shape[0] < required:
            return pd.Series(dtype=float)

        returns = data.pct_change().abs()

        # Volume in rupee terms (close × volume)
        rupee_vol = data * volumes
        rupee_vol = rupee_vol.replace(0.0, np.nan)

        # Current illiquidity (recent window)
        illiq_recent = (returns.iloc[-self.lookback :] / rupee_vol.iloc[-self.lookback :]).mean()

        # Past illiquidity (earlier window)
        if data.shape[0] >= self.change_lookback + self.lookback:
            illiq_past = (
                returns.iloc[-(self.change_lookback + self.lookback) : -self.change_lookback]
                / rupee_vol.iloc[-(self.change_lookback + self.lookback) : -self.change_lookback]
            ).mean()
        else:
            illiq_past = illiq_recent

        # Change in illiquidity: falling illiquidity → positive signal
        illiq_change = -(illiq_recent - illiq_past) / illiq_past.replace(0.0, np.nan)

        return illiq_change.dropna()


class KyleLambda(AlphaModel):
    """
    Kyle's Lambda: price impact coefficient.

    Regress absolute returns on signed volume (buy - sell) to estimate
    the permanent price impact per unit of order flow.  Higher lambda →
    more informed trading / less liquid.

    λ = Cov(Δp, OFI) / Var(OFI)

    Parameters
    ----------
    lookback : int
        Estimation window (trading days / bars).
    """

    def __init__(
        self,
        lookback: int = 60,
        norm: SignalNorm = SignalNorm.ZSCORE,
        **kwargs,
    ):
        super().__init__(
            name="kyle_lambda",
            lookback=lookback,
            norm=norm,
            direction=SignalDirection.LONG_SHORT,
            **kwargs,
        )

    def _compute_raw_signal(
        self,
        data: pd.DataFrame,
        volumes: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        """
        data : wide DataFrame (dates × symbols) of close prices.
        volumes : wide DataFrame of traded volume.
        """
        if volumes is None:
            return pd.Series(dtype=float)

        if data.shape[0] < self.lookback + 2:
            return pd.Series(dtype=float)

        returns = data.pct_change()
        price_diff = data.diff()

        # Signed order flow using tick-rule direction
        direction = np.sign(price_diff)
        direction = direction.replace(0, np.nan).ffill().fillna(1.0)
        signed_volume = direction * volumes

        signals = {}
        for sym in data.columns:
            if sym not in volumes.columns:
                continue

            dp = returns[sym].iloc[-self.lookback :].values
            sv = signed_volume[sym].iloc[-self.lookback :].values

            mask = np.isfinite(dp) & np.isfinite(sv)
            if mask.sum() < 20:
                continue

            dp_clean = dp[mask]
            sv_clean = sv[mask]

            sv_var = np.var(sv_clean)
            if sv_var < 1e-15:
                continue

            # Kyle's lambda = Cov(dp, sv) / Var(sv)
            lam = np.cov(dp_clean, sv_clean)[0, 1] / sv_var

            # Negative lambda change → market becoming more liquid → positive signal
            signals[sym] = -lam

        return pd.Series(signals, dtype=float)


class BidAskBounceNeutralizer:
    """
    Neutralizes signals against the Bid/Ask bounce using the Roll (1984) measure.
    High Roll measure indicates high bid/ask bounce which can create artificial mean reversion.
    """

    @staticmethod
    def compute_roll_measure(close_prices: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Compute Roll's measure of effective spread: 2 * sqrt(max(0, -Cov(dp_t, dp_{t-1})))
        """
        dp = close_prices.diff()
        dp_lag = dp.shift(1)

        roll_measures = {}
        for sym in close_prices.columns:
            cov = dp[sym].iloc[-window:].cov(dp_lag[sym].iloc[-window:])
            if pd.isna(cov) or cov >= 0:
                roll_measures[sym] = 0.0
            else:
                roll_measures[sym] = 2 * np.sqrt(-cov)

        return pd.Series(roll_measures)

    @staticmethod
    def neutralize(signal: pd.Series, close_prices: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Dampen the signal for stocks with high bid-ask bounce.
        """
        roll = BidAskBounceNeutralizer.compute_roll_measure(close_prices, window)
        # Normalize roll to 0-1
        roll_norm = roll / roll.max() if roll.max() > 0 else pd.Series(0.0, index=roll.index)

        # Penalize signal: higher roll measure = higher penalty
        penalty = 1.0 - roll_norm

        # Ensure indices match
        common = signal.index.intersection(penalty.index)
        return signal[common] * penalty[common]


class BorrowingCostAdjuster:
    """
    Adjusts alpha signals to account for borrowing costs (hard-to-borrow stocks).
    Reduces the strength of short signals if borrowing costs are prohibitive.
    """

    @staticmethod
    def adjust(
        signal: pd.Series, borrow_fees: pd.Series, max_fee_threshold: float = 0.10
    ) -> pd.Series:
        """
        signal: cross-sectional alpha signal
        borrow_fees: annualized borrow fee rate (e.g., 0.05 for 5%)
        max_fee_threshold: above this fee, shorts are completely zeroed out
        """
        adjusted = signal.copy()

        common = signal.index.intersection(borrow_fees.index)
        if common.empty:
            return adjusted

        short_mask = adjusted[common] < 0

        # Calculate penalty: linearly scales from 1.0 at 0% fee to 0.0 at max_fee_threshold
        fees = borrow_fees[common]
        penalty = 1.0 - (fees / max_fee_threshold)
        penalty = penalty.clip(0.0, 1.0)

        adjusted.loc[common[short_mask]] *= penalty[short_mask]

        return adjusted
