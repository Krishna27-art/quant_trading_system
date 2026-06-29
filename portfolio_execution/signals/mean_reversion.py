"""
Mean Reversion Alpha Models — Indian Markets

Implements:
1. ResidualMeanReversion: Z-score of OLS residual returns (market-neutral).
2. PairsCointegration: Engle–Granger cointegration-based pairs trading.
3. OrnsteinUhlenbeck: OU-process calibration for mean-reverting spreads.
4. BollingerMeanReversion: Bollinger-band z-scores with adaptive bandwidth.

All signals output standardized z-scores via the AlphaModel base class.
"""

import numpy as np
import pandas as pd

from portfolio_execution.signals.base import AlphaModel, SignalDirection, SignalNorm
from utils.logger import get_logger

logger = get_logger(__name__)


class ResidualMeanReversion(AlphaModel):
    """
    Statistical mean-reversion on residual returns.

    For each stock, regress returns on the cross-sectional mean (market)
    return using a rolling OLS.  The z-score of the residual cumulative
    return is the signal: large negative z → expected to revert upward.

    Parameters
    ----------
    lookback : int
        Rolling window for OLS regression (trading days).
    zscore_lookback : int
        Window for computing z-score of residual (trading days).
    entry_z : float
        Z-score threshold for entry (absolute value).
    """

    def __init__(
        self,
        lookback: int = 60,
        zscore_lookback: int = 20,
        entry_z: float = 2.0,
        norm: SignalNorm = SignalNorm.ZSCORE,
        **kwargs,
    ):
        super().__init__(
            name="residual_mean_reversion",
            lookback=lookback,
            norm=norm,
            direction=SignalDirection.LONG_SHORT,
            **kwargs,
        )
        self.zscore_lookback = zscore_lookback
        self.entry_z = entry_z

    def _compute_raw_signal(
        self,
        data: pd.DataFrame,
        **kwargs,
    ) -> pd.Series:
        """
        data : wide DataFrame (dates × symbols) of adjusted close prices.

        Returns z-score of residual cumulative return (negative z = buy signal).
        """
        required = self.lookback + self.zscore_lookback * 2
        if data.shape[0] < required:
            return pd.Series(dtype=float)

        log_ret = np.log(data / data.shift(1)).dropna(how="all")
        market_ret = log_ret.mean(axis=1).to_frame(
            name="market"
        )  # equal-weighted market return as factor

        from sklearn.linear_model import Ridge

        T = len(log_ret)
        residuals = pd.DataFrame(np.nan, index=log_ret.index, columns=log_ret.columns)

        # Correct — expanding window, only past data at each point
        for t in range(self.lookback, T):
            # Only use data strictly before time t
            X_past = market_ret.iloc[:t].values
            Y_past = log_ret.iloc[:t].values

            # Fit one model per stock
            betas = []
            for n in range(Y_past.shape[1]):
                y = Y_past[:, n]
                valid = ~np.isnan(y)
                if valid.sum() < 20:
                    betas.append(np.zeros(X_past.shape[1]))
                    continue
                model = Ridge(alpha=1.0, fit_intercept=True)
                model.fit(X_past[valid], y[valid])
                betas.append(model.coef_)

            betas = np.array(betas)  # shape: (N_stocks, N_factors)

            # Current period residual using PAST betas only
            x_now = market_ret.iloc[t].values
            y_now = log_ret.iloc[t].values
            resid = y_now - (betas @ x_now)
            residuals.iloc[t] = resid

        signals = {}
        for sym in residuals.columns:
            resid_col = residuals[sym].dropna()
            if len(resid_col) < self.zscore_lookback:
                continue

            # Cumulative residual over zscore_lookback
            rolling_cum = resid_col.rolling(self.zscore_lookback).sum().dropna()
            if len(rolling_cum) < self.zscore_lookback:
                continue

            mu = rolling_cum.rolling(self.zscore_lookback).mean()
            sigma = rolling_cum.rolling(self.zscore_lookback).std()

            current_cum = rolling_cum.iloc[-1]
            current_mu = mu.iloc[-1]
            current_sigma = sigma.iloc[-1]

            if pd.isna(current_sigma) or current_sigma < 1e-10:
                continue

            z = (current_cum - current_mu) / current_sigma
            # Negative z-score → signal to go long (expecting reversion up)
            signals[sym] = -z

        return pd.Series(signals, dtype=float)


class PairsCointegration(AlphaModel):
    """
    Engle–Granger cointegration framework for pairs trading.

    Given a pre-defined pair (or auto-detected pairs), compute:
    1. Cointegration test (ADF on OLS residual of log prices).
    2. Hedge ratio via OLS.
    3. Spread z-score as signal.

    Parameters
    ----------
    lookback : int
        Window for cointegration estimation.
    zscore_lookback : int
        Window for z-scoring the spread.
    adf_pvalue : float
        Maximum p-value for ADF test to consider pair cointegrated.
    """

    def __init__(
        self,
        lookback: int = 252,
        zscore_lookback: int = 20,
        adf_pvalue: float = 0.05,
        norm: SignalNorm = SignalNorm.RAW,
        **kwargs,
    ):
        super().__init__(
            name="pairs_cointegration",
            lookback=lookback,
            norm=norm,
            direction=SignalDirection.LONG_SHORT,
            **kwargs,
        )
        self.zscore_lookback = zscore_lookback
        self.adf_pvalue = adf_pvalue

    @staticmethod
    def _adf_test(series: np.ndarray, max_lags: int = 10) -> tuple[float, float]:
        """
        Augmented Dickey-Fuller test on a 1-D series.

        Returns (t-stat, p-value) using Mackinnon approximate p-values.
        """
        n = len(series)
        if n < 20:
            return 0.0, 1.0

        # First difference
        dy = np.diff(series)
        y_lag = series[:-1]

        # Build regression matrix:  Δy_t = α + γ * y_{t-1} + Σ β_i Δy_{t-i} + ε
        n_lags = min(max_lags, n // 5)
        n_obs = len(dy) - n_lags
        if n_obs < 10:
            return 0.0, 1.0

        x_cols = [np.ones(n_obs), y_lag[n_lags:]]
        for i in range(1, n_lags + 1):
            x_cols.append(dy[n_lags - i : -i] if i < len(dy) else dy[:n_obs])

        X = np.column_stack(x_cols)
        Y = dy[n_lags:]

        try:
            betas, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        except np.linalg.LinAlgError:
            return 0.0, 1.0

        resid = Y - X @ betas
        se = np.sqrt(np.sum(resid**2) / (n_obs - len(betas)))

        # Standard error of gamma (coefficient on y_{t-1})
        try:
            cov = se**2 * np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            return 0.0, 1.0

        gamma_se = np.sqrt(cov[1, 1])
        if gamma_se < 1e-15:
            return 0.0, 1.0

        t_stat = betas[1] / gamma_se

        # Mackinnon approximate p-value for constant-only case
        # Using standard approximation: critical values -3.43, -2.86, -2.57
        if t_stat < -3.43:
            p_val = 0.01
        elif t_stat < -2.86:
            p_val = 0.05
        elif t_stat < -2.57:
            p_val = 0.10
        else:
            p_val = min(1.0, 0.10 + 0.3 * (t_stat + 2.57))

        return float(t_stat), float(max(0.0, min(1.0, p_val)))

    def find_cointegrated_pairs(
        self,
        prices: pd.DataFrame,
        candidates: list[tuple[str, str]] | None = None,
        top_n: int = 20,
    ) -> list[dict]:
        """
        Scan candidate pairs for cointegration.

        Parameters
        ----------
        prices : pd.DataFrame
            Wide format (dates × symbols) of log-prices.
        candidates : list of (sym_a, sym_b) tuples, optional
            If None, scans all combinations (expensive for >100 symbols).
        top_n : int
            Max pairs to return.

        Returns
        -------
        list of dict with keys: sym_a, sym_b, hedge_ratio, adf_stat,
        adf_pval, spread_mean, spread_std.
        """
        log_prices = np.log(prices.dropna(axis=1, how="any"))
        symbols = log_prices.columns.tolist()

        if candidates is None:
            # Generate all unique pairs (capped at 200 symbols)
            if len(symbols) > 200:
                logger.warning("Too many symbols for exhaustive scan; pass explicit candidates.")
                symbols = symbols[:200]
            candidates = [
                (symbols[i], symbols[j])
                for i in range(len(symbols))
                for j in range(i + 1, len(symbols))
            ]

        results = []
        for sym_a, sym_b in candidates:
            if sym_a not in log_prices.columns or sym_b not in log_prices.columns:
                continue

            y = log_prices[sym_a].values
            x = log_prices[sym_b].values

            if len(y) < self.lookback:
                continue

            y_w = y[-self.lookback :]
            x_w = x[-self.lookback :]

            # OLS hedge ratio
            x_mat = np.column_stack([np.ones(len(x_w)), x_w])
            try:
                betas, _, _, _ = np.linalg.lstsq(x_mat, y_w, rcond=None)
            except np.linalg.LinAlgError:
                continue

            spread = y_w - betas[0] - betas[1] * x_w
            adf_stat, adf_pval = self._adf_test(spread)

            if adf_pval <= self.adf_pvalue:
                results.append(
                    {
                        "sym_a": sym_a,
                        "sym_b": sym_b,
                        "hedge_ratio": float(betas[1]),
                        "intercept": float(betas[0]),
                        "adf_stat": adf_stat,
                        "adf_pval": adf_pval,
                        "spread_mean": float(spread.mean()),
                        "spread_std": float(spread.std()),
                    }
                )

        results.sort(key=lambda r: r["adf_pval"])
        return results[:top_n]

    def compute_spread_zscore(
        self,
        prices: pd.DataFrame,
        sym_a: str,
        sym_b: str,
        hedge_ratio: float,
        intercept: float = 0.0,
    ) -> pd.Series:
        """Compute rolling z-score of the cointegrated spread using dynamic rolling beta."""
        lp = np.log(prices[[sym_a, sym_b]].dropna())
        y = lp[sym_a]
        x = lp[sym_b]

        # Rolling hedge ratio to prevent lookahead bias
        cov = y.rolling(self.lookback).cov(x)
        var = x.rolling(self.lookback).var()

        rolling_hr = cov / var
        rolling_intercept = (
            y.rolling(self.lookback).mean() - rolling_hr * x.rolling(self.lookback).mean()
        )

        # Dynamic spread
        spread = y - rolling_intercept - rolling_hr * x

        # shift(1) ensures current bar not included in its own rolling window
        mu = spread.shift(1).rolling(self.zscore_lookback).mean()
        sigma = spread.shift(1).rolling(self.zscore_lookback).std()
        sigma = sigma.replace(0.0, np.nan)
        z = (spread - mu) / sigma
        return z

    def _compute_raw_signal(
        self,
        data: pd.DataFrame,
        pairs: list[dict] | None = None,
        **kwargs,
    ) -> pd.Series:
        """
        Compute aggregate mean-reversion signal across cointegrated pairs.

        For each pair where the spread z-score is extreme, assign signal:
          - sym_a gets −z (short when spread is high, long when low)
          - sym_b gets +z × hedge_ratio
        Signals are averaged across all pairs per symbol.
        """
        if pairs is None:
            pairs = self.find_cointegrated_pairs(data)

        symbol_signals: dict[str, list[float]] = {}

        for pair in pairs:
            z_series = self.compute_spread_zscore(
                data,
                pair["sym_a"],
                pair["sym_b"],
                pair["hedge_ratio"],
                pair.get("intercept", 0.0),
            )
            if z_series.dropna().empty:
                continue

            z_val = z_series.dropna().iloc[-1]

            # sym_a signal: negative z means spread reverts down → short sym_a
            symbol_signals.setdefault(pair["sym_a"], []).append(-z_val)
            symbol_signals.setdefault(pair["sym_b"], []).append(z_val * pair["hedge_ratio"])

        # Average across all pair signals per symbol
        return pd.Series(
            {sym: np.mean(sigs) for sym, sigs in symbol_signals.items()},
            dtype=float,
        )


class OrnsteinUhlenbeck:
    """
    Ornstein–Uhlenbeck process calibration for mean-reverting spreads.

    dX_t = θ (μ − X_t) dt + σ dW_t

    Calibrates θ (speed of reversion), μ (long-run mean), σ (volatility)
    from historical spread data.
    """

    def __init__(self, dt: float = 1.0 / 252.0):
        self.dt = dt
        self.theta: float = 0.0
        self.mu: float = 0.0
        self.sigma: float = 0.0
        self.halflife: float = np.nan

    def fit(self, spread: np.ndarray) -> "OrnsteinUhlenbeck":
        """
        Calibrate OU parameters via maximum likelihood.

        Parameters
        ----------
        spread : np.ndarray
            Time series of the spread.
        """
        n = len(spread)
        if n < 20:
            return self

        x = spread[:-1]
        y = spread[1:]

        # AR(1) regression: y = a + b * x + eps
        x_mat = np.column_stack([np.ones(len(x)), x])
        try:
            betas, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
        except np.linalg.LinAlgError:
            return self

        a, b = betas[0], betas[1]
        resid = y - x_mat @ betas
        sigma_e = np.std(resid, ddof=2)

        if b >= 1.0 or b <= 0.0:
            # Not mean-reverting
            self.theta = 0.0
            self.mu = float(np.mean(spread))
            self.sigma = float(np.std(spread))
            self.halflife = np.inf
            return self

        self.theta = -np.log(b) / self.dt
        self.mu = a / (1.0 - b)
        self.sigma = sigma_e * np.sqrt(-2.0 * np.log(b) / (self.dt * (1.0 - b**2)))
        self.halflife = np.log(2.0) / self.theta if self.theta > 0 else np.inf

        return self

    def zscore(self, current_value: float) -> float:
        """Compute z-score of current spread vs OU equilibrium."""
        eq_var = self.sigma**2 / (2.0 * self.theta) if self.theta > 0 else 1.0
        eq_std = np.sqrt(eq_var) if eq_var > 0 else 1.0
        return (current_value - self.mu) / eq_std

    def expected_value(self, current_value: float, horizon: float) -> float:
        """Expected spread value at given horizon (in years)."""
        return self.mu + (current_value - self.mu) * np.exp(-self.theta * horizon)


class BollingerMeanReversion(AlphaModel):
    """
    Bollinger-band z-score mean reversion.

    Signal = −(price − SMA) / (k × rolling_std)

    Negative signal → price above band → short.
    Positive signal → price below band → long.

    Parameters
    ----------
    lookback : int
        SMA / std rolling window (trading days).
    num_std : float
        Number of standard deviations for band width.
    adaptive : bool
        If True, adjust bandwidth based on recent volatility regime.
    """

    def __init__(
        self,
        lookback: int = 20,
        num_std: float = 2.0,
        adaptive: bool = True,
        norm: SignalNorm = SignalNorm.ZSCORE,
        **kwargs,
    ):
        super().__init__(
            name="bollinger_mean_reversion",
            lookback=lookback,
            norm=norm,
            direction=SignalDirection.LONG_SHORT,
            **kwargs,
        )
        self.num_std = num_std
        self.adaptive = adaptive

    def _compute_raw_signal(
        self,
        data: pd.DataFrame,
        **kwargs,
    ) -> pd.Series:
        """data : wide DataFrame (dates × symbols) of adjusted close."""
        if data.shape[0] < self.lookback + 10:
            return pd.Series(dtype=float)

        sma = data.iloc[-self.lookback :].mean()
        std = data.iloc[-self.lookback :].std()

        if self.adaptive:
            # Adjust bandwidth by comparing recent vol to longer-term vol
            long_std = (
                data.iloc[-self.lookback * 3 :].std() if data.shape[0] >= self.lookback * 3 else std
            )
            vol_ratio = std / long_std.replace(0.0, np.nan)
            vol_ratio = vol_ratio.clip(0.5, 2.0)
            effective_k = self.num_std * vol_ratio
        else:
            effective_k = self.num_std

        current_price = data.iloc[-1]
        std = std.replace(0.0, np.nan)

        z = (current_price - sma) / (effective_k * std)

        # Negative z-score for mean reversion signal (high z → short)
        return (-z).dropna()
