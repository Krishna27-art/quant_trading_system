"""
Regime-Conditioned Alpha Models — Indian Markets

Implements:
1. RegimeClassifier: Detects market regime (trending, mean-reverting,
   high-vol) using ADX, VIX (India VIX), and MA structure.
2. RegimeConditionedAlpha: Wrapper that activates/deactivates child
   alpha models based on the current regime.
3. AdaptiveRegimeBlend: Continuously blends alpha models with
   regime-dependent weights.

Indian-specific: Uses India VIX (^INDIAVIX on NSE) and Nifty 50/200
DMA structure for regime detection.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from portfolio_execution.signals.base import AlphaModel, SignalDirection, SignalNorm
from utils.logger import get_logger

logger = get_logger(__name__)


class MarketRegime(Enum):
    """Market regime categories."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


@dataclass
class RegimeState:
    """Snapshot of regime classification at a point in time."""

    timestamp: pd.Timestamp
    regime: MarketRegime
    adx: float = 0.0
    vix: float = 0.0
    ma_score: float = 0.0  # positive = bullish MA structure
    confidence: float = 0.0  # 0–1 confidence in classification
    metadata: dict = field(default_factory=dict)


class RegimeClassifier:
    """
    Multi-factor regime classifier for Indian equity markets.

    Classification logic:
    - ADX > adx_trend_thresh → trending
      - Price > SMA → trending_up; else trending_down
    - ADX < adx_range_thresh → mean-reverting
    - VIX > vix_high_thresh → high_volatility (overrides trend)
    - VIX < vix_low_thresh → low_volatility

    Parameters
    ----------
    adx_period : int
        ADX calculation period.
    adx_trend_thresh : float
        ADX above this = trending.
    adx_range_thresh : float
        ADX below this = mean-reverting / range-bound.
    vix_high_thresh : float
        India VIX above this = high-vol regime.
    vix_low_thresh : float
        India VIX below this = low-vol regime.
    sma_short : int
        Short moving average (days) for MA structure.
    sma_long : int
        Long moving average (days) for MA structure.
    """

    def __init__(
        self,
        adx_period: int = 14,
        adx_trend_thresh: float = 25.0,
        adx_range_thresh: float = 20.0,
        vix_high_thresh: float = 22.0,
        vix_low_thresh: float = 13.0,
        sma_short: int = 50,
        sma_long: int = 200,
    ):
        self.adx_period = adx_period
        self.adx_trend_thresh = adx_trend_thresh
        self.adx_range_thresh = adx_range_thresh
        self.vix_high_thresh = vix_high_thresh
        self.vix_low_thresh = vix_low_thresh
        self.sma_short = sma_short
        self.sma_long = sma_long
        self._history: list[RegimeState] = []

    def compute_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """
        Compute Average Directional Index (ADX).

        ADX = smoothed(|+DI − −DI| / (+DI + −DI)) × 100
        """
        n = self.adx_period

        # True Range
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=high.index,
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=high.index,
        )

        # Wilder smoothing (EMA with alpha = 1/n)
        alpha = 1.0 / n
        atr = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_di = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
        minus_di = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr

        di_sum = plus_di + minus_di
        di_sum = di_sum.replace(0.0, np.nan)
        dx = 100.0 * (plus_di - minus_di).abs() / di_sum
        adx = dx.ewm(alpha=alpha, adjust=False).mean()

        return adx

    def compute_ma_structure(
        self,
        close: pd.Series,
    ) -> pd.Series:
        """
        Moving average structure score.

        Score = (SMA_short / SMA_long − 1) × 100

        Positive → price above long MA (bullish structure).
        """
        sma_s = close.rolling(self.sma_short).mean()
        sma_l = close.rolling(self.sma_long).mean()
        sma_l = sma_l.replace(0.0, np.nan)
        return (sma_s / sma_l - 1.0) * 100.0

    def classify(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        vix: pd.Series | None = None,
        timestamp: pd.Timestamp | None = None,
    ) -> RegimeState:
        """
        Classify current market regime.

        Parameters
        ----------
        high, low, close : pd.Series
            Index-level (e.g. Nifty 50) OHLC series.
        vix : pd.Series, optional
            India VIX series (same date index as OHLC).

        Returns
        -------
        RegimeState
        """
        ts = timestamp or pd.Timestamp.now(tz="Asia/Kolkata")
        adx_series = self.compute_adx(high, low, close)
        ma_score = self.compute_ma_structure(close)

        current_adx = float(adx_series.iloc[-1]) if len(adx_series) > 0 else 0.0
        current_ma = float(ma_score.iloc[-1]) if len(ma_score) > 0 else 0.0
        current_vix = float(vix.iloc[-1]) if vix is not None and len(vix) > 0 else 15.0

        # Classification hierarchy
        if current_vix >= self.vix_high_thresh:
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = min(1.0, (current_vix - self.vix_high_thresh) / 10.0 + 0.5)
        elif current_adx >= self.adx_trend_thresh:
            regime = MarketRegime.TRENDING_UP if current_ma > 0 else MarketRegime.TRENDING_DOWN
            confidence = min(1.0, (current_adx - self.adx_trend_thresh) / 15.0 + 0.5)
        elif current_adx <= self.adx_range_thresh:
            regime = MarketRegime.MEAN_REVERTING
            confidence = min(1.0, (self.adx_range_thresh - current_adx) / 10.0 + 0.5)
        elif current_vix <= self.vix_low_thresh:
            regime = MarketRegime.LOW_VOLATILITY
            confidence = min(1.0, (self.vix_low_thresh - current_vix) / 5.0 + 0.5)
        else:
            regime = MarketRegime.UNKNOWN
            confidence = 0.3

        state = RegimeState(
            timestamp=ts,
            regime=regime,
            adx=current_adx,
            vix=current_vix,
            ma_score=current_ma,
            confidence=confidence,
        )
        self._history.append(state)
        logger.info(
            f"Regime classified: {regime.value}",
            extra={"adx": f"{current_adx:.1f}", "vix": f"{current_vix:.1f}"},
        )
        return state

    def regime_history(self) -> pd.DataFrame:
        """Return regime history as DataFrame."""
        if not self._history:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "timestamp": s.timestamp,
                    "regime": s.regime.value,
                    "adx": s.adx,
                    "vix": s.vix,
                    "ma_score": s.ma_score,
                    "confidence": s.confidence,
                }
                for s in self._history
            ]
        ).set_index("timestamp")

    def regime_duration(self) -> dict[str, int]:
        """Count consecutive periods each regime has been active."""
        if not self._history:
            return {}
        counts: dict[str, int] = {}
        current = self._history[-1].regime.value
        count = 0
        for state in reversed(self._history):
            if state.regime.value == current:
                count += 1
            else:
                break
        counts[current] = count
        return counts


class RegimeConditionedAlpha(AlphaModel):
    """
    Wrapper that activates/deactivates child alpha models based on
    the current market regime.

    Configuration is a mapping from MarketRegime → list of AlphaModel.
    Only models mapped to the current regime are evaluated; the rest
    return zero signal.

    Parameters
    ----------
    regime_model_map : dict
        MarketRegime → list of AlphaModel.
    classifier : RegimeClassifier
        Regime classifier instance.
    blend_weights : dict, optional
        MarketRegime → dict of {model_name: weight}.
        If None, equal weights within each regime.
    """

    def __init__(
        self,
        regime_model_map: dict[MarketRegime, list[AlphaModel]],
        classifier: RegimeClassifier,
        blend_weights: dict[MarketRegime, dict[str, float]] | None = None,
        norm: SignalNorm = SignalNorm.ZSCORE,
        **kwargs,
    ):
        all_models = []
        for models in regime_model_map.values():
            all_models.extend(models)
        model_names = [m.name for m in all_models]

        super().__init__(
            name="regime_conditioned_" + "_".join(model_names[:3]),
            lookback=max(m.lookback for m in all_models) if all_models else 252,
            norm=norm,
            direction=SignalDirection.LONG_SHORT,
            **kwargs,
        )
        self.regime_model_map = regime_model_map
        self.classifier = classifier
        self.blend_weights = blend_weights or {}

    def _compute_raw_signal(
        self,
        data: pd.DataFrame,
        index_ohlc: dict[str, pd.Series] | None = None,
        vix: pd.Series | None = None,
        **kwargs,
    ) -> pd.Series:
        """
        Parameters
        ----------
        data : pd.DataFrame
            Price panel (wide format: dates × symbols).
        index_ohlc : dict, optional
            Keys: 'high', 'low', 'close' — index-level series for
            regime classification.  If None, uses cross-sectional
            median as proxy.
        vix : pd.Series, optional
            India VIX series.
        """
        # Derive index-level OHLC from data if not provided
        if index_ohlc is None:
            idx_close = data.median(axis=1)
            idx_high = data.max(axis=1)
            idx_low = data.min(axis=1)
        else:
            idx_high = index_ohlc["high"]
            idx_low = index_ohlc["low"]
            idx_close = index_ohlc["close"]

        regime_state = self.classifier.classify(idx_high, idx_low, idx_close, vix)
        active_models = self.regime_model_map.get(regime_state.regime, [])

        if not active_models:
            logger.info(f"No models active for regime {regime_state.regime.value}")
            return pd.Series(0.0, index=data.columns, dtype=float)

        # Determine weights
        weights_map = self.blend_weights.get(regime_state.regime, {})
        if not weights_map:
            w = 1.0 / len(active_models)
            weights_map = {m.name: w for m in active_models}

        # Generate and blend signals
        combined = pd.Series(0.0, index=data.columns, dtype=float)
        total_weight = 0.0

        for model in active_models:
            raw = model._compute_raw_signal(data, **kwargs)
            if raw.empty:
                continue
            z = self._zscore(raw)
            weight = weights_map.get(model.name, 1.0 / len(active_models))
            common = combined.index.intersection(z.index)
            combined.loc[common] += weight * z.loc[common]
            total_weight += weight

        if total_weight > 0:
            combined /= total_weight

        # Scale by regime confidence
        combined *= regime_state.confidence

        return combined


class AdaptiveRegimeBlend(AlphaModel):
    """
    Continuously blends alpha models with regime-adaptive weights.

    Instead of binary on/off, each model gets a weight that varies
    smoothly based on regime indicators.  Uses a sigmoid mapping
    from regime features to model weights.

    Parameters
    ----------
    models : list of AlphaModel
        Alpha models to blend.
    regime_features : dict
        Mapping from regime feature name → preferred range tuple
        (center, width) per model.  E.g., {'adx': {model.name: (30, 10)}}.
    """

    def __init__(
        self,
        models: list[AlphaModel],
        model_regime_prefs: dict[str, dict[str, tuple[float, float]]] | None = None,
        norm: SignalNorm = SignalNorm.ZSCORE,
        **kwargs,
    ):
        super().__init__(
            name="adaptive_regime_blend",
            lookback=max(m.lookback for m in models) if models else 252,
            norm=norm,
            direction=SignalDirection.LONG_SHORT,
            **kwargs,
        )
        self.models = models
        self.classifier = RegimeClassifier()
        # model_regime_prefs: feature → { model_name: (center, width) }
        # Default: momentum prefers high ADX, mean-rev prefers low ADX
        self.model_regime_prefs = model_regime_prefs or {}

    @staticmethod
    def _sigmoid_weight(value: float, center: float, width: float) -> float:
        """Gaussian-like weight: peaks at center, decays with width."""
        return float(np.exp(-0.5 * ((value - center) / max(width, 1e-6)) ** 2))

    def _compute_model_weights(
        self,
        adx: float,
        vix: float,
        ma_score: float,
    ) -> dict[str, float]:
        """Compute adaptive weights for each model based on regime features."""
        features = {"adx": adx, "vix": vix, "ma_score": ma_score}
        weights = {}

        for model in self.models:
            w = 1.0
            for feat_name, feat_val in features.items():
                if feat_name in self.model_regime_prefs:
                    prefs = self.model_regime_prefs[feat_name]
                    if model.name in prefs:
                        center, width = prefs[model.name]
                        w *= self._sigmoid_weight(feat_val, center, width)
            weights[model.name] = w

        # Normalize weights to sum to 1
        total = sum(weights.values())
        if total > 1e-10:
            weights = {k: v / total for k, v in weights.items()}
        else:
            eq = 1.0 / len(self.models)
            weights = {m.name: eq for m in self.models}

        return weights

    def _compute_raw_signal(
        self,
        data: pd.DataFrame,
        index_ohlc: dict[str, pd.Series] | None = None,
        vix: pd.Series | None = None,
        **kwargs,
    ) -> pd.Series:
        # Compute regime features
        if index_ohlc is None:
            idx_close = data.median(axis=1)
            idx_high = data.max(axis=1)
            idx_low = data.min(axis=1)
        else:
            idx_high = index_ohlc["high"]
            idx_low = index_ohlc["low"]
            idx_close = index_ohlc["close"]

        adx_series = self.classifier.compute_adx(idx_high, idx_low, idx_close)
        current_adx = float(adx_series.iloc[-1]) if len(adx_series) > 0 else 20.0
        current_vix = float(vix.iloc[-1]) if vix is not None and len(vix) > 0 else 15.0
        ma_score = self.classifier.compute_ma_structure(idx_close)
        current_ma = float(ma_score.iloc[-1]) if len(ma_score) > 0 else 0.0

        weights = self._compute_model_weights(current_adx, current_vix, current_ma)

        combined = pd.Series(0.0, index=data.columns, dtype=float)
        for model in self.models:
            raw = model._compute_raw_signal(data, **kwargs)
            if raw.empty:
                continue
            z = self._zscore(raw)
            w = weights.get(model.name, 0.0)
            common = combined.index.intersection(z.index)
            combined.loc[common] += w * z.loc[common]

        return combined
