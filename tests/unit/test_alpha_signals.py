"""
Unit Tests for Alpha Signals Engine
"""

import numpy as np
import pandas as pd
import pytest

from portfolio_execution.signals.composite import CompositeAlphaModel
from portfolio_execution.signals.mean_reversion import ResidualMeanReversion
from portfolio_execution.signals.momentum import CrossSectionalMomentum, TimeSeriesMomentum


@pytest.fixture
def sample_price_data():
    """Generates synthetic price data for a small universe of stocks."""
    dates = pd.date_range(start="2024-01-01", periods=300, freq="B")
    symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]

    # Generate random walk prices
    np.random.seed(42)
    prices = {}
    for i, sym in enumerate(symbols):
        # Different drifts/vols to differentiate momentum/reversion
        drift = 0.0002 * (i - 2)
        vol = 0.01 + 0.002 * i
        returns = np.random.normal(drift, vol, len(dates))
        prices[sym] = 100.0 * np.exp(np.cumsum(returns))

    return pd.DataFrame(prices, index=dates)


def test_cross_sectional_momentum(sample_price_data):
    # lookback=60, skip=5
    model = CrossSectionalMomentum(lookback=60, skip=5)
    sig_obj = model.generate(sample_price_data)

    assert sig_obj is not None
    assert isinstance(sig_obj.timestamp, pd.Timestamp)
    assert len(sig_obj.signal) == 5
    assert not sig_obj.signal.isnull().any()

    # Z-scored signals should sum to 0 (neutralized)
    assert np.allclose(sig_obj.signal.sum(), 0.0, atol=1e-7)


def test_time_series_momentum(sample_price_data):
    model = TimeSeriesMomentum(lookback=60)
    sig_obj = model.generate(sample_price_data)

    assert sig_obj is not None
    assert len(sig_obj.signal) == 5
    # Volatility scaled signals should not all be identical
    assert sig_obj.signal.std() > 0.0


def test_stat_arb_reversal(sample_price_data):
    model = ResidualMeanReversion(lookback=60, zscore_lookback=20)
    sig_obj = model.generate(sample_price_data)

    assert sig_obj is not None
    assert len(sig_obj.signal) == 5
    assert np.allclose(sig_obj.signal.sum(), 0.0, atol=1e-7)


def test_composite_alpha_blending(sample_price_data):
    model1 = CrossSectionalMomentum(lookback=60, skip=5)
    model2 = ResidualMeanReversion(lookback=60, zscore_lookback=20)

    composite = CompositeAlphaModel(models=[model1, model2], dynamic_weights=False)
    sig_obj = composite.generate(sample_price_data)

    assert sig_obj is not None
    assert len(sig_obj.signal) == 5
    # Blended signal should be normalized
    assert np.allclose(sig_obj.signal.sum(), 0.0, atol=1e-7)


def test_india_macro_alpha(sample_price_data):
    from portfolio_execution.signals.india_macro_alpha import IndiaMacroAlpha

    model = IndiaMacroAlpha()
    sig_obj = model.generate(sample_price_data)

    assert sig_obj is not None
    assert len(sig_obj.signal) == 5
