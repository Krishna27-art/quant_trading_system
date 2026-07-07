"""
Unit tests for CanonicalFeatureBuilder and Historical Macro extraction.
Verifies deterministic feature calculation and fail-closed validation.
"""
import numpy as np
import pandas as pd
from data_platform.features.canonical_builder import (
    CanonicalFeatureBuilder,
    INTRADAY_FEATURES,
    SWING_FEATURES,
)
from data_platform.feature_store.macro import extract_historical_macro


def test_canonical_builder_intraday():
    dates = pd.date_range("2026-01-01 09:15", periods=30, freq="1min")
    df = pd.DataFrame({
        "open": np.linspace(100, 105, 30),
        "high": np.linspace(101, 106, 30),
        "low": np.linspace(99, 104, 30),
        "close": np.linspace(100.5, 105.5, 30),
        "volume": np.full(30, 1000),
    }, index=dates)

    feats = CanonicalFeatureBuilder.build_features(df, "INTRADAY", extra={"vix": 14.5})
    assert all(col in feats.columns for col in INTRADAY_FEATURES)
    assert feats["vix"].iloc[-1] == 14.5
    assert CanonicalFeatureBuilder.validate_schema(feats, "INTRADAY") is True


def test_canonical_builder_swing_with_series_macro():
    dates = pd.date_range("2026-01-01", periods=50, freq="D")
    df = pd.DataFrame({
        "open": np.linspace(1000, 1100, 50),
        "high": np.linspace(1010, 1110, 50),
        "low": np.linspace(990, 1090, 50),
        "close": np.linspace(1005, 1105, 50),
        "volume": np.full(50, 50000),
    }, index=dates)

    vix_series = pd.Series(np.linspace(12.0, 22.0, 50), index=dates)
    feats = CanonicalFeatureBuilder.build_features(df, "SWING", extra={"vix": vix_series, "nifty_pcr": 1.1})
    
    assert all(col in feats.columns for col in SWING_FEATURES)
    assert np.isclose(feats["vix"].iloc[-1], 22.0)
    assert feats["nifty_pcr"].iloc[0] == 1.1
    assert CanonicalFeatureBuilder.validate_schema(feats, "SWING") is True


def test_canonical_builder_fail_closed_validation():
    # Empty dataframe
    df_empty = pd.DataFrame()
    assert CanonicalFeatureBuilder.validate_schema(df_empty, "INTRADAY") is False

    # Missing column
    df = pd.DataFrame({"close": [10, 11], "volume": [100, 100]})
    feats = pd.DataFrame({"rsi_14m": [50, 55], "vix": [15, 15]})  # missing vwap_dist, etc.
    assert CanonicalFeatureBuilder.validate_schema(feats, "INTRADAY") is False


def test_extract_historical_macro_fallback():
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    # With offline/fallback behavior, should return dataframe with vix=15.0 and nifty_pcr=1.0
    macro_df = extract_historical_macro(dates)
    assert len(macro_df) == 10
    assert "vix" in macro_df.columns
    assert "nifty_pcr" in macro_df.columns
    assert not macro_df["vix"].isna().any()


def test_model_registry_imputer_persistence(tmp_path):
    from sklearn.impute import SimpleImputer
    from prediction_intelligence.base_logistic import ModelRegistry

    reg = ModelRegistry(model_dir=str(tmp_path))
    reg.purge()
    reg = ModelRegistry(model_dir=str(tmp_path))

    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [10.0, np.nan, 30.0, 40.0]})
    imputer.fit(X)

    reg.save_imputer("INTRADAY", imputer)
    loaded = reg.get_imputer("INTRADAY")
    assert loaded is not None
    res = loaded.transform([[np.nan, np.nan]])
    assert np.isclose(res[0, 0], 2.0)
    assert np.isclose(res[0, 1], 30.0)
    reg.purge()
