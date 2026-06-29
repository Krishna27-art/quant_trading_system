"""
Data Quality Tests

Tests that verify data quality, validation rules, and integrity.
"""

import numpy as np
import pandas as pd
import pytest

from data_platform.validation.corporate_rules import CorporateValidator
from data_platform.validation.equity_rules import EquityValidator
from data_platform.validation.options_rules import OptionsValidator


class TestEquityDataQuality:
    """Test equity data quality rules."""

    @pytest.fixture
    def validator(self):
        """Create equity validator."""
        return EquityValidator()

    @pytest.fixture
    def valid_equity_data(self):
        """Create valid equity data."""
        return pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=100),
                "Symbol": ["RELIANCE"] * 100,
                "Open": np.random.uniform(90, 110, 100),
                "High": np.random.uniform(110, 130, 100),
                "Low": np.random.uniform(70, 90, 100),
                "Close": np.random.uniform(90, 110, 100),
                "Volume": np.random.uniform(1000000, 10000000, 100),
            }
        )

    def test_required_columns_present(self, validator, valid_equity_data):
        """Test all required columns are present."""
        result = validator.validate(valid_equity_data)
        assert result.is_acceptable()

    def test_negative_prices_rejected(self, validator):
        """Test negative prices are rejected."""
        df = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=10),
                "Symbol": ["RELIANCE"] * 10,
                "Open": [-100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "High": [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
                "Low": [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
                "Close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "Volume": [1000000] * 10,
            }
        )

        result = validator.validate(df)
        assert not result.is_acceptable()
        assert any(
            "positive" in error.lower() or "negative" in error.lower()
            for error in [r.message for r in result.results if not r.passed]
        )

    def test_negative_volume_rejected(self, validator):
        """Test negative volume is rejected."""
        df = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=10),
                "Symbol": ["RELIANCE"] * 10,
                "Open": [100] * 10,
                "High": [105] * 10,
                "Low": [95] * 10,
                "Close": [100] * 10,
                "Volume": [
                    -1000000,
                    1100000,
                    1200000,
                    1300000,
                    1400000,
                    1500000,
                    1600000,
                    1700000,
                    1800000,
                    1900000,
                ],
            }
        )

        result = validator.validate(df)
        assert result.failed_count > 0
        assert any(
            "volume" in error.lower()
            for error in [r.message for r in result.results if not r.passed]
        )

    def test_high_low_consistency(self, validator):
        """Test High >= Low constraint."""
        df = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=10),
                "Symbol": ["RELIANCE"] * 10,
                "Open": [100] * 10,
                "High": [95, 106, 107, 108, 109, 110, 111, 112, 113, 114],  # First High < Low
                "Low": [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
                "Close": [100] * 10,
                "Volume": [1000000] * 10,
            }
        )

        result = validator.validate(df)
        assert not result.is_acceptable()

    def test_close_within_high_low(self, validator):
        """Test Close is within High-Low range."""
        df = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=10),
                "Symbol": ["RELIANCE"] * 10,
                "Open": [100] * 10,
                "High": [105] * 10,
                "Low": [95] * 10,
                "Close": [110, 101, 102, 103, 104, 105, 106, 107, 108, 109],  # First Close > High
                "Volume": [1000000] * 10,
            }
        )

        result = validator.validate(df)
        assert not result.is_acceptable()

    def test_no_duplicate_dates(self, validator):
        """Test no duplicate dates."""
        dates = list(pd.date_range("2024-01-01", periods=10))
        dates[5] = dates[4]  # Duplicate date

        df = pd.DataFrame(
            {
                "Date": dates,
                "Symbol": ["RELIANCE"] * 10,
                "Open": [100] * 10,
                "High": [105] * 10,
                "Low": [95] * 10,
                "Close": [100] * 10,
                "Volume": [1000000] * 10,
            }
        )

        result = validator.validate(df)
        assert result.failed_count > 0

    def test_date_chronological_order(self, validator):
        """Test dates are in chronological order."""
        dates = list(pd.date_range("2024-01-01", periods=10))
        dates[5], dates[6] = dates[6], dates[5]  # Swap dates

        df = pd.DataFrame(
            {
                "Date": dates,
                "Symbol": ["RELIANCE"] * 10,
                "Open": [100] * 10,
                "High": [105] * 10,
                "Low": [95] * 10,
                "Close": [100] * 10,
                "Volume": [1000000] * 10,
            }
        )

        result = validator.validate(df)
        assert result.failed_count > 0

    def test_missing_values_rejected(self, validator):
        """Test missing values are rejected."""
        df = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=10),
                "Symbol": ["RELIANCE"] * 10,
                "Open": [100] * 10,
                "High": [105] * 10,
                "Low": [95] * 10,
                "Close": [
                    100,
                    np.nan,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                ],  # Close is checked for null
                "Volume": [1000000] * 10,
            }
        )

        result = validator.validate(df)
        assert not result.is_acceptable()
        assert any(
            "missing" in error.lower() or "null" in error.lower()
            for error in [r.message for r in result.results if not r.passed]
        )

    def test_volume_not_zero(self, validator):
        """Test volume is not zero."""
        df = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=10),
                "Symbol": ["RELIANCE"] * 10,
                "Open": [100] * 10,
                "High": [105] * 10,
                "Low": [95] * 10,
                "Close": [100] * 10,
                "Volume": [
                    0,
                    1100000,
                    1200000,
                    1300000,
                    1400000,
                    1500000,
                    1600000,
                    1700000,
                    1800000,
                    1900000,
                ],
            }
        )

        result = validator.validate(df)
        assert result.failed_count > 0


class TestOptionsDataQuality:
    """Test options data quality rules."""

    @pytest.fixture
    def validator(self):
        """Create options validator."""
        return OptionsValidator("options")

    @pytest.fixture
    def valid_options_data(self):
        """Create valid options data."""
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=100),
                "symbol": ["RELIANCE"] * 100,
                "strike": np.random.uniform(2000, 3000, 100),
                "expiry": pd.date_range("2024-02-01", periods=100),
                "option_type": ["CE", "PE"] * 50,
                "open": np.random.uniform(10, 100, 100),
                "high": np.random.uniform(100, 200, 100),
                "low": np.random.uniform(5, 10, 100),
                "close": np.random.uniform(10, 100, 100),
                "volume": np.random.uniform(1000, 10000, 100),
                "open_interest": np.random.uniform(10000, 100000, 100),
                "underlying_price": np.random.uniform(2000, 3000, 100),
            }
        )

    def test_required_columns_present(self, validator, valid_options_data):
        """Test all required columns are present."""
        result = validator.validate(valid_options_data)
        assert result.is_acceptable()

    def test_negative_prices_rejected(self, validator):
        """Test negative prices are rejected."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=10),
                "symbol": ["RELIANCE"] * 10,
                "strike": [2500] * 10,
                "expiry": pd.date_range("2024-02-01", periods=10),
                "option_type": ["CE"] * 10,
                "open": [-10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                "high": [20] * 10,
                "low": [5] * 10,
                "close": [10] * 10,
                "volume": [1000] * 10,
                "open_interest": [10000] * 10,
                "underlying_price": [2500.0] * 10,
            }
        )

        result = validator.validate(df)
        assert not result.is_acceptable()

    def test_valid_option_types(self, validator):
        """Test only CE and PE option types are accepted."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=10),
                "symbol": ["RELIANCE"] * 10,
                "strike": [2500] * 10,
                "expiry": pd.date_range("2024-02-01", periods=10),
                "option_type": [
                    "CE",
                    "PE",
                    "XX",
                    "CE",
                    "PE",
                    "CE",
                    "PE",
                    "CE",
                    "PE",
                    "CE",
                ],  # Invalid 'XX'
                "open": [10] * 10,
                "high": [20] * 10,
                "low": [5] * 10,
                "close": [10] * 10,
                "volume": [1000] * 10,
                "open_interest": [10000] * 10,
                "underlying_price": [2500.0] * 10,
            }
        )

        result = validator.validate(df)
        assert not result.is_acceptable()


class TestCorporateActionsDataQuality:
    """Test corporate actions data quality rules."""

    @pytest.fixture
    def validator(self):
        """Create corporate actions validator."""
        return CorporateValidator("corporate")

    @pytest.fixture
    def valid_corporate_actions_data(self):
        """Create valid corporate actions data."""
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-20", periods=30),
                "symbol": ["RELIANCE", "TCS", "INFY"] * 10,
                "action_type": ["DIVIDEND", "SPLIT", "BONUS"] * 10,
                "announcement_date": pd.date_range("2024-01-01", periods=30),
                "record_date": pd.date_range("2024-01-15", periods=30),
                "ex_date": pd.date_range("2024-01-20", periods=30),
                "ratio": [1.0, 2.0, 1.5] * 10,
                "amount": [10.0, 0.0, 0.0] * 10,
            }
        )

    def test_required_columns_present(self, validator, valid_corporate_actions_data):
        """Test all required columns are present."""
        result = validator.validate(valid_corporate_actions_data)
        assert result.is_acceptable()

    def test_valid_event_types(self, validator):
        """Test only valid event types are accepted."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-20", periods=10),
                "symbol": ["RELIANCE"] * 10,
                "action_type": [
                    "DIVIDEND",
                    "SPLIT",
                    "INVALID",
                    "BONUS",
                    "RIGHTS",
                    "BUYBACK",
                    "MERGER",
                    "DEMERGER",
                    "SUSPENSION",
                    "DELISTING",
                ],
                "announcement_date": pd.date_range("2024-01-01", periods=10),
                "record_date": pd.date_range("2024-01-15", periods=10),
                "ex_date": pd.date_range("2024-01-20", periods=10),
                "ratio": [1.0] * 10,
                "amount": [10.0] * 10,
            }
        )

        result = validator.validate(df)
        assert not result.is_acceptable()

    def test_date_consistency(self, validator):
        """Test announcement_date <= record_date <= ex_date."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-20", periods=10),
                "symbol": ["RELIANCE"] * 10,
                "action_type": ["DIVIDEND"] * 10,
                "announcement_date": pd.date_range("2024-01-01", periods=10),
                "record_date": pd.date_range("2024-01-15", periods=10),
                "ex_date": pd.date_range("2024-01-10", periods=10),  # ex_date before record_date
                "ratio": [1.0] * 10,
                "amount": [10.0] * 10,
            }
        )

        result = validator.validate(df)
        assert not result.is_acceptable()
