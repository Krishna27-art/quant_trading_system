"""
Equity Data Validation Rules

Implements specific validation rules for daily/intraday stock price bars:
- OHLC price ordering checks (high >= open/close >= low)
- Non-negative volume and turnover
- Missing values on key columns
"""

import pandas as pd

from data_platform.validation.base_validator import (
    BaseValidator,
    ValidationReport,
    ValidationResult,
    ValidationSeverity,
)


class EquityValidator(BaseValidator):
    """
    Validator for stock equity price history.
    """

    def __init__(self, dataset_name: str = "equity") -> None:
        super().__init__(dataset_name)

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """
        Runs validations on the equity price dataset.
        """
        report = ValidationReport(dataset_name=self.dataset_name, total_records=len(df))
        if df.empty:
            report.add_result(
                ValidationResult(
                    rule_name="empty_dataset_check",
                    passed=False,
                    severity=ValidationSeverity.CRITICAL,
                    message="Dataset is empty",
                )
            )
            return report

        df_canon = self._canonicalize_columns(df)

        # Check required columns
        req_cols = ["date", "symbol", "open", "high", "low", "close", "volume"]
        missing_cols = [c for c in req_cols if c not in df_canon.columns]
        report.add_result(
            ValidationResult(
                rule_name="required_columns_check",
                passed=len(missing_cols) == 0,
                severity=ValidationSeverity.CRITICAL,
                message=(
                    "All required columns present"
                    if not missing_cols
                    else f"Missing required columns: {missing_cols}"
                ),
            )
        )

        if missing_cols:
            return report

        # Not null checks
        report.add_result(self._check_not_null(df_canon, ["date", "symbol", "close"]))

        # Duplicate checks
        report.add_result(self._check_no_duplicates(df_canon, ["date", "symbol"]))

        # Date order check
        report.add_result(self._check_date_order(df_canon, "date"))

        # Price positive checks
        report.add_result(self._check_price_positive(df_canon))

        # Volume positive checks
        report.add_result(self._check_volume_positive(df_canon))

        # Price ordering (High >= Open/Close/Low, Low <= Open/Close/High)
        high_low_passed = (df_canon["high"] >= df_canon["low"]).all()
        high_open_passed = (df_canon["high"] >= df_canon["open"]).all()
        high_close_passed = (df_canon["high"] >= df_canon["close"]).all()
        low_open_passed = (df_canon["low"] <= df_canon["open"]).all()
        low_close_passed = (df_canon["low"] <= df_canon["close"]).all()

        price_ordering_passed = (
            high_low_passed
            and high_open_passed
            and high_close_passed
            and low_open_passed
            and low_close_passed
        )

        bad_records = 0
        if not price_ordering_passed:
            bad_mask = (
                (df_canon["high"] < df_canon["low"])
                | (df_canon["high"] < df_canon["open"])
                | (df_canon["high"] < df_canon["close"])
                | (df_canon["low"] > df_canon["open"])
                | (df_canon["low"] > df_canon["close"])
            )
            bad_records = int(bad_mask.sum())

        report.add_result(
            ValidationResult(
                rule_name="price_ordering_check",
                passed=price_ordering_passed,
                severity=ValidationSeverity.CRITICAL,
                message=(
                    "Prices are logically ordered (High >= Close/Open >= Low)"
                    if price_ordering_passed
                    else f"OHLC ordering violated in {bad_records} records"
                ),
                details={"bad_records_count": bad_records},
            )
        )

        return report
