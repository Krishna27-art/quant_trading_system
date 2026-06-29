"""
Options Data Validation Rules

Implements specific validation rules for option chains:
- Valid option type (CE/PE)
- Valid strike prices and expiry dates
- Non-negative open interest (OI)
- Required columns (strike, expiry, option_type, underlying_price, etc.)
"""

import pandas as pd

from data_platform.validation.base_validator import (
    BaseValidator,
    ValidationReport,
    ValidationResult,
    ValidationSeverity,
)


class OptionsValidator(BaseValidator):
    """
    Validator for stock and index option contract chains.
    """

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """
        Runs validations on the options chain dataset.
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
        req_cols = [
            "date",
            "symbol",
            "strike",
            "option_type",
            "expiry",
            "underlying_price",
            "close",
            "volume",
            "open_interest",
        ]
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
        report.add_result(
            self._check_not_null(
                df_canon, ["date", "symbol", "strike", "option_type", "expiry", "close"]
            )
        )

        # Strike prices positive
        strikes_positive = (df_canon["strike"] > 0).all()
        bad_strikes = int((df_canon["strike"] <= 0).sum()) if not strikes_positive else 0
        report.add_result(
            ValidationResult(
                rule_name="strike_positive_check",
                passed=strikes_positive,
                severity=ValidationSeverity.CRITICAL,
                message=(
                    "All strike prices are positive"
                    if strikes_positive
                    else f"{bad_strikes} non-positive strikes found"
                ),
            )
        )

        # Option type CE or PE only
        option_types = df_canon["option_type"].astype(str).str.upper()
        type_valid = option_types.isin(["CE", "PE", "CALL", "PUT"]).all()
        invalid_types_count = int((~option_types.isin(["CE", "PE", "CALL", "PUT"])).sum())
        report.add_result(
            ValidationResult(
                rule_name="option_type_check",
                passed=type_valid,
                severity=ValidationSeverity.CRITICAL,
                message=(
                    "All option types are valid (CE/PE)"
                    if type_valid
                    else f"{invalid_types_count} invalid option types found"
                ),
            )
        )

        # Open Interest non-negative
        oi_non_negative = (df_canon["open_interest"] >= 0).all()
        bad_oi = int((df_canon["open_interest"] < 0).sum()) if not oi_non_negative else 0
        report.add_result(
            ValidationResult(
                rule_name="open_interest_non_negative_check",
                passed=oi_non_negative,
                severity=ValidationSeverity.WARNING,
                message=(
                    "All Open Interest values non-negative"
                    if oi_non_negative
                    else f"{bad_oi} negative OI values found"
                ),
            )
        )

        # Price positive check
        report.add_result(self._check_price_positive(df_canon))

        return report
