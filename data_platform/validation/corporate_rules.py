"""
Corporate Actions Validation Rules

Implements specific validation rules for corporate action adjustments:
- Split ratio format and positive value check
- Non-negative dividend amount check
- Valid action types (split, dividend, bonus, rights)
"""

import pandas as pd

from data_platform.validation.base_validator import (
    BaseValidator,
    ValidationReport,
    ValidationResult,
    ValidationSeverity,
)


class CorporateValidator(BaseValidator):
    """
    Validator for stock corporate actions records.
    """

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """
        Runs validations on the corporate actions dataset.
        """
        report = ValidationReport(dataset_name=self.dataset_name, total_records=len(df))
        if df.empty:
            report.add_result(
                ValidationResult(
                    rule_name="empty_dataset_check",
                    passed=False,
                    severity=ValidationSeverity.WARNING,
                    message="Dataset is empty",
                )
            )
            return report

        df_canon = self._canonicalize_columns(df)

        # Check required columns
        req_cols = ["date", "symbol", "action_type"]
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

        # Action type check (SPLIT, DIVIDEND, BONUS, RIGHTS)
        action_types = df_canon["action_type"].astype(str).str.upper()
        valid_actions = ["SPLIT", "DIVIDEND", "BONUS", "RIGHTS"]
        action_valid = action_types.isin(valid_actions).all()
        invalid_actions_count = int((~action_types.isin(valid_actions)).sum())

        report.add_result(
            ValidationResult(
                rule_name="action_type_check",
                passed=action_valid,
                severity=ValidationSeverity.CRITICAL,
                message=(
                    "All action types are valid"
                    if action_valid
                    else f"{invalid_actions_count} invalid action types found"
                ),
            )
        )

        # Split check: ratio should be positive if present
        if "ratio" in df_canon.columns:
            split_mask = action_types == "SPLIT"
            if split_mask.any():
                ratios = df_canon.loc[split_mask, "ratio"].dropna()
                ratios_valid = (ratios > 0).all()
                bad_ratios = int((ratios <= 0).sum())
                report.add_result(
                    ValidationResult(
                        rule_name="split_ratio_check",
                        passed=ratios_valid,
                        severity=ValidationSeverity.CRITICAL,
                        message=(
                            "All split ratios positive"
                            if ratios_valid
                            else f"{bad_ratios} non-positive split ratios found"
                        ),
                    )
                )

        # Dividend check: value/amount should be non-negative if present
        value_col = "value" if "value" in df_canon.columns else "amount"
        if value_col in df_canon.columns:
            div_mask = action_types == "DIVIDEND"
            if div_mask.any():
                divs = df_canon.loc[div_mask, value_col].dropna()
                divs_valid = (divs >= 0).all()
                bad_divs = int((divs < 0).sum())
                report.add_result(
                    ValidationResult(
                        rule_name="dividend_value_check",
                        passed=divs_valid,
                        severity=ValidationSeverity.CRITICAL,
                        message=(
                            "All dividend values non-negative"
                            if divs_valid
                            else f"{bad_divs} negative dividend values found"
                        ),
                    )
                )

        # Date consistency check
        if (
            "announcement_date" in df_canon.columns
            and "record_date" in df_canon.columns
            and "ex_date" in df_canon.columns
        ):
            dates_consistent = (
                (df_canon["announcement_date"] <= df_canon["record_date"])
                & (df_canon["record_date"] <= df_canon["ex_date"])
            ).all()
            report.add_result(
                ValidationResult(
                    rule_name="date_consistency_check",
                    passed=dates_consistent,
                    severity=ValidationSeverity.CRITICAL,
                    message=(
                        "Date consistency checked (announcement <= record <= ex)"
                        if dates_consistent
                        else "Date consistency violated"
                    ),
                )
            )

        return report
