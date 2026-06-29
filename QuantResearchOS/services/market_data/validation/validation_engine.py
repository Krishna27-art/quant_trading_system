from typing import Any

import pandas as pd


class ValidationEngine:
    def __init__(self):
        pass

    def validate_ohlcv(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Validate OHLCV data for:
        - Missing candles
        - Negative prices
        - High < Low
        - Zero volume
        - Duplicate timestamps
        - NaN values (percentage-based penalty)
        """
        if df.empty:
            return {"status": "error", "reason": "Empty DataFrame", "score": 0.0}

        score = 100.0
        issues = []
        total_cells = df.shape[0] * df.shape[1]

        # Check negative prices
        for col in ["open", "high", "low", "close"]:
            if (df[col] < 0).any():
                score -= 20
                issues.append(f"Negative prices found in {col}")

        # Check high < low
        if (df["high"] < df["low"]).any():
            score -= 30
            issues.append("High price is less than Low price")

        # Check zero volume
        zero_volume_pct = (df["volume"] == 0).mean()
        if zero_volume_pct > 0.1:
            score -= zero_volume_pct * 100
            issues.append(f"High percentage of zero volume: {zero_volume_pct:.2%}")

        # Check duplicate timestamps
        if "timestamp" in df.columns:
            dup_count = df["timestamp"].duplicated().sum()
            if dup_count > 0:
                dup_pct = dup_count / len(df)
                score -= dup_pct * 100
                issues.append(f"Found {dup_count} duplicate timestamps ({dup_pct:.2%})")

        # Check missing/NaN values (percentage-based penalty)
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            nan_pct = nan_count / total_cells
            score -= min(50, nan_pct * 100)
            issues.append(f"Found {nan_count} missing values ({nan_pct:.2%} of data)")

        return {
            "status": "passed" if score > 80 else "quarantine",
            "score": max(0.0, score),
            "issues": issues,
        }
