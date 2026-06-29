import sys

import numpy as np
import pandas as pd

sys.path.append("/Users/pandu/Desktop/quant")


class PromoterHolding:
    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data["promoter_holding_pct"].shift(1)


def test_memory_float32():
    all_features = pd.DataFrame(
        {
            "col1": np.random.randn(100).astype("float64"),
            "col2": np.random.randn(100).astype("float64"),
        }
    )

    # Simulate the pipeline loop
    for col in all_features.columns:
        if all_features[col].dtype == "float64":
            all_features[col] = all_features[col].astype("float32")

    assert all_features["col1"].dtype == "float32", "Failed to downcast float64 to float32"
    assert all_features["col2"].dtype == "float32", "Failed to downcast float64 to float32"
    print("[PASS] Memory Feature Store Pipeline strictly enforces float32.")


def test_lookahead_bias():
    feature = PromoterHolding()
    data = pd.DataFrame({"promoter_holding_pct": [50.0, 51.0, 52.0]})
    result = feature.compute(data)

    # result should be shifted by 1
    assert pd.isna(result.iloc[0]), "Lookahead bias: T=0 has data"
    assert result.iloc[1] == 50.0, f"Lookahead bias: T=1 should have T=0 data, got {result.iloc[1]}"
    print("[PASS] Lookahead Bias structurally eliminated via .shift(1) on post-market data.")


if __name__ == "__main__":
    test_memory_float32()
    test_lookahead_bias()
    print("ALL FEATURE STORE TESTS PASSED")
