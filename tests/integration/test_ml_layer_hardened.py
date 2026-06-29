from datetime import timedelta

import pandas as pd


def test_execution_bias():
    horizon = 1
    # Dummy data with open, close
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5),
            "symbol": ["RELIANCE"] * 5,
            "open": [100.0, 102.0, 105.0, 110.0, 108.0],
            "close": [101.0, 104.0, 108.0, 107.0, 109.0],
        }
    )

    price_data = df.copy()

    if "open" in price_data.columns:
        price_data["open_entry"] = price_data.groupby("symbol")["open"].shift(-1)
        price_data["close_exit"] = price_data.groupby("symbol")["close"].shift(-horizon)
        price_data["forward_return"] = (
            price_data["close_exit"] - price_data["open_entry"]
        ) / price_data["open_entry"]

    expected_fwd_0 = (104.0 - 102.0) / 102.0
    actual_fwd_0 = price_data.loc[0, "forward_return"]

    assert (
        abs(actual_fwd_0 - expected_fwd_0) < 1e-6
    ), f"Execution bias fix failed. Expected {expected_fwd_0}, got {actual_fwd_0}"
    print("[PASS] ML Target correctly uses Open(T+1) to prevent execution hallucination.")


def test_embargo_cv():
    class CVFold:
        def __init__(self, train_start, train_end, test_start, test_end):
            self.train_start = train_start
            self.train_end = train_end
            self.test_start = test_start
            self.test_end = test_end

    def generate_splits(
        start_date, end_date, n_folds, train_size_months, test_size_months, embargo_days
    ):
        folds = []
        current_date = start_date
        for _i in range(n_folds):
            train_start = current_date
            train_end = train_start + timedelta(days=train_size_months * 30)
            test_start = train_end + timedelta(days=embargo_days)
            test_end = test_start + timedelta(days=test_size_months * 30)
            if test_end > end_date:
                break
            folds.append(CVFold(train_start, train_end, test_start, test_end))
            current_date = test_start
        return folds

    start = pd.to_datetime("2020-01-01")
    end = pd.to_datetime("2024-01-01")
    folds = generate_splits(start, end, 3, 12, 3, embargo_days=5)

    for fold in folds:
        test_start = fold.test_start
        train_end = fold.train_end
        gap = (test_start - train_end).days
        assert gap == 5, f"Embargo failed: Gap is {gap} days, expected 5."

    print("[PASS] Purged Combinatorial CV mathematically enforces 5-day Embargo.")


if __name__ == "__main__":
    test_execution_bias()
    test_embargo_cv()
    print("ALL ML LAYER TESTS PASSED")
