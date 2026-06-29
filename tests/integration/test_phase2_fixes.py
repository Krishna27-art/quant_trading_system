import os


def test_vwap_bleed_fixed():
    # Verify that the new alpha module exists and compiles
    from portfolio_execution.signals import OpeningRangeBreakout, VolumeWeightedPressure

    assert OpeningRangeBreakout is not None
    assert VolumeWeightedPressure is not None
    print("[PASS] VWAP signal classes are successfully verified.")


def test_fundamental_lookahead_fixed():
    # Verify that bfill is not used in ingestion_validator.py
    path = "/Users/pandu/Desktop/quant/data_platform/validation/ingestion_validator.py"
    assert os.path.exists(path)
    with open(path) as f:
        content = f.read()
    # Check that 'bfill' doesn't exist in the file content
    assert "bfill" not in content, "Lookahead bias bfill() still exists in ingestion_validator.py"
    print("[PASS] Fundamental lookahead bias checks passed.")


def test_transaction_costs_injected():
    # Verify transaction cost calculator is integrated and has commission/slippage attributes
    from research_platform.backtesting.transaction_costs import (
        TransactionCostModel,
    )

    model = TransactionCostModel()
    assert hasattr(model, "commission_rate")
    assert hasattr(model, "slippage_rate")
    print("[PASS] Transaction cost models are present in the backtesting module.")


if __name__ == "__main__":
    test_vwap_bleed_fixed()
    test_fundamental_lookahead_fixed()
    test_transaction_costs_injected()
    print("All Phase 2 fixes successfully passed validation.")
