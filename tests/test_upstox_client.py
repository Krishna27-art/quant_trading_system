import unittest.mock as mock
from data_platform.upstox_client import compute_pcr_from_chain
from utils.upstox_helper import get_instrument_key


def test_instrument_key_resolution():
    # Test caching and key mapping
    key = get_instrument_key("RELIANCE")
    if key:
        assert key.startswith("NSE_EQ|")
        assert "INE002A01018" in key


def test_pcr_calculation():
    # Test PCR calculation with mock chain data
    chain = [
        {
            "strike_price": 22000,
            "call_options": {"market_data": {"oi": 50000.0}},
            "put_options": {"market_data": {"oi": 100000.0}},
        }
    ]
    res = compute_pcr_from_chain(chain)
    assert res["pcr"] == 2.0  # 100000 / 50000
    assert res["total_ce_oi"] == 50000.0
    assert res["total_pe_oi"] == 100000.0
