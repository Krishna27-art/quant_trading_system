"""
Unit tests and latency benchmark for BarAggregator.
Verifies sliding window OHLCV bar construction and sub-35ms cache reads.
"""
import time
from data_platform.feeds.bar_aggregator import BarAggregator
from data_platform.feeds.feed_manager import TickData


def test_bar_aggregator_basic():
    agg = BarAggregator(max_bars=100)
    base_ts = 1700000000.0  # arbitrary epoch aligned to minute

    # Tick 1: opens the bar at price 100
    t1 = TickData(
        symbol="RELIANCE",
        ltp=100.0,
        bid=99.9,
        ask=100.1,
        volume=10,
        timestamp=base_ts,
    )
    completed = agg.on_tick(t1)
    assert len(completed) == 0

    # Tick 2: same minute, higher price
    t2 = TickData(
        symbol="RELIANCE",
        ltp=105.0,
        bid=104.9,
        ask=105.1,
        volume=20,
        timestamp=base_ts + 30,
    )
    agg.on_tick(t2)

    # Check active bar
    df_1m = agg.get_bars_df("RELIANCE", "1m")
    assert len(df_1m) == 1
    assert df_1m["open"].iloc[0] == 100.0
    assert df_1m["high"].iloc[0] == 105.0
    assert df_1m["low"].iloc[0] == 100.0
    assert df_1m["close"].iloc[0] == 105.0
    assert df_1m["volume"].iloc[0] == 30


def test_bar_aggregator_rollover():
    agg = BarAggregator(max_bars=100)
    base_ts = 1700000000.0

    # 10 ticks across 10 consecutive minutes
    for i in range(10):
        t = TickData(
            symbol="TCS",
            ltp=3000.0 + i,
            bid=2999.0,
            ask=3001.0,
            volume=5,
            timestamp=base_ts + (i * 60),
        )
        agg.on_tick(t)

    df_1m = agg.get_bars_df("TCS", "1m")
    assert len(df_1m) == 10
    assert df_1m["close"].iloc[-1] == 3009.0


def test_bar_aggregator_latency_benchmark():
    agg = BarAggregator(max_bars=500)
    base_ts = 1700000000.0

    # Populate 100 bars
    for i in range(100):
        t = TickData(
            symbol="INFY",
            ltp=1500.0 + (i % 5),
            bid=1499.0,
            ask=1501.0,
            volume=100,
            timestamp=base_ts + (i * 60),
        )
        agg.on_tick(t)

    # Benchmark read latency
    start_t = time.perf_counter()
    for _ in range(50):
        df = agg.get_cached_ohlcv("INFY", "1m", min_bars=50)
    end_t = time.perf_counter()

    avg_latency_ms = ((end_t - start_t) / 50.0) * 1000.0
    assert len(df) == 100
    # Must be well under 35ms target (typically < 2ms in memory)
    assert avg_latency_ms < 35.0, f"Average latency too high: {avg_latency_ms:.2f} ms"
