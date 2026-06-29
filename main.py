"""
Central Trading Orchestrator Main Entry Point

Wires the entire platform:
    Data Feed → Bar Aggregator → State Manager → Alpha Signals → Risk Filter → OMS → EMS

Supports three execution modes:
    - BACKTEST: Simulates historical execution using 1m bars.
    - PAPER: Simulates live execution using real-time simulated ticks.
    - LIVE: Connects to live broker feed and execution adapters.
"""

import argparse
import asyncio
import contextlib
import datetime
import sys
import threading
import time
from typing import Any

from data_platform.feeds.feed_manager import FeedManager, FeedTier
from data_platform.feeds.feed_manager import TickData as FeedTick
from portfolio_execution.config import ExecutionMode, TradingConfig
from portfolio_execution.orchestrator import TickData, TradingOrchestrator
from portfolio_execution.signals.alternative_data import NewsSentimentAlpha, OptionFlowAlpha
from portfolio_execution.signals.composite import CompositeAlphaModel
from portfolio_execution.signals.cross_asset_signals import IndexFuturesBasisAlpha
from portfolio_execution.signals.fundamental_pit import FundamentalPITAlpha
from portfolio_execution.signals.mean_reversion import BollingerMeanReversion
from portfolio_execution.signals.momentum import TimeSeriesMomentum
from portfolio_execution.signals.volatility_surface import VolatilitySurfaceAlpha
from utils.logger import get_logger

logger = get_logger("main_entry")


class BarAggregator:
    """Aggregates tick-by-tick feed data into 1-minute bars for the Orchestrator."""

    def __init__(self, on_bar_completed: callable):
        self.on_bar_completed = on_bar_completed
        self.current_bars: dict[str, dict[str, Any]] = {}
        self.last_sequence: dict[str, int] = {}
        self.lock = threading.Lock()

    def process_tick(self, tick: FeedTick) -> None:
        """
        Process a single tick and emit a completed 1m bar when the minute rolls over.
        """
        with self.lock:
            symbol = tick.symbol
            tick_time = datetime.datetime.fromtimestamp(tick.timestamp)
            tick_minute = tick_time.replace(second=0, microsecond=0)
            price = tick.ltp
            volume = tick.volume

            current = self.current_bars.get(symbol)

            if tick.sequence_num > 0:
                if tick.sequence_num <= self.last_sequence.get(symbol, -1):
                    logger.debug(
                        f"Dropping out-of-order tick for {symbol}: seq {tick.sequence_num}"
                    )
                    return
                self.last_sequence[symbol] = tick.sequence_num

            if current is None:
                # Initialize new bar
                self.current_bars[symbol] = {
                    "symbol": symbol,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": volume,
                    "minute": tick_minute,
                }
            elif current["minute"] > tick_minute:
                # Drop late tick from previous minute
                logger.debug(
                    f"Dropping late tick for {symbol}: {tick_minute} < {current['minute']}"
                )
                return
            elif current["minute"] < tick_minute:
                # The minute rolled over. Complete and emit the old bar
                old_bar = current
                completed_tick = TickData(
                    symbol=old_bar["symbol"],
                    timestamp=old_bar["minute"],
                    open=old_bar["open"],
                    high=old_bar["high"],
                    low=old_bar["low"],
                    close=old_bar["close"],
                    volume=float(old_bar["volume"]),
                )

                # Async callback dispatch or direct execution
                try:
                    self.on_bar_completed(completed_tick)
                except Exception as e:
                    logger.error(f"Error executing bar callback: {e}", exc_info=True)

                # Start the new bar
                self.current_bars[symbol] = {
                    "symbol": symbol,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": volume,
                    "minute": tick_minute,
                }
            else:
                # Update the active bar
                current["high"] = max(current["high"], price)
                current["low"] = min(current["low"], price)
                current["close"] = price
                current["volume"] += volume


def generate_mock_historical_bars(symbol: str, date_str: str) -> list[TickData]:
    """Generate mock 1-minute bars for backtesting."""
    base_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    start_time = base_date.replace(hour=9, minute=15, second=0)

    bars = []
    current_price = 2400.0 if symbol == "RELIANCE" else (3500.0 if symbol == "TCS" else 1500.0)
    import numpy as np

    np.random.seed(42)

    # 375 minutes in a standard NSE trading session (9:15 AM to 3:30 PM)
    for i in range(375):
        timestamp = start_time + datetime.timedelta(minutes=i)
        change_pct = np.random.normal(0.0001, 0.001)
        open_price = current_price
        close_price = open_price * (1 + change_pct)
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0.0005, 0.0005)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0.0005, 0.0005)))
        volume = float(np.random.randint(1000, 10000))

        bars.append(
            TickData(
                symbol=symbol,
                timestamp=timestamp,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
            )
        )
        current_price = close_price

    return bars


def feed_process_worker(symbols: list[str], duration_s: int, tick_event):
    import numpy as np

    from data_platform.ring_buffer import SPSCTickRingBuffer

    logger.info("[FEED] Starting feed process...")
    feed_strategy_buf = SPSCTickRingBuffer("feed_to_strategy", create=False)

    def on_feed_tick(feed_tick: FeedTick):
        # Spin wait if buffer full (simple backpressure)
        while not feed_strategy_buf.write_tick(
            feed_tick.symbol, feed_tick.ltp, feed_tick.volume, feed_tick.timestamp
        ):
            pass  # Busy wait on full buffer instead of sleep
        tick_event.set()

    async def mock_ws_connect(symbols: list[str], on_tick: callable):
        prices = {"RELIANCE": 2400.0, "TCS": 3500.0, "INFY": 1500.0}
        start_time = time.time()
        while time.time() - start_time < duration_s:
            for sym in symbols:
                base_price = prices[sym]
                noise = np.random.normal(0, 0.5)
                prices[sym] = base_price + noise

                tick = FeedTick(
                    symbol=sym,
                    ltp=prices[sym],
                    bid=prices[sym] - 0.2,
                    ask=prices[sym] + 0.2,
                    volume=np.random.randint(10, 100),
                    timestamp=time.time(),
                    received_at=time.time(),
                    feed_tier=FeedTier.PRIMARY,
                )
                on_tick(tick)
            await asyncio.sleep(1.0)

    feed_manager = FeedManager(ws_connect_fn=mock_ws_connect, on_tick=on_feed_tick)
    feed_manager.subscribe(symbols)

    # Run the feed manager for duration
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        feed_task = loop.create_task(feed_manager.start())
        loop.run_until_complete(asyncio.sleep(duration_s))
        feed_manager.stop()
        feed_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            loop.run_until_complete(feed_task)
    finally:
        loop.close()
        feed_strategy_buf.close()
        logger.info("[FEED] Process exited.")


def strategy_process_worker(
    config: TradingConfig, prev_day_data: dict, duration_s: int, tick_event, order_event
):
    from data_platform.ring_buffer import SPSCOrderRingBuffer, SPSCTickRingBuffer

    logger.info("[STRATEGY] Starting strategy process...")
    feed_strategy_buf = SPSCTickRingBuffer("feed_to_strategy", create=False)
    strategy_exec_buf = SPSCOrderRingBuffer("strategy_to_exec", create=False)

    orchestrator = TradingOrchestrator(config)
    momentum_model = TimeSeriesMomentum()
    mr_model = BollingerMeanReversion()
    news_alpha = NewsSentimentAlpha()
    options_alpha = OptionFlowAlpha()
    fundamental_alpha = FundamentalPITAlpha()
    basis_alpha = IndexFuturesBasisAlpha()
    vol_alpha = VolatilitySurfaceAlpha()

    composite_alpha = CompositeAlphaModel(
        models=[
            momentum_model,
            mr_model,
            news_alpha,
            options_alpha,
            fundamental_alpha,
            basis_alpha,
            vol_alpha,
        ],
        dynamic_weights=False,
    )
    composite_alpha.static_weights = {
        momentum_model.name: 0.20,
        mr_model.name: 0.15,
        news_alpha.name: 0.10,
        options_alpha.name: 0.15,
        fundamental_alpha.name: 0.15,
        basis_alpha.name: 0.10,
        vol_alpha.name: 0.15,
    }
    orchestrator.register_alpha(composite_alpha)
    orchestrator.initialize_session(prev_day_data)

    def handle_completed_bar(tick_bar: TickData):
        orders = orchestrator.on_tick(tick_bar)
        if orders:
            for order in orders:
                while not strategy_exec_buf.write_order(
                    order.order_id, order.symbol, order.quantity
                ):
                    pass  # Busy wait on full buffer
                order_event.set()

    aggregator = BarAggregator(on_bar_completed=handle_completed_bar)

    start_time = time.time()
    while time.time() - start_time < duration_s:
        tick_data = feed_strategy_buf.read_tick()
        if tick_data:
            symbol, ltp, volume, timestamp = tick_data
            feed_tick = FeedTick(
                symbol=symbol,
                ltp=ltp,
                bid=ltp,
                ask=ltp,
                volume=volume,
                timestamp=timestamp,
                received_at=timestamp,
                feed_tier=FeedTier.PRIMARY,
            )
            aggregator.process_tick(feed_tick)
        else:
            tick_event.wait(timeout=1.0)
            tick_event.clear()

    orchestrator.stop()
    feed_strategy_buf.close()
    strategy_exec_buf.close()
    logger.info("[STRATEGY] Process exited.")


def execution_process_worker(duration_s: int, order_event):
    from data_platform.ring_buffer import SPSCOrderRingBuffer

    logger.info("[EXEC] Starting execution process...")
    strategy_exec_buf = SPSCOrderRingBuffer("strategy_to_exec", create=False)

    start_time = time.time()
    while time.time() - start_time < duration_s:
        order_data = strategy_exec_buf.read_order()
        if order_data:
            order_id, symbol, quantity = order_data
            logger.info(f"🚀 [EXEC] Executing order {order_id} for {symbol} qty={quantity}")
        else:
            order_event.wait(timeout=1.0)
            order_event.clear()

    strategy_exec_buf.close()
    logger.info("[EXEC] Process exited.")


def run_multi_process_topology(
    config: TradingConfig, prev_day_data: dict, symbols: list[str], duration_s: int
):
    import multiprocessing

    from data_platform.ring_buffer import SPSCOrderRingBuffer, SPSCTickRingBuffer

    logger.info("Initializing Zero-Copy Shared Ring Buffers...")
    buf1 = SPSCTickRingBuffer("feed_to_strategy", create=True)
    buf2 = SPSCOrderRingBuffer("strategy_to_exec", create=True)

    tick_event = multiprocessing.Event()
    order_event = multiprocessing.Event()

    feed_p = multiprocessing.Process(
        target=feed_process_worker, args=(symbols, duration_s, tick_event)
    )
    strat_p = multiprocessing.Process(
        target=strategy_process_worker,
        args=(config, prev_day_data, duration_s, tick_event, order_event),
    )
    exec_p = multiprocessing.Process(
        target=execution_process_worker, args=(duration_s, order_event)
    )

    exec_p.start()
    strat_p.start()
    feed_p.start()

    feed_p.join()
    strat_p.join()
    exec_p.join()

    buf1.unlink()
    buf2.unlink()
    logger.info("Multi-process topology shutdown successfully.")


def main():
    # Setup global exception hook to trigger the emergency kill switch
    def global_exception_hook(exctype, value, traceback):
        logger.critical(
            "🚨 Unhandled global exception caught! Triggering emergency kill switch.",
            exc_info=(exctype, value, traceback),
        )
        from risk_governance.pre_trade.kill_switch import execute_kill_switch

        try:
            execute_kill_switch(dry_run=False)
        except Exception as e:
            logger.error(f"Failed to execute kill switch from excepthook: {e}")
        # Call the original excepthook
        sys.__excepthook__(exctype, value, traceback)

    sys.excepthook = global_exception_hook

    parser = argparse.ArgumentParser(description="Institutional Quant Engine Runner")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["backtest", "paper", "live"],
        default="backtest",
        help="Execution mode (backtest, paper, live)",
    )
    parser.add_argument(
        "--date", type=str, default="2024-01-15", help="Backtest session date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="RELIANCE,TCS,INFY",
        help="Comma-separated stock symbols to trade",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=15,
        help="Ticking loop running time in seconds (for paper/live demo)",
    )
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    logger.info(f"Starting Quant Engine in {args.mode.upper()} mode for symbols: {symbols}")

    # Load configuration
    config = TradingConfig.from_env()
    config.instruments = symbols

    # Configure mode mapping
    if args.mode == "backtest":
        config.mode = ExecutionMode.BACKTEST
    elif args.mode == "paper":
        config.mode = ExecutionMode.PAPER
    else:
        config.mode = ExecutionMode.LIVE

    # Instantiate central orchestrator
    orchestrator = TradingOrchestrator(config)

    # Register Alpha Signal Models
    logger.info("Registering Alpha Signal Models...")
    momentum_model = TimeSeriesMomentum()
    mr_model = BollingerMeanReversion()

    # New Institutional Alpha Models
    news_alpha = NewsSentimentAlpha()
    options_alpha = OptionFlowAlpha()
    fundamental_alpha = FundamentalPITAlpha()
    basis_alpha = IndexFuturesBasisAlpha()
    vol_alpha = VolatilitySurfaceAlpha()

    # Wrap in composite blend
    composite_alpha = CompositeAlphaModel(
        models=[
            momentum_model,
            mr_model,
            news_alpha,
            options_alpha,
            fundamental_alpha,
            basis_alpha,
            vol_alpha,
        ],
        dynamic_weights=False,
    )
    composite_alpha.static_weights = {
        momentum_model.name: 0.20,
        mr_model.name: 0.15,
        news_alpha.name: 0.10,
        options_alpha.name: 0.15,
        fundamental_alpha.name: 0.15,
        basis_alpha.name: 0.10,
        vol_alpha.name: 0.15,
    }
    orchestrator.register_alpha(composite_alpha)

    # Initialize session levels (high, low, close)
    prev_day_data = {
        "RELIANCE": {"high": 2420.0, "low": 2380.0, "close": 2400.0},
        "TCS": {"high": 3530.0, "low": 3480.0, "close": 3510.0},
        "INFY": {"high": 1520.0, "low": 1490.0, "close": 1505.0},
    }
    orchestrator.initialize_session(prev_day_data)

    if config.mode == ExecutionMode.BACKTEST:
        logger.info(f"Generating mock historical 1m bars for date {args.date}...")
        historical_ticks = []
        for sym in symbols:
            historical_ticks.extend(generate_mock_historical_bars(sym, args.date))

        # Sort ticks chronologically to simulate chronological arrival
        historical_ticks.sort(key=lambda t: t.timestamp)

        # Execute backtest
        metrics = orchestrator.run_backtest(historical_ticks)

        print("\n" + "=" * 50)
        print("BACKTEST RESULTS SUMMARY")
        print("=" * 50)
        print(f"Total Trades Executed : {metrics['total_trades']}")
        print(f"Total Orders Placed  : {metrics['total_orders']}")
        print(f"Orders Filled        : {metrics['orders_filled']}")
        print(f"Orders Rejected      : {metrics['orders_rejected']}")
        print(f"Net Daily P&L        : ₹{metrics['total_pnl']:,.2f}")
        print(f"Final Portfolio NAV  : ₹{metrics['final_nav']:,.2f}")
        print(f"Final Positions      : {metrics['final_positions']}")
        print("=" * 50 + "\n")

    elif config.mode in (ExecutionMode.PAPER, ExecutionMode.LIVE):
        logger.info("Initializing Live Ticking Loop using Multi-Process Topology...")
        try:
            run_multi_process_topology(config, prev_day_data, symbols, args.duration)
        except KeyboardInterrupt:
            logger.info("Loop interrupted by user")
        finally:
            logger.info("Quant Engine shut down successfully")


if __name__ == "__main__":
    main()
