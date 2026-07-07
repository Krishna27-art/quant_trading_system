import asyncio
import datetime
import json
import multiprocessing
import time

import redis

from agents.llm_client import llm
from data_platform.feeds.feed_manager import FeedManager, TickData
from utils.structured_logger import get_structured_logger

logger = get_structured_logger("multiprocess_scheduler")

# --- Process Workers ---


_quotes_cache = {}
_last_fetch_time = 0.0


async def upstox_ws_connect(symbols: list[str], on_tick) -> None:
    """Simulates a live WebSocket feed using periodic batch REST polls."""
    from data_platform.upstox_client import get_bulk_quotes
    from data_platform.feeds.feed_manager import TickData, FeedTier

    clean_symbols = [s.replace(".NS", "").upper() for s in symbols]
    logger.info("Initializing Upstox WebSocket simulator...")

    while True:
        try:
            now = time.time()
            quotes = get_bulk_quotes(clean_symbols)
            for sym in symbols:
                clean_sym = sym.replace(".NS", "").upper()
                quote = quotes.get(clean_sym)
                if quote:
                    tick = TickData(
                        symbol=sym,
                        ltp=float(quote["last_price"] or 0.0),
                        bid=float(quote.get("bid") or quote["last_price"] or 0.0),
                        ask=float(quote.get("ask") or quote["last_price"] or 0.0),
                        volume=int(quote["volume"] or 0),
                        timestamp=float(quote["timestamp"] or now),
                        received_at=now,
                        feed_tier=FeedTier.PRIMARY
                    )
                    on_tick(tick)
            await asyncio.sleep(2.0)
        except Exception as e:
            logger.error(f"Upstox WS simulator error: {e}")
            await asyncio.sleep(5.0)


def upstox_poll_tick(symbol: str) -> TickData | None:
    """Polls a single symbol using cached batch REST responses to avoid rate limits."""
    global _quotes_cache, _last_fetch_time
    now = time.time()

    # Fetch all symbols at once if cache is older than 2 seconds
    if now - _last_fetch_time > 2.0 or not _quotes_cache:
        try:
            from data_platform.upstox_client import get_bulk_quotes
            from config.universe import NSE_UNIVERSE
            symbols_list = [s["symbol"] for s in NSE_UNIVERSE]
            _quotes_cache = get_bulk_quotes(symbols_list)
            _last_fetch_time = now
        except Exception as e:
            logger.error(f"Failed to fetch bulk quotes for REST fallback: {e}")
            return None

    sym_clean = symbol.replace(".NS", "").upper()
    quote = _quotes_cache.get(sym_clean)
    if not quote:
        return None

    from data_platform.feeds.feed_manager import TickData, FeedTier
    return TickData(
        symbol=symbol,
        ltp=float(quote["last_price"] or 0.0),
        bid=float(quote.get("bid") or quote["last_price"] or 0.0),
        ask=float(quote.get("ask") or quote["last_price"] or 0.0),
        volume=int(quote["volume"] or 0),
        timestamp=float(quote["timestamp"] or now),
        received_at=now,
        feed_tier=FeedTier.FALLBACK
    )


def run_feed_manager():
    """Run the unified FeedManager instead of separate feed processes."""
    logger.info("Starting Feed Manager Process with Upstox Integration...")

    redis_client = redis.Redis(host="localhost", port=6379, db=0)

    # on_tick publishes to 'live_ticks' stream
    def on_tick_callback(tick: TickData):
        try:
            # Publish to Redis stream
            redis_client.xadd("live_ticks", tick.to_dict())
            logger.debug(f"Tick published to Redis: {tick.symbol} @ {tick.ltp}")
        except Exception as e:
            logger.error(f"Failed to publish tick to Redis: {e}")

    # Create a FeedManager instance
    feed_manager = FeedManager(
        ws_connect_fn=upstox_ws_connect,
        rest_poll_fn=upstox_poll_tick,
        staleness_threshold_s=15.0,  # 15s threshold
        max_consecutive_failures=5,
        rest_poll_interval_s=2.0,
        on_tick=on_tick_callback,
        on_failover=lambda old, new: logger.warning(f"Feed failover: {old} -> {new}"),
        redis_client=redis_client
    )

    # Subscribe to symbols
    from config.universe import NSE_UNIVERSE
    symbols = [f"{s['symbol']}.NS" for s in NSE_UNIVERSE]
    feed_manager.subscribe(symbols)

    # Start the feed manager (this is async, so run in event loop)
    try:
        asyncio.run(feed_manager.start())
    except KeyboardInterrupt:
        logger.info("Feed manager stopped")
    except Exception as e:
        logger.error(f"Feed manager error: {e}")


def run_inference_loop():
    """
    Event-driven ML inference worker.
    Wakes on each Redis tick, runs the full generate_live_predictions pipeline,
    and results are written directly to the database by the prediction script.
    Runs on a short sleep cycle when market is active rather than blocking on Redis.
    """
    logger.info("Starting Inference Loop Process (real ML pipeline)...")
    import time as _time

    # Import here so the subprocess only loads what it needs
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    try:
        from scripts.generate_live_predictions import run as _run_predictions
    except ImportError as e:
        logger.error(f"Failed to import generate_live_predictions: {e}. Inference loop cannot start.")
        return

    redis_client = redis.Redis(host="localhost", port=6379, db=0)
    logger.info("Inference worker connected to Redis. Waiting for ticks...")

    last_id = "$"
    while True:
        try:
            # Block up to 60s for any new tick before triggering inference
            stream_data = redis_client.xread({"live_ticks": last_id}, block=60000, count=1)
            if stream_data:
                for _stream_name, messages in stream_data:
                    if messages:
                        last_id = messages[-1][0].decode()

                # Run the full prediction pipeline (writes to DB directly)
                logger.info("Inference trigger: running generate_live_predictions on new tick...")
                _run_predictions()
                logger.info("Inference cycle complete.")
            else:
                logger.debug("No new ticks received in 60s heartbeat window.")

        except Exception as e:
            logger.error(f"Error in inference worker: {e}")
            _time.sleep(5)


def run_execution_loop():
    logger.info("Starting Execution (OMS) Process...")
    # Listens to 'oms_signals' and runs Pre-trade Risk Checks -> executes Zerodha/Paper order
    redis_client = redis.Redis(host="localhost", port=6379, db=0)
    logger.info("OMS worker connected to Redis. Monitoring oms_signals stream...")

    last_id = "$"
    while True:
        try:
            stream_data = redis_client.xread({"oms_signals": last_id}, block=5000, count=10)
            if not stream_data:
                continue

            for _stream_name, messages in stream_data:
                for message_id, message_body in messages:
                    last_id = message_id.decode()
                    signal = {k.decode(): v.decode() for k, v in message_body.items()}
                    logger.info(
                        f"Execution Gate: Processing signal for {signal.get('symbol')} ({signal.get('signal')})"
                    )

                    # Runs risk engine check & routes order
                    # e.g., oms.route_order(signal)
        except Exception as e:
            logger.error(f"Error in OMS execution: {e}")
            time.sleep(2)


# --- Deterministic Asyncio-based Event Cron for LLM Jobs ---


async def self_correcting_timer(target_hour, target_minute, callback_coro):
    """Self-correcting timer loop that handles cron-like tasks without drift."""
    while True:
        time.time()
        # Find next occurrences
        dt_now = datetime.datetime.now()
        dt_target = dt_now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
        if dt_target <= dt_now:
            dt_target += datetime.timedelta(days=1)

        wait_seconds = (dt_target - dt_now).total_seconds()
        logger.info(f"Next job scheduled for {dt_target}. Waiting {wait_seconds:.2f} seconds...")
        await asyncio.sleep(wait_seconds)

        # Trigger
        try:
            await callback_coro()
        except Exception as e:
            logger.error(f"Error executing scheduled task: {e}", exc_info=True)


async def pre_market_intelligence_job():
    logger.info("Running Pre-Market Intelligence Job...")
    system_prompt = (
        "You are an Indian equities day trading analyst. Analyze the following data "
        "and produce a structured pre-market brief with these sections: Global Cues, "
        "Sector Outlook, Stocks to Watch, Risk Factors. Be specific and quantitative where possible. "
        "Return the output as a clean JSON object."
    )
    user_prompt = "Dow Jones: +1.2%, Nasdaq: +1.5%, Nikkei: +0.8%, SGX Nifty: +100pts. India VIX: 12.5. FII: +1500cr. Top News: IT sector earnings beat expectations."

    response_text = await llm.ask_claude_async(system_prompt, user_prompt)
    try:
        parsed = json.loads(response_text)
        logger.info(f"Pre-Market Intelligence generated regime: {parsed.get('regime', 'Unknown')}")
    except Exception:
        logger.error("Failed to parse LLM pre-market JSON", exc_info=True)


async def post_trade_analysis_job():
    logger.info("Running Post-Trade Analysis Job...")
    system_prompt = (
        "Analyze these day trades. Identify patterns in winners vs losers. "
        "What market conditions predicted success? What should be avoided tomorrow?"
    )
    user_prompt = "Trades: [WIN on TCS +1.5%], [LOSS on HDFC -0.75%]. Market was trending up."

    await llm.ask_claude_async(system_prompt, user_prompt)
    logger.info("Post-Trade Reflection generated.")


async def daily_snapshot_pruning_job():
    logger.info("Running Daily Snapshot Pruning Job...")
    try:
        from data_platform.sources.ingestion.raw_bronze import RawBronzeLayer
        from utils.versioned_datasets import VersionedDataset
        from config.universe import NSE_UNIVERSE
        from config.settings import BRONZE_EQUITY_HISTORY_DIR
        
        bronze_layer = RawBronzeLayer()
        versioned_store = VersionedDataset(BRONZE_EQUITY_HISTORY_DIR)
        
        for stock in NSE_UNIVERSE:
            sym = stock["symbol"]
            dataset_name = f"equity_history_{sym}"
            
            # Prune bronze raw responses
            try:
                deleted_bronze = bronze_layer.delete_old_snapshots(dataset_name, keep_count=10)
                if deleted_bronze > 0:
                    logger.info(f"Pruned {deleted_bronze} old raw bronze snapshots for {dataset_name}")
            except Exception as e:
                logger.error(f"Failed to prune raw bronze snapshots for {dataset_name}: {e}")
                
            # Prune versioned dataset snapshots
            try:
                deleted_versioned = versioned_store.delete_old_snapshots(dataset_name, keep_count=10)
                if deleted_versioned > 0:
                    logger.info(f"Pruned {deleted_versioned} old versioned snapshots for {dataset_name}")
            except Exception as e:
                logger.error(f"Failed to prune versioned snapshots for {dataset_name}: {e}")
                
        logger.info("Daily Snapshot Pruning Job completed successfully.")
    except Exception as e:
        logger.error(f"Error executing daily snapshot pruning: {e}")


async def outcome_resolution_job():
    logger.info("Running Daily Outcome Resolution and Calibration Job...")
    try:
        from scripts.resolve_outcomes import resolve_unresolved_predictions
        # Run synchronous function in background thread executor to prevent blocking the event loop
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, resolve_unresolved_predictions)
        logger.info("Daily Outcome Resolution and Calibration Job completed successfully.")
    except Exception as e:
        logger.error(f"Error executing outcome resolution and calibration job: {e}")


async def cron_event_loop():
    # Pre-market job at 6:00 AM
    pre_market_task = asyncio.create_task(self_correcting_timer(6, 0, pre_market_intelligence_job))
    # Post-market job at 4:00 PM
    post_market_task = asyncio.create_task(self_correcting_timer(16, 0, post_trade_analysis_job))
    # Daily outcome resolution and calibration at 5:30 PM (17:30 IST)
    resolution_task = asyncio.create_task(self_correcting_timer(17, 30, outcome_resolution_job))
    # Daily snapshot pruning job at 11:00 PM
    pruning_task = asyncio.create_task(self_correcting_timer(23, 0, daily_snapshot_pruning_job))
    await asyncio.gather(pre_market_task, post_market_task, resolution_task, pruning_task)


# --- Master Orchestrator ---


def main():
    logger.info("Initializing Institutional Multiprocessing Orchestrator...")

    processes = [
        multiprocessing.Process(target=run_feed_manager, name="FeedManager"),
        multiprocessing.Process(target=run_inference_loop, name="InferenceEngine"),
        multiprocessing.Process(target=run_execution_loop, name="ExecutionOMS"),
    ]

    # Start all background workers
    for p in processes:
        p.daemon = True
        p.start()
        logger.info(f"Started worker process: {p.name} [PID: {p.pid}]")

    # Start Async Cron Loop in the main thread (handles LLM Pre/Post tasks)
    try:
        asyncio.run(cron_event_loop())
    except KeyboardInterrupt:
        logger.info("Received terminate signal. Shutting down worker processes...")
        for p in processes:
            p.terminate()
            p.join()
        logger.info("Shutdown completed.")


if __name__ == "__main__":
    main()
