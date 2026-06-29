import asyncio
import datetime
import json
import multiprocessing
import time

import redis

from agents.llm_client import llm
from data.angel_one_feed import AngelOneDataFeeder
from data.multiplexer import TickMultiplexer
from data.upstox_feed import UpstoxDataFeeder
from utils.structured_logger import get_structured_logger

logger = get_structured_logger("multiprocess_scheduler")

# --- Process Workers ---


def run_angelone_feed():
    logger.info("Starting Angel One Feed Process...")
    feeder = AngelOneDataFeeder()
    feeder.start_stream()


def run_upstox_feed():
    logger.info("Starting Upstox Feed Process...")
    feeder = UpstoxDataFeeder()
    asyncio.run(feeder.connect_and_stream())


def run_multiplexer():
    logger.info("Starting Multiplexer Process...")
    mux = TickMultiplexer()
    mux.start()


def run_inference_loop():
    logger.info("Starting Inference Loop Process...")
    # Event-driven ML Inference
    # Reads 'live_ticks' stream, calculates features (Polars-based), runs meta-model, pushes to 'oms_signals'
    redis_client = redis.Redis(host="localhost", port=6379, db=0)
    logger.info("Inference worker connected to Redis. Monitoring live_ticks stream...")

    last_id = "$"  # Listen only to new ticks
    while True:
        try:
            # Block waiting for ticks
            stream_data = redis_client.xread({"live_ticks": last_id}, block=5000, count=100)
            if not stream_data:
                continue

            for _stream_name, messages in stream_data:
                for message_id, message_body in messages:
                    last_id = message_id.decode()
                    # Event-driven trigger: Calculate features & run models
                    # e.g., run_ensemble_inference(message_body)
                    tick = {k.decode(): v.decode() for k, v in message_body.items()}
                    logger.debug(
                        f"Event Triggered: Tick received for {tick.get('symbol')}. Running ML..."
                    )

                    # Mock output signal
                    signal = {
                        "symbol": tick.get("symbol"),
                        "ltp": float(tick.get("ltp", 0)),
                        "timestamp": int(time.time()),
                        "signal": "BUY",
                        "probability": 0.58,
                    }
                    redis_client.xadd("oms_signals", signal, maxlen=10000)
        except Exception as e:
            logger.error(f"Error in inference worker: {e}")
            time.sleep(2)


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


async def cron_event_loop():
    # Pre-market job at 6:00 AM
    pre_market_task = asyncio.create_task(self_correcting_timer(6, 0, pre_market_intelligence_job))
    # Post-market job at 4:00 PM
    post_market_task = asyncio.create_task(self_correcting_timer(16, 0, post_trade_analysis_job))
    await asyncio.gather(pre_market_task, post_market_task)


# --- Master Orchestrator ---


def main():
    logger.info("Initializing Institutional Multiprocessing Orchestrator...")

    processes = [
        multiprocessing.Process(target=run_angelone_feed, name="AngelOneFeed"),
        multiprocessing.Process(target=run_upstox_feed, name="UpstoxFeed"),
        multiprocessing.Process(target=run_multiplexer, name="Multiplexer"),
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
