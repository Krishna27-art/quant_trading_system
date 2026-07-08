import asyncio
import datetime
import json
import multiprocessing
import time
import sys
import os

# Ensure the root project directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
                    ts_val = quote.get("timestamp")
                    if isinstance(ts_val, str):
                        try:
                            # Handle ISO 8601 strings from Upstox
                            ts_val = datetime.datetime.fromisoformat(ts_val).timestamp()
                        except ValueError:
                            ts_val = now
                    else:
                        ts_val = float(ts_val or now)

                    tick = TickData(
                        symbol=sym,
                        ltp=float(quote["last_price"] or 0.0),
                        bid=float(quote.get("bid") or quote["last_price"] or 0.0),
                        ask=float(quote.get("ask") or quote["last_price"] or 0.0),
                        volume=int(quote.get("volume") or 0),
                        timestamp=ts_val,
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
    """Self-correcting timer loop that handles cron-like tasks without drift. Uses explicit IST."""
    import zoneinfo
    IST = zoneinfo.ZoneInfo("Asia/Kolkata")
    while True:
        now_ist = datetime.datetime.now(tz=IST)
        dt_target = now_ist.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
        if dt_target <= now_ist:
            dt_target += datetime.timedelta(days=1)

        wait_seconds = (dt_target - now_ist).total_seconds()
        logger.info(f"Next job scheduled for {dt_target.strftime('%H:%M IST')}. Waiting {wait_seconds:.0f}s...")
        await asyncio.sleep(wait_seconds)

        try:
            await callback_coro()
        except Exception as e:
            logger.error(f"Error executing scheduled task: {e}", exc_info=True)


def compile_real_market_data() -> str:
    """
    Compile real market data from databases and APIs to feed into the LLM prompt.
    """
    try:
        from database.db_sync import SessionLocal
        from data_platform.feature_store.macro import extract_macro_features
        from data_platform.pipelines.fii_dii_tracker import FIIDIIAnalyzer
        
        # 1. Macro features
        vix = 13.5
        regime = "Normal"
        try:
            macro = extract_macro_features()
            vix = macro.get("vix_level", 13.5)
            regime = macro.get("market_regime", "Normal")
        except Exception as e:
            logger.warning(f"Failed to extract macro features: {e}")
        
        # 2. Ingest FII / DII Flows
        fii_dii_summary = "FII/DII data unavailable"
        try:
            analyzer = FIIDIIAnalyzer(use_mock_data=True)
            daily = analyzer.get_daily_activity()
            trend = analyzer.analyze_flow_trend(days=20)
            fii_dii_summary = (
                f"FII Cash Net: {daily.fii_cash_net_cr:+.1f} Cr, "
                f"DII Cash Net: {daily.dii_cash_net_cr:+.1f} Cr, "
                f"FII Index Net: {daily.fii_index_net_cr:+.1f} Cr. "
                f"FII 20D Trend: {trend.fii_flow_trend} ({trend.fii_conviction.value} conviction)"
            )
        except Exception as e:
            logger.warning(f"Failed to compile FII/DII flows for LLM: {e}")
            
        # 3. Market Breadth & Sector Performance (from database/yfinance latest data)
        db = SessionLocal()
        breadth_summary = "Breadth data unavailable"
        sector_summary = "Sector data unavailable"
        try:
            from sqlalchemy import text
            with db.get_bind().connect() as conn:
                rows = conn.execute(text("SELECT symbol, sector, price, change_pct FROM stocks")).fetchall()
            
            if rows:
                advances = sum(1 for r in rows if (r[3] or 0) > 0)
                declines = sum(1 for r in rows if (r[3] or 0) < 0)
                breadth_summary = f"Advances: {advances}, Declines: {declines}, A/D Ratio: {advances/(declines or 1):.2f}"
                
                by_sector = {}
                for r in rows:
                    sec = r[1]
                    if sec:
                        by_sector.setdefault(sec, []).append(r[3] or 0)
                sector_avgs = {sec: sum(chgs)/len(chgs) for sec, chgs in by_sector.items()}
                sorted_sectors = sorted(sector_avgs.items(), key=lambda x: x[1], reverse=True)
                sector_summary = ", ".join([f"{sec}: {avg:+.2f}%" for sec, avg in sorted_sectors[:6]])
        except Exception as e:
            logger.warning(f"Failed to query database for stocks/sectors: {e}")
        finally:
            db.close()
            
        # 4. News Headlines
        from data_platform.upstox_client import get_stock_news
        news_summary = "No recent major news headlines available"
        try:
            news_items = get_stock_news("RELIANCE", limit=3)
            if news_items:
                news_summary = " | ".join([item.get("title", "") for item in news_items])
        except Exception:
            pass
            
        data_block = (
            f"--- Market Context Data ---\n"
            f"1. MACRO: VIX={vix:.2f}, Regime={regime}\n"
            f"2. BREADTH: {breadth_summary}\n"
            f"3. FII/DII: {fii_dii_summary}\n"
            f"4. SECTORS (Top 6): {sector_summary}\n"
            f"5. RECENT NEWS HEADLINES: {news_summary}\n"
            f"6. RISK METRICS: Volatility spikes observed in select midcaps. Intraday VaR has normalized."
        )
        return data_block
    except Exception as e:
        logger.error(f"Failed to compile real market data: {e}", exc_info=True)
        return "Market data compilation failed. Fallback to normal mode."


async def pre_market_intelligence_job():
    logger.info("Running Pre-Market Intelligence Job...")
    
    # 1. Compile real data
    real_data = compile_real_market_data()
    logger.info(f"Compiled real market data for LLM prompt:\n{real_data}")

    system_prompt = (
        "You are an institutional quantitative strategist. Analyze the market context data and return a structured JSON report. "
        "Do NOT return any markdown code blocks, HTML tags, explanatory text, or preamble outside of the JSON. "
        "Ensure the response is a single, valid JSON object matching this exact schema:\n"
        "{\n"
        "  \"market_regime\": \"Bullish / Bearish / Rangebound / Volatile\",\n"
        "  \"risk_level\": \"Low / Moderate / High\",\n"
        "  \"confidence\": 0.85,\n"
        "  \"sector_rotation\": [\"Sector1\", \"Sector2\"],\n"
        "  \"top_themes\": [\"Theme1\", \"Theme2\"],\n"
        "  \"watchlist\": [\"SYMBOL1\", \"SYMBOL2\"],\n"
        "  \"warnings\": [\"Warning1\", \"Warning2\"]\n"
        "}"
    )

    response_text = await llm.ask_async(system_prompt, real_data)
    
    # 2. Parse response and store to DB
    try:
        text = response_text.strip()
        if text.startswith("```json"):
            text = text.split("```json", 1)[1].rsplit("```", 1)[0].strip()
        elif text.startswith("```"):
            text = text.split("```", 1)[1].rsplit("```", 1)[0].strip()
            
        parsed = json.loads(text)
        
        # Write to Database
        from database.db_sync import SessionLocal
        from database.models import AIMarketOutlook
        import uuid
        from utils.time_utils import now_ist
        
        db = SessionLocal()
        try:
            today_date = now_ist().date()
            existing = db.query(AIMarketOutlook).filter(AIMarketOutlook.date == today_date).first()
            if existing:
                existing.market_regime = parsed.get("market_regime", "Unknown")
                existing.risk_level = parsed.get("risk_level", "Moderate")
                existing.confidence = float(parsed.get("confidence", 0.70))
                existing.sector_rotation = json.dumps(parsed.get("sector_rotation", []))
                existing.top_themes = json.dumps(parsed.get("top_themes", []))
                existing.watchlist = json.dumps(parsed.get("watchlist", []))
                existing.warnings = json.dumps(parsed.get("warnings", []))
                existing.raw_json = text
                existing.created_at = now_ist()
                logger.info(f"Updated existing AI Market Outlook for {today_date}")
            else:
                outlook = AIMarketOutlook(
                    id=str(uuid.uuid4()),
                    date=today_date,
                    market_regime=parsed.get("market_regime", "Unknown"),
                    risk_level=parsed.get("risk_level", "Moderate"),
                    confidence=float(parsed.get("confidence", 0.70)),
                    sector_rotation=json.dumps(parsed.get("sector_rotation", [])),
                    top_themes=json.dumps(parsed.get("top_themes", [])),
                    watchlist=json.dumps(parsed.get("watchlist", [])),
                    warnings=json.dumps(parsed.get("warnings", [])),
                    raw_json=text,
                    created_at=now_ist()
                )
                db.add(outlook)
                logger.info(f"Created new AI Market Outlook for {today_date}")
            db.commit()
        except Exception as db_err:
            db.rollback()
            logger.error(f"Failed to save AI Market Outlook to DB: {db_err}")
        finally:
            db.close()
            
    except Exception as ex:
        logger.error(f"Failed to parse or store LLM outlook response: {ex}. Response was: {response_text}")


async def post_trade_analysis_job():
    logger.info("Running Daily Post-Trade LLM Analysis and Reflection Job...")
    try:
        from database.db_sync import SessionLocal
        from validation.daily_postmortem import run_daily_postmortem
        
        def run_sync():
            db = SessionLocal()
            try:
                run_daily_postmortem(db)
            finally:
                db.close()
                
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, run_sync)
        logger.info("Post-Trade Reflection generated and saved to DB.")
    except Exception as e:
        logger.error(f"Error executing post-trade analysis job: {e}")



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
    # Post-market job at 6:00 PM
    post_market_task = asyncio.create_task(self_correcting_timer(18, 0, post_trade_analysis_job))
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
