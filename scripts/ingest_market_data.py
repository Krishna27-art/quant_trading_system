"""
Market Data Ingestor.
Fetches real market prices, changes, volumes, and 52-week high/low data for the 150-stock NSE universe
using yfinance and populates the database (stocks and stock_prices tables).
Can run as a one-shot script or in a daemon loop.
"""

import os
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.universe import NSE_UNIVERSE
from database.connection import (
    initialize_pool,
    create_tables,
    insert_stock,
    insert_stock_price,
    execute_write,
)
from utils.logger import get_logger

logger = get_logger("market_data_ingestor")

# Standardize logger to output to stdout cleanly
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def fetch_and_store_stock(stock_info: dict) -> bool:
    symbol = stock_info["symbol"]
    # yfinance requires suffix .NS for Indian markets
    yf_symbol = f"{symbol}.NS"
    
    try:
        # Upsert stock metadata
        insert_stock({
            "symbol": symbol,
            "name": stock_info["name"],
            "sector": stock_info["sector"],
            "market_cap": stock_info["cap"]
        })
        
        # Download recent history to compute change and get volume
        ticker = yf.Ticker(yf_symbol)
        hist = ticker.history(period="5d")
        
        if hist.empty:
            logger.warning(f"No history returned for {symbol} ({yf_symbol})")
            return False
            
        latest = hist.iloc[-1]
        prev_close = hist.iloc[-2]["Close"] if len(hist) > 1 else latest["Open"]
        
        current_price = float(latest["Close"])
        change = current_price - float(prev_close)
        change_pct = (change / float(prev_close)) * 100.0 if prev_close != 0 else 0.0
        volume = int(latest["Volume"])
        
        # Try to get 52-week high and low from fast_info if available, else estimate
        high_52w = current_price
        low_52w = current_price
        try:
            fast_info = ticker.fast_info
            high_52w = getattr(fast_info, "year_high", current_price)
            low_52w = getattr(fast_info, "year_low", current_price)
        except Exception:
            # Fallback to current price if fast_info fails
            pass
            
        price_data = {
            "symbol": symbol,
            "price": current_price,
            "change": round(change, 2),
            "change_pct": round(change_pct, 3),
            "volume": volume,
            "high_52w": round(high_52w, 2),
            "low_52w": round(low_52w, 2),
        }
        
        insert_stock_price(price_data)
        logger.info(f"Successfully ingested {symbol}: Price={current_price:.2f}, Change={change_pct:+.2f}%")
        return True
        
    except Exception as e:
        logger.error(f"Error ingesting {symbol}: {e}")
        return False


def run_ingest(max_workers: int = 15):
    logger.info("Initializing database and tables...")
    initialize_pool()
    create_tables()
    
    logger.info(f"Starting market data ingestion for {len(NSE_UNIVERSE)} stocks...")
    start_time = time.time()
    
    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_and_store_stock, stock): stock for stock in NSE_UNIVERSE}
        for future in as_completed(futures):
            stock = futures[future]
            try:
                success = future.result()
                if success:
                    success_count += 1
            except Exception as exc:
                logger.error(f"{stock['symbol']} generated an exception: {exc}")
                
    elapsed = time.time() - start_time
    logger.info(f"Ingestion complete: {success_count}/{len(NSE_UNIVERSE)} stocks successfully updated in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    # If daemon argument is passed, run continuously
    daemon_mode = "--daemon" in sys.argv
    
    while True:
        run_ingest()
        if not daemon_mode:
            break
        logger.info("Sleeping for 5 minutes before next refresh...")
        time.sleep(300)
