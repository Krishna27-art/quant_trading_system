"""
Missing Prices Sync Script

Cleans up symbol names and fetches missing prices from yfinance.
Resolves the "Missing prices" DEGRADED health status by syncing
the database with the current NSE_UNIVERSE configuration.
"""

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.universe import NSE_UNIVERSE
from database.db_sync import SessionLocal
from database.models import Tick
from utils.logger import get_logger
from utils.time_utils import now_ist

logger = get_logger("sync_missing_prices")

# Symbol name corrections for renamed/delisted NSE symbols
SYMBOL_CORRECTIONS = {
    "AMARAJABAT": "AMARAJAENERG",
    "GMRINFRA": "GMRAIRPORT",
    "L&TFH": "LTF",
    "CEATTLTD": "CEATLTD",
    "CENTURYTEX": "CENTURYEXT",  # May need verification
    "ALOKTEXT": "ALEXANDER",   # May need verification
}


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol name by stripping whitespace and applying corrections."""
    symbol = symbol.strip().upper()
    return SYMBOL_CORRECTIONS.get(symbol, symbol)


def fetch_price(symbol: str) -> float | None:
    """Fetch latest price from yfinance for a symbol."""
    try:
        ticker = f"{symbol}.NS"
        data = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=True)
        if data is None or data.empty:
            return None
        latest = float(data["Close"].iloc[-1])
        return latest if latest > 0 else None
    except Exception as e:
        logger.warning(f"Failed to fetch price for {symbol}: {e}")
        return None


def sync_missing_prices():
    """Sync missing prices from NSE_UNIVERSE to database."""
    logger.info("Starting missing prices sync")
    
    db = SessionLocal()
    
    try:
        # Get current symbols in database
        existing_symbols = set()
        try:
            from sqlalchemy import text
            result = db.execute(text("SELECT DISTINCT symbol FROM ticks ORDER BY time DESC LIMIT 10000")).fetchall()
            existing_symbols = {row[0] for row in result if row[0]}
        except Exception as e:
            logger.warning(f"Could not fetch existing symbols: {e}")
        
        # Get expected symbols from NSE_UNIVERSE
        expected_symbols = {s["symbol"] for s in NSE_UNIVERSE}
        
        # Find missing symbols
        missing_symbols = expected_symbols - existing_symbols
        logger.info(f"Expected {len(expected_symbols)} symbols, found {len(existing_symbols)}, missing {len(missing_symbols)}")
        
        if not missing_symbols:
            logger.info("No missing symbols to sync")
            return
        
        # Fetch prices for missing symbols
        synced_count = 0
        failed_symbols = []
        
        for symbol in missing_symbols:
            normalized = normalize_symbol(symbol)
            price = fetch_price(normalized)
            
            if price:
                tick = Tick(
                    time=now_ist(),
                    symbol=symbol,
                    ltp=price,
                    volume=0,
                )
                db.add(tick)
                synced_count += 1
                logger.info(f"Synced {symbol} -> ₹{price}")
            else:
                failed_symbols.append(symbol)
                logger.warning(f"Failed to fetch price for {symbol}")
        
        db.commit()
        logger.info(f"Synced {synced_count} symbols, failed {len(failed_symbols)}")
        
        if failed_symbols:
            logger.warning(f"Failed symbols: {', '.join(failed_symbols[:20])}")
        
    except Exception as e:
        logger.error(f"Sync failed: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    sync_missing_prices()
