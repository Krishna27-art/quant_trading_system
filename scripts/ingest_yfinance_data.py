"""
Real Data Ingestion Script from Yahoo Finance

Fetches real stock data from Yahoo Finance and populates the database.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf

from database.connection import create_tables, insert_stock, insert_stock_price
from utils.logger import get_logger

logger = get_logger("ingest_yfinance_data")

# Top stocks to fetch (NSE stocks with .NS suffix)
TOP_STOCKS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "ITC.NS",
    "LT.NS",
    "MARUTI.NS",
    "SUNPHARMA.NS",
    "TATAMOTORS.NS",
    "AXISBANK.NS",
    "DRREDDY.NS",
]

SECTOR_MAPPING = {
    "RELIANCE.NS": "Energy",
    "TCS.NS": "IT",
    "HDFCBANK.NS": "Banking",
    "ICICIBANK.NS": "Banking",
    "INFY.NS": "IT",
    "SBIN.NS": "Banking",
    "BHARTIARTL.NS": "Telecom",
    "ITC.NS": "FMCG",
    "LT.NS": "Capital Goods",
    "MARUTI.NS": "Auto",
    "SUNPHARMA.NS": "Pharma",
    "TATAMOTORS.NS": "Auto",
    "AXISBANK.NS": "Banking",
    "DRREDDY.NS": "Pharma",
}


def fetch_stock_data(symbol):
    """Fetch stock data from Yahoo Finance."""
    try:
        logger.info(f"Fetching data for {symbol}")

        # Get stock info
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Get latest price data
        hist = ticker.history(period="5d")

        if hist is None or hist.empty:
            logger.warning(f"No price data for {symbol}")
            return None

        # Get latest row
        latest = hist.iloc[-1]
        previous = hist.iloc[-2] if len(hist) > 1 else hist.iloc[-1]

        # Calculate change
        price = float(latest["Close"])
        prev_price = float(previous["Close"])
        change = price - prev_price
        change_pct = (change / prev_price) * 100 if prev_price > 0 else 0

        # Get 52-week high/low from info
        high_52w = float(info.get("fiftyTwoWeekHigh", 0))
        low_52w = float(info.get("fiftyTwoWeekLow", 0))

        # Get name from info
        name = info.get("longName", symbol.replace(".NS", ""))

        # Get sector from mapping or info
        sector = SECTOR_MAPPING.get(symbol, info.get("sector", "Unknown"))

        return {
            "symbol": symbol.replace(".NS", ""),  # Remove .NS for database
            "name": name,
            "sector": sector,
            "price": price,
            "change": change,
            "change_pct": change_pct,
            "volume": int(latest["Volume"]),
            "high_52w": high_52w,
            "low_52w": low_52w,
        }

    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return None


def ingest_data():
    """Ingest real data from Yahoo Finance to database."""
    logger.info("Starting Yahoo Finance data ingestion")

    # Create tables
    create_tables()

    success_count = 0
    for symbol in TOP_STOCKS:
        try:
            # Fetch stock data
            stock_data = fetch_stock_data(symbol)
            if not stock_data:
                continue

            # Insert stock
            insert_stock(
                {
                    "symbol": stock_data["symbol"],
                    "name": stock_data["name"],
                    "sector": stock_data["sector"],
                    "market_cap": "",
                }
            )

            # Insert stock price
            insert_stock_price(
                {
                    "symbol": stock_data["symbol"],
                    "price": stock_data["price"],
                    "change": stock_data["change"],
                    "change_pct": stock_data["change_pct"],
                    "volume": stock_data["volume"],
                    "high_52w": stock_data["high_52w"],
                    "low_52w": stock_data["low_52w"],
                }
            )

            success_count += 1
            logger.info(
                f"Successfully ingested {stock_data['symbol']}: ₹{stock_data['price']:.2f} ({stock_data['change_pct']:+.2f}%)"
            )

        except Exception as e:
            logger.error(f"Error ingesting {symbol}: {e}")

    logger.info(
        f"Ingestion complete. Successfully ingested {success_count}/{len(TOP_STOCKS)} stocks"
    )


if __name__ == "__main__":
    ingest_data()
