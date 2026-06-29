#!/usr/bin/env python3
"""
Populate database with real market data from Yahoo Finance
"""

import os
import sys

import pandas as pd
import yfinance as yf

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3

from utils.logger import get_logger

logger = get_logger("populate_real_data")

# Use SQLite for local development
DB_PATH = "/Users/pandu/Desktop/quant/quant.db"


def get_connection():
    """Get SQLite connection"""
    return sqlite3.connect(DB_PATH)


def initialize_db():
    """Initialize SQLite database with required tables"""
    conn = get_connection()
    cursor = conn.cursor()

    # Create stocks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stocks (
            symbol TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            sector TEXT,
            market_cap TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create stock_prices table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            price REAL NOT NULL,
            change REAL,
            change_pct REAL,
            volume INTEGER,
            high_52w REAL,
            low_52w REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
    logger.info("Database initialized")


def insert_stock(symbol, name, sector, market_cap="N/A"):
    """Insert stock into database"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO stocks (symbol, name, sector, market_cap, updated_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
    """,
        (symbol, name, sector, market_cap),
    )
    conn.commit()
    conn.close()


def insert_stock_price(symbol, price, change, change_pct, volume, high_52w, low_52w):
    """Insert stock price into database"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO stock_prices (symbol, price, change, change_pct, volume, high_52w, low_52w)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        (symbol, price, change, change_pct, volume, high_52w, low_52w),
    )
    conn.commit()
    conn.close()


def get_latest_prices():
    """Get latest prices from database"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT sp.symbol, s.name, s.sector, s.market_cap,
               sp.price, sp.change, sp.change_pct, sp.volume,
               sp.high_52w, sp.low_52w, sp.timestamp
        FROM stock_prices sp
        JOIN stocks s ON sp.symbol = s.symbol
        WHERE sp.id IN (
            SELECT MAX(id) FROM stock_prices GROUP BY symbol
        )
    """)

    columns = [
        "symbol",
        "name",
        "sector",
        "market_cap",
        "price",
        "change",
        "change_pct",
        "volume",
        "high_52w",
        "low_52w",
        "timestamp",
    ]

    results = []
    for row in cursor.fetchall():
        results.append(dict(zip(columns, row, strict=False)))

    conn.close()
    return results


# Top NIFTY 50 stocks with their sectors
STOCKS = [
    ("RELIANCE", "Reliance Industries Ltd", "ENERGY"),
    ("TCS", "Tata Consultancy Services", "IT"),
    ("HDFCBANK", "HDFC Bank Ltd", "BANKING"),
    ("INFY", "Infosys Ltd", "IT"),
    ("ICICIBANK", "ICICI Bank Ltd", "BANKING"),
    ("HINDUNILVR", "Hindustan Unilever Ltd", "FMCG"),
    ("SBIN", "State Bank of India", "BANKING"),
    ("BHARTIARTL", "Bharti Airtel Ltd", "TELECOM"),
    ("ITC", "ITC Ltd", "FMCG"),
    ("KOTAKBANK", "Kotak Mahindra Bank", "BANKING"),
    ("LT", "Larsen & Toubro Ltd", "INFRASTRUCTURE"),
    ("AXISBANK", "Axis Bank Ltd", "BANKING"),
    ("ASIANPAINT", "Asian Paints Ltd", "CONSUMER GOODS"),
    ("MARUTI", "Maruti Suzuki India Ltd", "AUTOMOBILE"),
    ("SUNPHARMA", "Sun Pharmaceutical Industries Ltd", "PHARMA"),
    ("TATAMOTORS", "Tata Motors Ltd", "AUTOMOBILE"),
    ("BAJFINANCE", "Bajaj Finance Ltd", "FINANCE"),
    ("TITAN", "Titan Company Ltd", "CONSUMER GOODS"),
    ("DMART", "Avenue Supermarts Ltd", "RETAIL"),
    ("WIPRO", "Wipro Ltd", "IT"),
]


def get_stock_data(symbol):
    """Fetch real stock data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        hist = ticker.history(period="5d")

        if hist.empty:
            logger.warning(f"No data for {symbol}")
            return None

        latest = hist.iloc[-1]
        prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else latest["Close"]

        # Calculate 52-week high/low
        year_hist = ticker.history(period="1y")
        high_52w = year_hist["High"].max() if not year_hist.empty else latest["High"]
        low_52w = year_hist["Low"].min() if not year_hist.empty else latest["Low"]

        # Get market cap
        info = ticker.info
        market_cap = info.get("marketCap", 0)
        if market_cap:
            market_cap_str = (
                f"{market_cap / 1e7:.1f}L Cr" if market_cap > 1e7 else f"{market_cap / 1e5:.1f} Cr"
            )
        else:
            market_cap_str = "N/A"

        price = latest["Close"]
        change = price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close > 0 else 0

        return {
            "symbol": symbol,
            "name": info.get("longName", symbol),
            "sector": info.get("sector", "Unknown"),
            "price": round(price, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "volume": int(latest["Volume"]) if pd.notna(latest["Volume"]) else 0,
            "high_52w": round(high_52w, 2),
            "low_52w": round(low_52w, 2),
            "market_cap": market_cap_str,
        }
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None


def main():
    """Main function to populate database"""
    logger.info("Starting real data population...")

    # Initialize database
    initialize_db()

    # Insert stocks and prices
    for symbol, name, sector in STOCKS:
        logger.info(f"Processing {symbol}...")

        # Fetch real data
        stock_data = get_stock_data(symbol)
        if stock_data:
            # Insert stock metadata
            insert_stock(symbol, name, sector, stock_data["market_cap"])

            # Insert price
            insert_stock_price(
                symbol=stock_data["symbol"],
                price=stock_data["price"],
                change=stock_data["change"],
                change_pct=stock_data["change_pct"],
                volume=stock_data["volume"],
                high_52w=stock_data["high_52w"],
                low_52w=stock_data["low_52w"],
            )
            logger.info(
                f"Inserted {symbol}: ₹{stock_data['price']} ({stock_data['change_pct']:+.2f}%)"
            )
        else:
            logger.warning(f"Skipping {symbol} due to data fetch error")

    # Verify data
    logger.info("Verifying data...")
    prices = get_latest_prices()
    logger.info(f"Total stocks in database: {len(prices)}")

    if prices:
        logger.info("\nSample data:")
        for p in prices[:5]:
            logger.info(f"  {p['symbol']}: ₹{p['price']} ({p['change_pct']:+.2f}%)")

    logger.info("Data population complete!")


if __name__ == "__main__":
    main()
