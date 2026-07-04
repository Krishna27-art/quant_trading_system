#!/usr/bin/env python3
"""
Populate database with real market data from Upstox API
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables FIRST before any imports
load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import execute_query, get_connection
from data_platform.upstox_client import get_instrument_map
from utils.logger import get_logger

logger = get_logger("populate_upstox_data")

# Top NIFTY 50 stocks with their sectors (instrument keys will be fetched from Upstox)
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


def initialize_tables():
    """Initialize database tables if they don't exist"""
    logger.info("Initializing database tables...")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Create stocks table
        create_stocks = """
            CREATE TABLE IF NOT EXISTS stocks (
                symbol VARCHAR(32) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                sector VARCHAR(100),
                market_cap VARCHAR(50),
                instrument_key VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        cursor.execute(create_stocks)
        
        # Create stock_prices table
        create_prices = """
            CREATE TABLE IF NOT EXISTS stock_prices (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(32) NOT NULL,
                price DECIMAL(12, 2) NOT NULL,
                change DECIMAL(12, 2),
                change_pct DECIMAL(8, 2),
                volume BIGINT,
                high_52w DECIMAL(12, 2),
                low_52w DECIMAL(12, 2),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (symbol) REFERENCES stocks(symbol) ON DELETE CASCADE
            )
        """
        cursor.execute(create_prices)
        
        conn.commit()
        logger.info("Tables initialized")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error creating tables: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def insert_stock(symbol, name, sector, instrument_key=None, market_cap="N/A"):
    """Insert stock into database"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        query = """
            INSERT INTO stocks (symbol, name, sector, instrument_key, market_cap, updated_at)
            VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (symbol) DO UPDATE SET
                name = EXCLUDED.name,
                sector = EXCLUDED.sector,
                instrument_key = COALESCE(EXCLUDED.instrument_key, stocks.instrument_key),
                market_cap = EXCLUDED.market_cap,
                updated_at = CURRENT_TIMESTAMP
        """
        cursor.execute(query, (symbol, name, sector, instrument_key, market_cap))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Error inserting stock {symbol}: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def insert_stock_price(symbol, price, change, change_pct, volume, high_52w, low_52w):
    """Insert stock price into database"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        query = """
            INSERT INTO stock_prices (symbol, price, change, change_pct, volume, high_52w, low_52w)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (symbol, price, change, change_pct, volume, high_52w, low_52w))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Error inserting price for {symbol}: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def get_stock_data_from_upstox(instrument_key):
    """Fetch real stock data from Upstox API using the client"""
    from data_platform.upstox_client import get_stock_quote
    
    try:
        quote = get_stock_quote(instrument_key)
        if quote and 'ohlc' in quote:
            ohlc = quote['ohlc']
            last_price = quote.get('last_price', ohlc.get('close', 0))
            
            change = quote.get('net_change', 0)
            prev_close = ohlc.get('open', 0)
            change_pct = (change / prev_close * 100) if prev_close > 0 else 0
            volume = quote.get('volume', 0)
            
            # For 52-week high/low, we'd need historical data
            # Using current price as placeholder
            high_52w = last_price * 1.2  # Placeholder
            low_52w = last_price * 0.8   # Placeholder
            
            return {
                'price': round(last_price, 2),
                'change': round(change, 2),
                'change_pct': round(change_pct, 2),
                'volume': int(volume) if volume else 0,
                'high_52w': round(high_52w, 2),
                'low_52w': round(low_52w, 2),
            }
        return None
    except Exception as e:
        logger.error(f"Error fetching data for {instrument_key}: {e}")
        return None


def main():
    """Main function to populate database"""
    logger.info("Starting Upstox data population...")
    
    # Initialize tables
    initialize_tables()
    
    # Get instrument map from Upstox
    logger.info("Fetching instrument map from Upstox...")
    try:
        instrument_map = get_instrument_map()
        logger.info(f"Loaded {len(instrument_map)} instrument keys")
    except Exception as e:
        logger.error(f"Failed to fetch instrument map: {e}")
        logger.info("Continuing without instrument keys...")
        instrument_map = {}
    
    # Use a single connection for all operations to avoid pool exhaustion
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Insert stocks and prices
        for symbol, name, sector in STOCKS:
            logger.info(f"Processing {symbol}...")
            
            # Get instrument key from map
            instrument_key = instrument_map.get(symbol)
            if not instrument_key:
                logger.warning(f"No instrument key found for {symbol}")
            
            # Insert stock metadata
            try:
                query = """
                    INSERT INTO stocks (symbol, name, sector, instrument_key, market_cap, updated_at)
                    VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (symbol) DO UPDATE SET
                        name = EXCLUDED.name,
                        sector = EXCLUDED.sector,
                        instrument_key = COALESCE(EXCLUDED.instrument_key, stocks.instrument_key),
                        market_cap = EXCLUDED.market_cap,
                        updated_at = CURRENT_TIMESTAMP
                """
                cursor.execute(query, (symbol, name, sector, instrument_key, "N/A"))
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Error inserting stock {symbol}: {e}")
                continue
            
            # Fetch real data from Upstox if we have instrument key
            if instrument_key:
                stock_data = get_stock_data_from_upstox(instrument_key)
                if stock_data:
                    # Insert price
                    try:
                        query = """
                            INSERT INTO stock_prices (symbol, price, change, change_pct, volume, high_52w, low_52w)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """
                        cursor.execute(query, (
                            symbol,
                            stock_data['price'],
                            stock_data['change'],
                            stock_data['change_pct'],
                            stock_data['volume'],
                            stock_data['high_52w'],
                            stock_data['low_52w']
                        ))
                        conn.commit()
                        logger.info(
                            f"Inserted {symbol}: ₹{stock_data['price']} ({stock_data['change_pct']:+.2f}%)"
                        )
                    except Exception as e:
                        conn.rollback()
                        logger.error(f"Error inserting price for {symbol}: {e}")
                else:
                    logger.warning(f"Skipping {symbol} due to data fetch error")
            else:
                logger.warning(f"Skipping price data for {symbol} (no instrument key)")
        
        # Verify data
        logger.info("Verifying data...")
        cursor.execute("SELECT COUNT(*) FROM stocks")
        stock_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM stock_prices")
        price_count = cursor.fetchone()[0]
        
        logger.info(f"Total stocks in database: {stock_count}")
        logger.info(f"Total price records: {price_count}")
        
        # Show sample data
        query = """
            SELECT sp.symbol, s.name, s.sector, sp.price, sp.change, sp.change_pct
            FROM stock_prices sp
            JOIN stocks s ON sp.symbol = s.symbol
            WHERE sp.id IN (
                SELECT MAX(id) FROM stock_prices GROUP BY sp.symbol
            )
            LIMIT 5
        """
        cursor.execute(query)
        results = cursor.fetchall()
        if results:
            logger.info("\nSample data:")
            for row in results:
                logger.info(f"  {row[0]}: ₹{row[3]} ({row[5]:+.2f}%)")
        
        logger.info("Data population complete!")
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()
