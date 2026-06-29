"""
Real Data Ingestion Script from NSELib

Fetches real stock data from NSE and populates the database.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta

from nselib import capital_market

from database.connection import create_tables, insert_stock, insert_stock_price
from utils.logger import get_logger

logger = get_logger("ingest_nse_data")

# Top NSE stocks to fetch
TOP_STOCKS = [
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "ICICIBANK",
    "INFY",
    "SBIN",
    "BHARTIARTL",
    "ITC",
    "LT",
    "MARUTI",
    "SUNPHARMA",
    "TATAMOTORS",
    "AXISBANK",
    "DRREDDY",
    "WIPRO",
    "HCLTECH",
    "KOTAKBANK",
    "BAJFINANCE",
    "TITAN",
    "NTPC",
]


def fetch_stock_data(symbol):
    """Fetch stock data from NSELib."""
    try:
        logger.info(f"Fetching data for {symbol}")

        # Get latest price data using price_volume_data with date parameters
        from_date = (datetime.now() - timedelta(days=5)).strftime("%d-%m-%Y")
        to_date = datetime.now().strftime("%d-%m-%Y")

        price_data = capital_market.price_volume_data(
            symbol=symbol, from_date=from_date, to_date=to_date
        )

        if price_data is None or price_data.empty:
            logger.warning(f"No price data for {symbol}")
            return None

        # Get latest row
        latest = price_data.iloc[-1]

        # Debug: print columns
        logger.info(f"Columns in price_data: {price_data.columns.tolist()}")

        # Map columns based on actual NSELib structure
        return {
            "symbol": symbol,
            "name": symbol,  # Use symbol as name for now
            "sector": "Unknown",  # Default sector
            "price": float(latest.get("closePrice", latest.get("Close", 0))),
            "change": float(latest.get("change", latest.get("Change", 0))),
            "change_pct": float(latest.get("pChange", latest.get("% Change", 0))),
            "volume": int(latest.get("totalTradedQuantity", latest.get("Volume", 0))),
            "high_52w": float(latest.get("high52", latest.get("High52", 0))),
            "low_52w": float(latest.get("low52", latest.get("Low52", 0))),
        }

    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return None


def ingest_data():
    """Ingest real data from NSE to database."""
    logger.info("Starting NSE data ingestion")

    # Create tables
    create_tables()

    # Test with just one stock first
    test_symbol = "RELIANCE"
    logger.info(f"Testing with {test_symbol}")

    try:
        # Get latest price data using price_volume_data
        price_data = capital_market.price_volume_data(symbol=test_symbol)

        if price_data is not None and not price_data.empty:
            logger.info(f"Price data columns: {price_data.columns.tolist()}")
            logger.info(f"First row: {price_data.iloc[0].to_dict()}")
            logger.info(f"Last row: {price_data.iloc[-1].to_dict()}")
        else:
            logger.warning("No price data returned")
    except Exception as e:
        logger.error(f"Error testing NSELib: {e}")
        import traceback

        logger.error(traceback.format_exc())

    success_count = 0
    for symbol in TOP_STOCKS[:3]:  # Test with just 3 stocks first
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
            logger.info(f"Successfully ingested {symbol}")

        except Exception as e:
            logger.error(f"Error ingesting {symbol}: {e}")

    logger.info(f"Ingestion complete. Successfully ingested {success_count}/3 stocks")


if __name__ == "__main__":
    ingest_data()
