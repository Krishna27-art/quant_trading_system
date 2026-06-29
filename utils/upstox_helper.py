import gzip
import json
import os
import sys
import time
from pathlib import Path

import requests

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.logger import get_logger

logger = get_logger("upstox_helper")

INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"
CACHE_PATH = Path(__file__).parent.parent / "data" / "upstox_instruments_cache.json"


def download_and_cache_instruments() -> dict[str, dict]:
    """Downloads the complete instrument master list from Upstox and builds a symbol map."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Check if cache is fresh (less than 24 hours old)
    if CACHE_PATH.exists():
        mtime = CACHE_PATH.stat().st_mtime
        if time.time() - mtime < 86400:
            logger.info("Loading Upstox instrument keys from local cache")
            with open(CACHE_PATH) as f:
                return json.load(f)

    logger.info(f"Downloading Upstox instrument master from {INSTRUMENTS_URL}...")
    try:
        resp = requests.get(INSTRUMENTS_URL, stream=True, timeout=30)
        resp.raise_for_status()

        # Decompress gzip in memory
        with gzip.GzipFile(fileobj=resp.raw) as gzf:
            instruments_list = json.loads(gzf.read().decode("utf-8"))

        symbol_map = {}
        for inst in instruments_list:
            # We care about NSE Equities (NSE_EQ) and F&O (NSE_FO)
            # Instrument key format example: NSE_EQ|INE002A01018
            segment = inst.get("segment")
            trading_symbol = inst.get("trading_symbol")
            instrument_key = inst.get("instrument_key")

            if not trading_symbol or not instrument_key:
                continue

            # Store equity mapping: e.g. "RELIANCE" -> key
            if segment == "NSE_EQ":
                symbol_map[trading_symbol] = {
                    "instrument_key": instrument_key,
                    "isin": inst.get("isin"),
                    "name": inst.get("name"),
                    "segment": segment,
                }
            elif segment == "NSE_FO" and trading_symbol.startswith(("NIFTY", "BANKNIFTY")):
                # Index options/futures
                symbol_map[trading_symbol] = {
                    "instrument_key": instrument_key,
                    "isin": inst.get("isin"),
                    "name": inst.get("name"),
                    "segment": segment,
                }

        with open(CACHE_PATH, "w") as f:
            json.dump(symbol_map, f)

        logger.info(f"Cached {len(symbol_map)} NSE_EQ and F&O instruments to disk.")
        return symbol_map
    except Exception as e:
        logger.error(f"Failed to download or parse Upstox instruments: {e}")
        if CACHE_PATH.exists():
            logger.warning("Falling back to expired local instruments cache")
            with open(CACHE_PATH) as f:
                return json.load(f)
        return {}


def get_instrument_key(symbol: str) -> str | None:
    """Resolves a trading symbol like RELIANCE to an Upstox instrument key like NSE_EQ|INE002A01018."""
    symbol = symbol.upper()
    if symbol.endswith(".NS"):
        symbol = symbol[:-3]

    symbol_map = download_and_cache_instruments()
    info = symbol_map.get(symbol)
    return info["instrument_key"] if info else None


def get_upstox_client_config():
    """Builds and returns the upstox_client configuration with access token."""
    from dotenv import load_dotenv

    load_dotenv()

    import upstox_client

    token = os.getenv("UPSTOX_ACCESS_TOKEN")
    if not token:
        logger.error("UPSTOX_ACCESS_TOKEN is missing in .env")
        return None

    config = upstox_client.Configuration()
    config.access_token = token
    return config
