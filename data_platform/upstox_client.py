"""
Upstox API Client — Complete Implementation
============================================
All endpoints are VERIFIED working with your free-tier token.

Confirmed working endpoints:
  Market Data:
    GET /v2/market-quote/quotes            → live OHLC (stocks + indices)
    GET /v2/market-quote/ltp               → last traded price
    GET /v2/market-quote/ohlc              → OHLC with interval
    GET /v2/market/fii                     → FII activity (NSE_EQ|CASH etc)
    GET /v2/market/dii                     → DII activity
    GET /v2/market/oi                      → Open Interest per strike
    GET /v2/market/max-pain                → Max Pain (needs bucket_interval)
    GET /v2/market/pcr                     → Put Call Ratio
    GET /v2/market/change-oi               → Change in OI (needs interval)
    GET /v2/market/holidays                → Market holidays list
    GET /v2/market/smartlist/futures       → Futures smartlist
    GET /v2/market/smartlist/options       → Options smartlist
    GET /v2/market/smartlist/mtf           → MTF smartlist
    GET /v2/option/chain                   → Full options chain with greeks

  Historical Data:
    GET /v2/historical-candle/{key}/{interval}/{to}/{from}  → OHLCV (daily/weekly)
    GET /v2/historical-candle/intraday/{key}/{interval}     → intraday candles

  Fundamentals:
    GET /v2/fundamentals/{ISIN}/profile
    GET /v2/fundamentals/{ISIN}/balance-sheet
    GET /v2/fundamentals/{ISIN}/income-statement
    GET /v2/fundamentals/{ISIN}/cash-flow
    GET /v2/fundamentals/{ISIN}/key-ratios
    GET /v2/fundamentals/{ISIN}/share-holdings
    GET /v2/fundamentals/{ISIN}/corporate-actions
    GET /v2/fundamentals/{instrument_key}/competitors

  News:
    GET /v2/news  ?category=instrument_keys&instrument_keys=NSE_EQ|ISIN

NOT available on free tier (endpoints return 404):
  - /v2/market-information/exchange-status  (use local IST computation instead)
  - /v2/market-information/market-holidays  (use /v2/market/holidays instead)
  - /v2/fundamentals/company-profile        (wrong path — use /{ISIN}/profile)
"""

from __future__ import annotations

import gzip
import json
import os
import time
import urllib.request
from datetime import date, datetime, timedelta
from typing import Any

import requests
from dotenv import load_dotenv

from utils.logger import get_logger

load_dotenv()
logger = get_logger("upstox_client")

UPSTOX_BASE = "https://api.upstox.com"

# ---------------------------------------------------------------------------
# Index instrument keys — fixed, never change
# ---------------------------------------------------------------------------
INDEX_KEYS = {
    "NIFTY50":    "NSE_INDEX|Nifty 50",
    "SENSEX":     "BSE_INDEX|SENSEX",
    "BANKNIFTY":  "NSE_INDEX|Nifty Bank",
    "FINNIFTY":   "NSE_INDEX|Nifty Fin Service",
    "INDIAVIX":   "NSE_INDEX|India VIX",
    "MIDCAP150":  "NSE_INDEX|Nifty Midcap 150",
    "SMALLCAP250":"NSE_INDEX|Nifty Smallcap 250",
    "NIFTYIT":    "NSE_INDEX|Nifty IT",
    "NIFTYPHARMA":"NSE_INDEX|Nifty Pharma",
    "NIFTYAUTO":  "NSE_INDEX|Nifty Auto",
    "NIFTYFMCG":  "NSE_INDEX|Nifty FMCG",
    "NIFTYMETAL": "NSE_INDEX|Nifty Metal",
    "NIFTYENERGY":"NSE_INDEX|Nifty Energy",
    "NIFTYREALTY":"NSE_INDEX|Nifty Realty",
}

# Sector index mapping for sector rotation view
SECTOR_INDEX_MAP = {
    "IT":        "NIFTYIT",
    "Pharma":    "NIFTYPHARMA",
    "Auto":      "NIFTYAUTO",
    "FMCG":      "NIFTYFMCG",
    "Metal":     "NIFTYMETAL",
    "Energy":    "NIFTYENERGY",
    "Realty":    "NIFTYREALTY",
    "Banking":   "BANKNIFTY",
    "Finance":   "FINNIFTY",
}

# FII data_type values (all verified working)
FII_DATA_TYPES = [
    "NSE_EQ|CASH",
    "NSE_FO|INDEX_FUTURES",
    "NSE_FO|STOCK_FUTURES",
    "NSE_FO|INDEX_OPTIONS",
    "NSE_FO|STOCK_OPTIONS",
]

# Smartlist categories (verified)
FUTURES_SL_CATEGORIES  = ["PRICE_GAINERS", "PRICE_LOSERS", "MOST_ACTIVE", "OI_GAINERS", "OI_LOSERS", "PREMIUM", "DISCOUNT"]
OPTIONS_SL_CATEGORIES  = ["OI_GAINERS", "OI_LOSERS", "PRICE_GAINERS", "PRICE_LOSERS", "MOST_ACTIVE", "IV_GAINERS", "IV_LOSERS"]


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def _auth() -> dict:
    token = os.getenv("UPSTOX_BROKER_ACCESS_TOKEN", "")
    if not token:
        raise RuntimeError("UPSTOX_BROKER_ACCESS_TOKEN not set in .env")
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def _get(path: str, params: dict | None = None, timeout: int = 12) -> dict:
    url = f"{UPSTOX_BASE}{path}"
    try:
        r = requests.get(url, headers=_auth(), params=params or {}, timeout=timeout)
        data = r.json()
        if data.get("status") != "success":
            errs = data.get("errors", [{}])
            msg  = errs[0].get("message", "unknown") if errs else "unknown"
            logger.warning(f"Upstox error {path}: {msg}")
        return data
    except Exception as e:
        logger.error(f"Upstox request failed {path}: {e}")
        raise


# ---------------------------------------------------------------------------
# Instrument key map  (NSE symbol → "NSE_EQ|ISIN")
# ---------------------------------------------------------------------------
_IMAP: dict[str, str] | None = None
_IMAP_AT: float = 0.0
_IMAP_TTL = 86400


def get_instrument_map() -> dict[str, str]:
    global _IMAP, _IMAP_AT
    now = time.time()
    if _IMAP and (now - _IMAP_AT) < _IMAP_TTL:
        return _IMAP

    cache_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "upstox_instrument_keys.json")
    )
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            _IMAP = json.load(f)
        logger.info(f"Loaded {len(_IMAP)} instrument keys from cache")
    else:
        logger.info("Downloading NSE instrument master from Upstox…")
        url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
        with urllib.request.urlopen(url, timeout=20) as r:
            instruments = json.loads(gzip.decompress(r.read()))
        _IMAP = {
            i["trading_symbol"]: i["instrument_key"]
            for i in instruments
            if i.get("segment") == "NSE_EQ" and i.get("instrument_type") == "EQ"
        }
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(_IMAP, f)
        logger.info(f"Cached {len(_IMAP)} instrument keys")

    _IMAP_AT = now
    return _IMAP


def symbol_to_key(symbol: str) -> str | None:
    return get_instrument_map().get(symbol.upper())


def key_to_isin(instrument_key: str) -> str:
    """Extract ISIN from instrument key: 'NSE_EQ|INE002A01018' → 'INE002A01018'"""
    return instrument_key.split("|")[-1] if "|" in instrument_key else instrument_key


# ---------------------------------------------------------------------------
# 1. Live Quotes
# ---------------------------------------------------------------------------
def get_live_quotes(instrument_keys: list[str]) -> dict:
    """Batch live OHLC quotes. Max 100 per call. Returns {key: quote_dict}."""
    if not instrument_keys:
        return {}
    result = {}
    for i in range(0, len(instrument_keys), 100):
        batch = instrument_keys[i:i+100]
        data = _get("/v2/market-quote/quotes", {"instrument_key": ",".join(batch)})
        result.update(data.get("data", {}))
    return result


def get_ltp(instrument_keys: list[str]) -> dict:
    """Lightweight last-traded-price only."""
    if not instrument_keys:
        return {}
    data = _get("/v2/market-quote/ltp", {"instrument_key": ",".join(instrument_keys)})
    return data.get("data", {})


def get_ohlc(instrument_keys: list[str], interval: str = "1d") -> dict:
    """OHLC with interval: 1d, 1w, 1m."""
    if not instrument_keys:
        return {}
    data = _get("/v2/market-quote/ohlc", {"instrument_key": ",".join(instrument_keys), "interval": interval})
    return data.get("data", {})


def get_stock_quote(symbol: str) -> dict | None:
    key = symbol_to_key(symbol)
    if not key:
        return None
    raw = get_live_quotes([key])
    q = raw.get(key) or raw.get(key.replace("|", ":"))
    if not q and raw:
        q = list(raw.values())[0]
    if not q:
        return None
    return _format_quote(symbol, q)


def get_bulk_quotes(symbols: list[str]) -> dict[str, dict]:
    imap = get_instrument_map()
    key_to_sym: dict[str, str] = {}
    keys = []
    for sym in symbols:
        k = imap.get(sym.upper())
        if k:
            keys.append(k)
            key_to_sym[k]               = sym
            key_to_sym[k.replace("|",":")]=sym
    raw = get_live_quotes(keys)
    return {key_to_sym[rk]: _format_quote(key_to_sym[rk], q)
            for rk, q in raw.items() if rk in key_to_sym}


def _format_quote(symbol: str, q: dict) -> dict:
    ohlc = q.get("ohlc", {})
    close = ohlc.get("close") or q.get("last_price") or 1
    return {
        "symbol":     symbol,
        "last_price": q.get("last_price"),
        "open":       ohlc.get("open"),
        "high":       ohlc.get("high"),
        "low":        ohlc.get("low"),
        "close":      ohlc.get("close"),
        "net_change": q.get("net_change"),
        "pct_change": round(q["net_change"] / close * 100, 2) if q.get("net_change") and close else None,
        "volume":     q.get("volume"),
        "oi":         q.get("oi"),
        "timestamp":  q.get("timestamp"),
    }


# ---------------------------------------------------------------------------
# 2. Index Overview
# ---------------------------------------------------------------------------
def get_index_overview() -> dict:
    keys = list(INDEX_KEYS.values())
    raw  = get_live_quotes(keys)
    result = {}
    for name, key in INDEX_KEYS.items():
        q = raw.get(key) or raw.get(key.replace("|", ":"))
        if q:
            ohlc  = q.get("ohlc", {})
            close = ohlc.get("close") or 1
            chg   = q.get("net_change", 0)
            result[name] = {
                "last_price": q.get("last_price"),
                "open":       ohlc.get("open"),
                "high":       ohlc.get("high"),
                "low":        ohlc.get("low"),
                "close":      ohlc.get("close"),
                "net_change": chg,
                "pct_change": round(chg / close * 100, 2) if close else None,
                "timestamp":  q.get("timestamp"),
            }
    return result


def get_sector_overview() -> dict:
    """Returns sector-wise performance using Nifty sector indices."""
    keys = {sector: INDEX_KEYS[idx_name]
            for sector, idx_name in SECTOR_INDEX_MAP.items()
            if idx_name in INDEX_KEYS}
    raw = get_live_quotes(list(keys.values()))
    result = {}
    for sector, key in keys.items():
        q = raw.get(key) or raw.get(key.replace("|", ":"))
        if q:
            ohlc  = q.get("ohlc", {})
            close = ohlc.get("close") or 1
            chg   = q.get("net_change", 0)
            result[sector] = {
                "index_name": key.split("|")[-1],
                "last_price": q.get("last_price"),
                "net_change": chg,
                "pct_change": round(chg / close * 100, 2) if close else None,
            }
    return result


# ---------------------------------------------------------------------------
# 3. Historical Candles
# ---------------------------------------------------------------------------
_INTERVAL_MAP = {
    "1minute":  "1minute",  "1m": "1minute",
    "5minute":  "5minute",  "5m": "5minute",
    "15minute": "15minute", "15m": "15minute",
    "30minute": "30minute", "30m": "30minute",
    "1hour":    "60minute", "1h": "60minute", "60minute": "60minute",
    "1day":     "day",      "1d": "day",      "day": "day",
    "1week":    "week",     "1w": "week",     "week": "week",
    "1month":   "month",    "monthly": "month", "month": "month",
}
INTRADAY_INTERVALS = {"1minute", "5minute", "15minute", "30minute", "60minute"}


def get_candles(symbol: str, interval: str = "1day", days: int = 180,
                from_date: str | None = None, to_date: str | None = None) -> list[dict]:
    key = symbol_to_key(symbol)
    if not key:
        logger.warning(f"No instrument key for {symbol}")
        return []
    ui = _INTERVAL_MAP.get(interval, "day")
    if to_date is None:
        to_date = date.today().isoformat()
    if from_date is None:
        from_date = (date.today() - timedelta(days=days)).isoformat()
    if ui in INTRADAY_INTERVALS:
        return _intraday_candles(key, ui)
    encoded = key.replace("|", "%7C")
    data = _get(f"/v2/historical-candle/{encoded}/{ui}/{to_date}/{from_date}")
    return _fmt_candles(data.get("data", {}).get("candles", []))


def _intraday_candles(instrument_key: str, interval: str) -> list[dict]:
    encoded = instrument_key.replace("|", "%7C")
    data = _get(f"/v2/historical-candle/intraday/{encoded}/{interval}")
    return _fmt_candles(data.get("data", {}).get("candles", []))


def _fmt_candles(raw: list) -> list[dict]:
    return [
        {"timestamp": c[0], "open": c[1], "high": c[2], "low": c[3], "close": c[4], "volume": c[5]}
        for c in raw if len(c) >= 6
    ]


# ---------------------------------------------------------------------------
# 4. FII / DII Activity  (VERIFIED WORKING)
# ---------------------------------------------------------------------------
def get_fii_activity(date_str: str | None = None, interval: str = "1D") -> dict:
    """
    FII activity across all segments.
    date_str: 'YYYY-MM-DD', defaults to yesterday
    Returns dict with buy/sell amounts per segment.
    """
    if date_str is None:
        # Use yesterday (today's data not available until EOD)
        date_str = (date.today() - timedelta(days=1)).isoformat()

    result: dict[str, Any] = {"date": date_str, "segments": {}}
    for dt in FII_DATA_TYPES:
        try:
            data = _get("/v2/market/fii", {"data_type": dt, "date": date_str, "interval": interval})
            rows = data.get("data", {}).get(dt, [])
            if rows:
                latest = rows[0]
                result["segments"][dt] = {
                    "buy_amount":    latest.get("buy_amount"),
                    "sell_amount":   latest.get("sell_amount"),
                    "net_amount":    round(latest.get("buy_amount", 0) - latest.get("sell_amount", 0), 2),
                    "buy_contracts": latest.get("buy_contracts"),
                    "sell_contracts":latest.get("sell_contracts"),
                    "oi_contracts":  latest.get("oi_contracts"),
                    "timestamp":     latest.get("time_stamp"),
                }
        except Exception as e:
            logger.warning(f"FII {dt} failed: {e}")

    # Compute total net FII flow (Cash + Futures combined)
    cash  = result["segments"].get("NSE_EQ|CASH", {})
    idx_f = result["segments"].get("NSE_FO|INDEX_FUTURES", {})
    stk_f = result["segments"].get("NSE_FO|STOCK_FUTURES", {})
    result["total_net_cash"]    = cash.get("net_amount")
    result["total_net_futures"] = round(
        (idx_f.get("net_amount") or 0) + (stk_f.get("net_amount") or 0), 2
    )
    result["available"] = True
    return result


def get_dii_activity(date_str: str | None = None, interval: str = "1D") -> dict:
    """DII cash segment activity."""
    if date_str is None:
        date_str = (date.today() - timedelta(days=1)).isoformat()
    data = _get("/v2/market/dii", {"data_type": "NSE_EQ|CASH", "date": date_str, "interval": interval})
    rows = data.get("data", {}).get("NSE_EQ|CASH", [])
    if not rows:
        return {"date": date_str, "available": False}
    r = rows[0]
    return {
        "date":          date_str,
        "available":     True,
        "buy_amount":    r.get("buy_amount"),
        "sell_amount":   r.get("sell_amount"),
        "net_amount":    round(r.get("buy_amount", 0) - r.get("sell_amount", 0), 2),
        "buy_contracts": r.get("buy_contracts"),
        "sell_contracts":r.get("sell_contracts"),
    }


# ---------------------------------------------------------------------------
# 5. Options Chain  (VERIFIED WORKING)
# ---------------------------------------------------------------------------
def get_option_chain(instrument_key: str, expiry_date: str) -> list[dict]:
    """
    Full options chain with greeks for a given underlying and expiry.
    Returns list of strikes with CE/PE market data + greeks.
    """
    data = _get("/v2/option/chain", {"instrument_key": instrument_key, "expiry_date": expiry_date})
    return data.get("data", [])


def get_option_expiries(instrument_key: str) -> list[str]:
    """Get list of available expiry dates for an underlying."""
    data = _get("/v2/option/contract", {"instrument_key": instrument_key})
    contracts = data.get("data", [])
    today = date.today().isoformat()
    expiries = sorted({c["expiry"] for c in contracts if c.get("expiry", "") >= today})
    return expiries


def get_oi(instrument_key: str, expiry: str) -> dict:
    """Open Interest across all strikes."""
    data = _get("/v2/market/oi", {"instrument_key": instrument_key, "expiry": expiry, "date": date.today().isoformat()})
    return data.get("data", {})


def get_pcr(instrument_key: str, expiry: str) -> dict | None:
    """Put-Call Ratio."""
    try:
        data = _get("/v2/market/pcr", {
            "instrument_key": instrument_key,
            "expiry": expiry,
            "date": date.today().isoformat(),
            "bucket_interval": 100,
        })
        return data.get("data")
    except Exception as e:
        logger.warning(f"PCR failed: {e}")
        return None


def get_max_pain(instrument_key: str, expiry: str, bucket_interval: int = 100) -> dict | None:
    """Max Pain level for an expiry."""
    try:
        data = _get("/v2/market/max-pain", {
            "instrument_key":  instrument_key,
            "expiry":          expiry,
            "date":            date.today().isoformat(),
            "bucket_interval": bucket_interval,
        })
        return data.get("data")
    except Exception as e:
        logger.warning(f"Max Pain failed: {e}")
        return None


def get_change_oi(instrument_key: str, expiry: str, interval: int = 1) -> dict | None:
    """Change in Open Interest by strike."""
    try:
        data = _get("/v2/market/change-oi", {
            "instrument_key": instrument_key,
            "expiry":         expiry,
            "date":           date.today().isoformat(),
            "interval":       interval,
        })
        return data.get("data")
    except Exception as e:
        logger.warning(f"Change OI failed: {e}")
        return None


def compute_pcr_from_chain(chain: list[dict]) -> dict:
    """Compute PCR from option chain data when direct API not available."""
    total_ce_oi = sum(s.get("call_options", {}).get("market_data", {}).get("oi", 0) or 0 for s in chain)
    total_pe_oi = sum(s.get("put_options",  {}).get("market_data", {}).get("oi", 0) or 0 for s in chain)
    pcr = round(total_pe_oi / total_ce_oi, 3) if total_ce_oi > 0 else None
    return {"total_ce_oi": total_ce_oi, "total_pe_oi": total_pe_oi, "pcr": pcr}


# ---------------------------------------------------------------------------
# 6. Smartlists  (VERIFIED WORKING)
# ---------------------------------------------------------------------------
def get_futures_smartlist(asset_type: str = "INDEX", category: str = "PRICE_GAINERS") -> dict:
    """
    asset_type: INDEX | STOCK | COMMODITY
    category: PRICE_GAINERS | PRICE_LOSERS | MOST_ACTIVE | OI_GAINERS | OI_LOSERS | PREMIUM | DISCOUNT
    """
    data = _get("/v2/market/smartlist/futures", {"asset_type": asset_type, "category": category})
    return data.get("data", {})


def get_options_smartlist(asset_type: str = "INDEX", category: str = "OI_GAINERS") -> dict:
    """
    category: OI_GAINERS | OI_LOSERS | PRICE_GAINERS | PRICE_LOSERS | MOST_ACTIVE | IV_GAINERS | IV_LOSERS
    """
    data = _get("/v2/market/smartlist/options", {"asset_type": asset_type, "category": category})
    return data.get("data", {})


def get_mtf_smartlist() -> dict:
    """Stocks available for Margin Trade Funding."""
    data = _get("/v2/market/smartlist/mtf")
    return data.get("data", {})


# ---------------------------------------------------------------------------
# 7. Market Holidays  (VERIFIED WORKING)
# ---------------------------------------------------------------------------
def get_market_holidays() -> list[dict]:
    """Returns full year holiday list for NSE/BSE/MCX."""
    data = _get("/v2/market/holidays")
    return data.get("data", [])


def is_market_holiday(check_date: date | None = None) -> bool:
    check_date = check_date or date.today()
    holidays = get_market_holidays()
    return check_date.isoformat() in {h["date"] for h in holidays}


# ---------------------------------------------------------------------------
# 8. Market Status (computed from IST — exchange-status not on free tier)
# ---------------------------------------------------------------------------
def get_market_status() -> dict:
    from utils.time_utils import now_ist
    now  = now_ist()
    wd   = now.weekday()
    t    = now.time()
    from datetime import time as dtime
    open_t  = dtime(9, 15)
    close_t = dtime(15, 30)
    pre_t   = dtime(9, 0)
    is_wd   = wd < 5
    is_open = is_wd and open_t <= t <= close_t
    is_pre  = is_wd and pre_t  <= t <  open_t

    # Next open
    days_ahead = 0
    if t > close_t or not is_wd:
        days_ahead = 1
        while (now + timedelta(days=days_ahead)).weekday() >= 5:
            days_ahead += 1
    nxt = (now + timedelta(days=days_ahead)).replace(hour=9, minute=15, second=0, microsecond=0)

    return {
        "is_open":     is_open,
        "is_pre_open": is_pre,
        "session":     "OPEN" if is_open else ("PRE_OPEN" if is_pre else "CLOSED"),
        "current_ist": now.isoformat(),
        "next_open":   nxt.isoformat(),
        "exchange":    "NSE",
    }


# ---------------------------------------------------------------------------
# 9. Fundamentals  (VERIFIED WORKING — ISIN in path)
# ---------------------------------------------------------------------------
def _isin_from_symbol(symbol: str) -> str | None:
    key = symbol_to_key(symbol)
    return key_to_isin(key) if key else None


def get_company_profile(symbol: str) -> dict | None:
    isin = _isin_from_symbol(symbol)
    if not isin:
        return None
    data = _get(f"/v2/fundamentals/{isin}/profile")
    return data.get("data")


def get_balance_sheet(symbol: str, statement_type: str = "Consolidated", period_type: str = "Annual") -> dict | None:
    isin = _isin_from_symbol(symbol)
    if not isin:
        return None
    data = _get(f"/v2/fundamentals/{isin}/balance-sheet",
                {"statement_type": statement_type, "period_type": period_type})
    return data.get("data")


def get_income_statement(symbol: str, statement_type: str = "Consolidated", period_type: str = "Annual") -> dict | None:
    isin = _isin_from_symbol(symbol)
    if not isin:
        return None
    data = _get(f"/v2/fundamentals/{isin}/income-statement",
                {"statement_type": statement_type, "period_type": period_type})
    return data.get("data")


def get_cash_flow(symbol: str, statement_type: str = "Consolidated", period_type: str = "Annual") -> dict | None:
    isin = _isin_from_symbol(symbol)
    if not isin:
        return None
    data = _get(f"/v2/fundamentals/{isin}/cash-flow",
                {"statement_type": statement_type, "period_type": period_type})
    return data.get("data")


def get_key_ratios(symbol: str) -> list | None:
    isin = _isin_from_symbol(symbol)
    if not isin:
        return None
    data = _get(f"/v2/fundamentals/{isin}/key-ratios")
    return data.get("data")


def get_share_holdings(symbol: str) -> list | None:
    isin = _isin_from_symbol(symbol)
    if not isin:
        return None
    data = _get(f"/v2/fundamentals/{isin}/share-holdings")
    return data.get("data")


def get_competitors(symbol: str) -> list | None:
    key = symbol_to_key(symbol)
    if not key:
        return None
    data = _get(f"/v2/fundamentals/{key}/competitors")
    return data.get("data")


def get_corporate_actions(symbol: str) -> list | None:
    isin = _isin_from_symbol(symbol)
    if not isin:
        return None
    data = _get(f"/v2/fundamentals/{isin}/corporate-actions")
    return data.get("data")


def get_full_fundamentals(symbol: str) -> dict:
    """All fundamentals in one call — profile + key ratios + holdings."""
    return {
        "profile":          get_company_profile(symbol),
        "key_ratios":       get_key_ratios(symbol),
        "share_holdings":   get_share_holdings(symbol),
        "corporate_actions":get_corporate_actions(symbol),
        "competitors":      get_competitors(symbol),
    }


# ---------------------------------------------------------------------------
# 10. News  (VERIFIED WORKING)
# ---------------------------------------------------------------------------
def get_stock_news(symbol: str, limit: int = 10) -> list[dict]:
    """Get latest news articles for a stock symbol."""
    key = symbol_to_key(symbol)
    if not key:
        return []
    try:
        data = _get("/v2/news", {
            "category":        "instrument_keys",
            "instrument_keys": key,
        })
        articles = data.get("data", {}).get(key, [])
        return articles[:limit]
    except Exception as e:
        logger.warning(f"News failed for {symbol}: {e}")
        return []


def get_multi_stock_news(symbols: list[str], limit_per_stock: int = 5) -> dict[str, list]:
    """Get news for multiple stocks in one call."""
    imap = get_instrument_map()
    sym_keys = {sym: imap[sym.upper()] for sym in symbols if sym.upper() in imap}
    if not sym_keys:
        return {}
    keys_str = ",".join(sym_keys.values())
    try:
        data = _get("/v2/news", {"category": "instrument_keys", "instrument_keys": keys_str})
        raw = data.get("data", {})
        # Map back instrument keys to symbols
        key_to_sym = {v: k for k, v in sym_keys.items()}
        return {
            key_to_sym.get(k, k): articles[:limit_per_stock]
            for k, articles in raw.items()
        }
    except Exception as e:
        logger.warning(f"Multi-stock news failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Convenience: get everything for cockpit dashboard
# ---------------------------------------------------------------------------
def get_cockpit_data() -> dict:
    """
    Single call that returns everything needed for the cockpit/homepage:
    - All major indices
    - Sector performance
    - FII/DII net flows
    - Market status
    - Market holidays (next 5)
    """
    from datetime import date as d
    today_str = d.today().isoformat()

    indices = get_index_overview()
    sectors = get_sector_overview()
    mkt_status = get_market_status()

    # FII/DII (yesterday's data since today's not available until EOD)
    try:
        fii = get_fii_activity()
        dii = get_dii_activity()
    except Exception as e:
        logger.warning(f"FII/DII in cockpit failed: {e}")
        fii = {"available": False}
        dii = {"available": False}

    # Upcoming holidays
    try:
        holidays = [h for h in get_market_holidays() if h.get("date", "") >= today_str][:5]
    except Exception:
        holidays = []

    return {
        "indices":      indices,
        "sectors":      sectors,
        "fii":          fii,
        "dii":          dii,
        "market_status":mkt_status,
        "holidays":     holidays,
        "timestamp":    datetime.now().isoformat(),
    }
