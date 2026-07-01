"""
FastAPI Backend — Quant Terminal

Endpoints:
  GET  /                          health ping
  GET  /api/health                liveness
  GET  /api/health/status         component health
  GET  /api/indices               NSE index ticks
  GET  /api/stocks                all stocks (sector/search filter)
  GET  /api/stocks/{symbol}       single stock
  GET  /api/predictions           ML predictions (filter by result)
  GET  /api/calibration           win-rate bucketed by confidence
  GET  /api/ticker                top-10 movers for ticker bar
  GET  /api/sectors               sector heatmap
  GET  /api/metrics/performance   portfolio performance metrics
  GET  /api/metrics/model         model IC / accuracy metrics
  GET  /api/oms/orders            open orders (read-only)
  GET  /api/news/{symbol}         news headlines
  POST /api/auth/login            dev-only token (disabled in LIVE env)
  GET  /metrics                   Prometheus scrape endpoint
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from pydantic import BaseModel
import yfinance as yf

load_dotenv()

from api.auth import verify_token, router as auth_router
from database.connection import (
    create_tables,
    get_latest_prices,
    get_predictions,
    get_sector_performance,
    get_stock_price,
    initialize_pool,
)
from utils.logger import get_logger
from utils.time_utils import now_ist

logger = get_logger("api")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Quant Terminal API",
    description="Backend API for Institutional Quant Research OS",
    version="1.0.0",
)

app.include_router(auth_router, prefix="/api/auth")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://[::]:3000",
        "http://[::1]:3000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.mount("/metrics", make_asgi_app())

# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    env = os.getenv("ENV", "LOCAL")
    if env in ("LIVE", "PAPER") and "api.mock_data" in sys.modules:
        logger.critical("FATAL: mock_data module loaded in production environment!")
        sys.exit(1)
    try:
        initialize_pool()
        create_tables()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database init failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    from database.connection import close_all_connections
    close_all_connections()
    logger.info("Database connections closed")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class IndexData(BaseModel):
    id: str
    name: str
    value: float
    change: float


class StockData(BaseModel):
    symbol: str
    name: str
    price: float
    change: float
    change_pct: float
    volume: int
    market_cap: str
    sector: str
    signal: str
    high_52w: float | None = None
    low_52w: float | None = None
    timestamp: str | None = None


class PredictionData(BaseModel):
    date: str
    symbol: str
    prediction: str
    horizon: str
    confidence: float
    entry_price: float | None = None
    stop_loss: float | None = None
    target_price: float | None = None
    actual: str | None = None
    result: str | None = None
    reason: str | None = None


class HealthStatus(BaseModel):
    name: str
    status: str
    value: str
    message: str


class MetricData(BaseModel):
    key: str
    value: str
    color: str


class TickerItem(BaseModel):
    name: str
    value: float
    change: float
    up: bool


class SectorItem(BaseModel):
    name: str
    change: float
    top_stock: str | None = None
    volume: int | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float(v, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _to_int(v, default: int = 0) -> int:
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _latest_signal_map() -> dict[str, str]:
    """Map symbols to the latest stored model signal, if one exists."""
    try:
        rows = get_predictions(limit=5000)
    except Exception:
        return {}

    signals: dict[str, str] = {}
    for row in rows:
        symbol = str(row.get("symbol") or "").upper()
        prediction = str(row.get("prediction") or "").upper()
        if symbol and symbol not in signals and prediction in {"BUY", "SELL", "HOLD"}:
            signals[symbol] = prediction
    return signals


def _stock_from_row(p: dict, signal_map: dict[str, str] | None = None) -> StockData:
    symbol = str(p["symbol"]).upper()
    ts = p.get("timestamp")
    return StockData(
        symbol=symbol,
        name=p.get("name") or symbol,
        price=_to_float(p.get("price")),
        change=_to_float(p.get("change")),
        change_pct=_to_float(p.get("change_pct")),
        volume=_to_int(p.get("volume")),
        market_cap=str(p.get("market_cap") or ""),
        sector=str(p.get("sector") or "Unknown"),
        signal=(signal_map or {}).get(symbol, str(p.get("signal") or "HOLD")),
        high_52w=_to_float(p.get("high_52w")) or None,
        low_52w=_to_float(p.get("low_52w")) or None,
        timestamp=ts.isoformat() if hasattr(ts, "isoformat") else (str(ts) if ts else None),
    )


def _ema(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    alpha = 2 / (window + 1)
    out = [values[0]]
    for value in values[1:]:
        out.append((value * alpha) + (out[-1] * (1 - alpha)))
    return out


def _rsi(values: list[float], window: int = 14) -> float | None:
    if len(values) <= window:
        return None
    gains: list[float] = []
    losses: list[float] = []
    for prev, cur in zip(values[-window - 1:-1], values[-window:], strict=False):
        diff = cur - prev
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    avg_loss = sum(losses) / window
    if avg_loss == 0:
        return 100.0
    rs = (sum(gains) / window) / avg_loss
    return 100 - (100 / (1 + rs))


def _derive_history_payload(hist) -> dict:
    closes = [round(float(v), 2) for v in hist["Close"]]
    highs = [round(float(v), 2) for v in hist["High"]]
    lows = [round(float(v), 2) for v in hist["Low"]]
    opens = [round(float(v), 2) for v in hist["Open"]]
    volumes = [int(v) for v in hist["Volume"]]
    dates = [d.strftime("%Y-%m-%d %H:%M") for d in hist.index]

    ema20 = _ema(closes, 20)
    ema50 = _ema(closes, 50)
    latest_close = closes[-1]
    latest_rsi = _rsi(closes)
    high_window = max(highs[-60:]) if highs else latest_close
    low_window = min(lows[-60:]) if lows else latest_close

    indicators = {
        "rsi_14": round(latest_rsi, 2) if latest_rsi is not None else None,
        "ema_20": round(ema20[-1], 2) if ema20 else None,
        "ema_50": round(ema50[-1], 2) if ema50 else None,
        "volume": volumes[-1] if volumes else None,
        "close_vs_ema20_pct": round(((latest_close / ema20[-1]) - 1) * 100, 2) if ema20 and ema20[-1] else None,
    }

    levels = {
        "resistance": round(high_window, 2),
        "support": round(low_window, 2),
        "midpoint": round((high_window + low_window) / 2, 2),
    }

    return {
        "dates": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "prices": closes,
        "volumes": volumes,
        "indicators": indicators,
        "levels": levels,
        "source": "yfinance",
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "version": "1.0.0", "timestamp": now_ist().isoformat()}


@app.get("/api/health")
def health_check():
    return {"status": "healthy", "timestamp": now_ist().isoformat()}


@app.get("/api/health/status", response_model=list[HealthStatus])
def get_system_health():
    """Component-level health. Checks DB connectivity and model availability."""
    from pathlib import Path
    statuses = []

    # Database
    try:
        get_latest_prices()
        statuses.append(HealthStatus(name="Database", status="healthy", value="Connected", message="Query successful"))
    except Exception as e:
        statuses.append(HealthStatus(name="Database", status="degraded", value="Error", message=str(e)[:120]))

    # Model artifacts
    model_dir = Path(os.getenv("MODEL_DIR", "data/production/models"))
    model_items = []
    if model_dir.exists():
        for item in model_dir.iterdir():
            if item.is_dir() or item.suffix in (".joblib", ".pkl"):
                model_items.append(item.name)
                
    if model_items:
        statuses.append(HealthStatus(
            name="ML Models",
            status="healthy",
            value=f"{len(model_items)} loaded",
            message=", ".join(model_items),
        ))
    else:
        statuses.append(HealthStatus(
            name="ML Models",
            status="degraded",
            value="Missing",
            message=f"No models found in {model_dir}.",
        ))

    # Data pipeline freshness (rough check: latest price timestamp)
    try:
        prices = get_latest_prices()
        priced = {str(p.get("symbol") or "").upper() for p in prices}
        try:
            from config.universe import NSE_UNIVERSE
            expected = {str(s["symbol"]).upper() for s in NSE_UNIVERSE}
        except Exception:
            expected = priced
        missing = sorted(expected - priced)
        coverage = f"{len(priced)}/{len(expected)}"
        msg = f"Missing prices: {', '.join(missing[:8])}" if missing else "All configured symbols priced"
        statuses.append(HealthStatus(
            name="Data Pipeline",
            status="healthy" if prices and not missing else "degraded",
            value=coverage,
            message=msg,
        ))
    except Exception as e:
        statuses.append(HealthStatus(name="Data Pipeline", status="degraded", value="Error", message=str(e)[:120]))

    statuses.append(HealthStatus(name="API Gateway", status="healthy", value="100%", message="Serving requests"))
    return statuses


@app.get("/api/indices", response_model=list[IndexData])
def api_get_indices():
    try:
        from database.connection import get_indices
        rows = get_indices()
        return [
            IndexData(
                id=str(r.get("id", "")),
                name=r.get("name", ""),
                value=_to_float(r.get("value")),
                change=_to_float(r.get("change")),
            )
            for r in rows
        ]
    except Exception as e:
        logger.error(f"get_indices failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Database unavailable")


@app.get("/api/stocks", response_model=list[StockData])
def get_stocks(
    sector: str | None = Query(default=None),
    search: str | None = Query(default=None),
    current_user: dict = Depends(verify_token),
):
    try:
        prices = get_latest_prices()
        signal_map = _latest_signal_map()
        stocks = [_stock_from_row(p, signal_map) for p in prices]

        if sector:
            stocks = [s for s in stocks if s.sector.lower() == sector.lower()]

        if search:
            q = search.lower()
            stocks = [s for s in stocks if q in s.symbol.lower() or q in s.name.lower()]

        return stocks
    except Exception as e:
        logger.error(f"get_stocks failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Database unavailable")


@app.get("/api/stocks/{symbol}", response_model=StockData)
def get_stock(symbol: str):
    try:
        p = get_stock_price(symbol.upper())
        if not p:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
        return _stock_from_row(p, _latest_signal_map())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"get_stock({symbol}) failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Database unavailable")


@app.get("/api/stocks/{symbol}/history")
def get_stock_history(symbol: str, days: int = Query(default=180, le=365)):
    """Fetch daily stock history from yfinance for charting."""
    try:
        yf_symbol = f"{symbol.upper()}.NS"
        ticker = yf.Ticker(yf_symbol)
        
        # Determine period/interval
        if days <= 5:
            period = "5d"
            interval = "15m"
        elif days <= 30:
            period = "1mo"
            interval = "1h"
        elif days <= 90:
            period = "3mo"
            interval = "1d"
        elif days <= 180:
            period = "6mo"
            interval = "1d"
        else:
            period = "1y"
            interval = "1d"
            
        hist = ticker.history(period=period, interval=interval, auto_adjust=True)
        if hist.empty:
            latest = get_stock_price(symbol.upper())
            if not latest:
                return {"dates": [], "prices": [], "volumes": [], "source": "empty"}
            ts = latest.get("timestamp")
            return {
                "dates": [ts.isoformat() if hasattr(ts, "isoformat") else str(ts or "")],
                "open": [_to_float(latest.get("price"))],
                "high": [_to_float(latest.get("price"))],
                "low": [_to_float(latest.get("price"))],
                "prices": [_to_float(latest.get("price"))],
                "volumes": [_to_int(latest.get("volume"))],
                "indicators": {},
                "levels": {},
                "source": "database_latest",
            }

        return _derive_history_payload(hist.dropna(subset=["Close"]))
    except Exception as e:
        logger.error(f"Failed to fetch history for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions", response_model=list[PredictionData])
def api_get_predictions(
    filter: str | None = Query(default=None, description="correct | wrong | pending"),
    symbol: str | None = Query(default=None),
    horizon: str | None = Query(default=None, description="INTRADAY | SWING | LONGTERM"),
    limit: int = Query(default=200, le=1000),
):
    try:
        rows = get_predictions(result=filter, limit=limit)

        if symbol:
            rows = [r for r in rows if r.get("symbol", "").upper() == symbol.upper()]
        if horizon:
            rows = [r for r in rows if r.get("horizon", "").upper() == horizon.upper()]

        result = []
        for p in rows:
            pd_val = p.get("prediction_date") or p.get("prediction_time")
            # Handle both datetime objects and string values
            if pd_val:
                if isinstance(pd_val, str):
                    date_str = pd_val
                else:
                    date_str = pd_val.isoformat()
            else:
                date_str = ""
            result.append(PredictionData(
                date=date_str,
                symbol=p["symbol"],
                prediction=p["prediction"],
                horizon=p["horizon"],
                confidence=_to_float(p.get("confidence")),
                entry_price=_to_float(p.get("entry_price")) or None,
                stop_loss=_to_float(p.get("stop_loss")) or None,
                target_price=_to_float(p.get("target_price")) or None,
                actual=p.get("actual"),
                result=p.get("result"),
                reason=p.get("reason"),
            ))
        return result
    except Exception as e:
        logger.error(f"get_predictions failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Database unavailable")


@app.get("/api/calibration")
def api_get_calibration():
    """
    Actual win-rate bucketed by model confidence.
    Only counts resolved predictions (result = correct | wrong).
    Use this to verify the model is calibrated — bucket win rate should
    approximate the confidence midpoint.
    """
    try:
        rows = get_predictions(limit=2000)
        buckets: dict[str, list[int]] = {
            "50-60": [0, 0],
            "60-70": [0, 0],
            "70-80": [0, 0],
            "80-90": [0, 0],
            "90-100": [0, 0],
        }
        for p in rows:
            if p.get("result") not in ("correct", "wrong"):
                continue
            conf = _to_float(p.get("confidence")) * 100
            if conf < 50:
                continue
            key = (
                "50-60" if conf < 60 else
                "60-70" if conf < 70 else
                "70-80" if conf < 80 else
                "80-90" if conf < 90 else
                "90-100"
            )
            buckets[key][1] += 1
            if p.get("result") == "correct":
                buckets[key][0] += 1

        return {
            "calibration": {
                k: round(v[0] / v[1], 3) if v[1] > 0 else None
                for k, v in buckets.items()
            },
            "sample_counts": {k: v[1] for k, v in buckets.items()},
        }
    except Exception as e:
        logger.error(f"get_calibration failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Database unavailable")


@app.get("/api/ticker", response_model=list[TickerItem])
def get_ticker_data(n: int = Query(default=10, le=50)):
    """Top n movers by absolute change_pct for the ticker bar."""
    try:
        prices = get_latest_prices()
        if not prices:
            return []
        sorted_prices = sorted(prices, key=lambda x: abs(_to_float(x.get("change_pct"))), reverse=True)
        return [
            TickerItem(
                name=s["symbol"],
                value=_to_float(s.get("price")),
                change=_to_float(s.get("change_pct")),
                up=_to_float(s.get("change_pct")) >= 0,
            )
            for s in sorted_prices[:n]
        ]
    except Exception as e:
        logger.error(f"get_ticker_data failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Database unavailable")


@app.get("/api/sectors", response_model=list[SectorItem])
def get_sector_data():
    try:
        data = get_sector_performance()
        if data:
            return [SectorItem(**row) for row in data]

        # Compute from price table if no precomputed sector table
        prices = get_latest_prices()
        if not prices:
            return []

        sectors: dict[str, dict] = {}
        for s in prices:
            sec = s.get("sector") or "Unknown"
            chg = _to_float(s.get("change_pct"))
            vol = _to_int(s.get("volume"))
            if sec not in sectors:
                sectors[sec] = {"total_chg": 0.0, "count": 0, "top_stock": s["symbol"], "top_chg": chg, "vol": 0}
            sectors[sec]["total_chg"] += chg
            sectors[sec]["count"] += 1
            sectors[sec]["vol"] += vol
            if chg > sectors[sec]["top_chg"]:
                sectors[sec]["top_chg"] = chg
                sectors[sec]["top_stock"] = s["symbol"]

        return [
            SectorItem(
                name=sec,
                change=round(v["total_chg"] / max(1, v["count"]), 3),
                top_stock=v["top_stock"],
                volume=v["vol"],
            )
            for sec, v in sectors.items()
        ]
    except Exception as e:
        logger.error(f"get_sector_data failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Database unavailable")


@app.get("/api/metrics/performance", response_model=list[MetricData])
def get_performance_metrics():
    try:
        from database.connection import get_performance_metrics as _get
        rows = _get()
        return [MetricData(**r) for r in rows] if rows else []
    except Exception as e:
        logger.error(f"get_performance_metrics failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Database unavailable")


@app.get("/api/metrics/model", response_model=list[MetricData])
def get_model_metrics():
    try:
        from database.connection import get_model_metrics as _get
        rows = _get()
        return [MetricData(**r) for r in rows] if rows else []
    except Exception as e:
        logger.error(f"get_model_metrics failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Database unavailable")


@app.get("/api/oms/orders")
def get_orders():
    """
    Read-only view of open OMS orders.
    The OMS write path is internal — this endpoint exposes order state
    for dashboard monitoring only.
    """
    try:
        from database.connection import get_open_orders
        return get_open_orders()
    except ImportError:
        return []
    except Exception as e:
        logger.error(f"get_orders failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Database unavailable")


@app.get("/api/news/{symbol}")
def get_news(symbol: str):
    """
    News headlines for a symbol.
    Reads from the RSS ingestion table if available.
    Returns empty list rather than mock data when table is empty.
    """
    try:
        from database.connection import get_news_for_symbol
        return get_news_for_symbol(symbol.upper())
    except ImportError:
        logger.warning("get_news_for_symbol not implemented in database.connection")
        return []
    except Exception as e:
        logger.error(f"get_news({symbol}) failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Database unavailable")





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
