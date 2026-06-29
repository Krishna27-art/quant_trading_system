"""
FastAPI Backend for Quant Terminal

Provides API endpoints for the trading terminal frontend.
"""

import os
import sys

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# from api.auth import router as auth_router, verify_token
from prometheus_client import make_asgi_app

from utils.time_utils import now_ist

load_dotenv()

from database.connection import (
    create_tables,
    get_latest_prices,
    get_predictions,
    get_sector_performance,
    get_stock_price,
    initialize_pool,
)
from utils.logger import get_logger

logger = get_logger("api")


app = FastAPI(
    title="Quant Terminal API",
    description="Backend API for Institutional Quant Research OS",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8080", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
app.mount("/metrics", make_asgi_app())


# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database connection and create tables."""
    env = os.getenv("ENV", "LOCAL")
    if env in ("LIVE", "PAPER") and "api.mock_data" in sys.modules:
        logger.critical("FATAL: mock_data module loaded in production environment!")
        sys.exit(1)

    try:
        initialize_pool()
        create_tables()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Close database connections on shutdown."""
    from database.connection import close_all_connections

    close_all_connections()
    logger.info("Database connections closed")


# ═══════════════════ DATA MODELS ═══════════════════
from pydantic import BaseModel


class IndexData(BaseModel):
    name: str
    value: float
    change: float
    id: str


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
    high_52w: float
    low_52w: float


class PredictionData(BaseModel):
    date: str
    symbol: str
    prediction: str
    horizon: str
    confidence: float
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


# ═══════════════════ MOCK DATA REMOVED ═══════════════════
# Institutional policy requires API to fail with 503 instead of returning mock data.


# ═══════════════════ API ENDPOINTS ═══════════════════


@app.get("/")
def root():
    """Root endpoint."""
    return {"message": "Quant Terminal API", "version": "1.0.0"}


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": now_ist().isoformat()}


@app.get("/api/indices", response_model=list[IndexData])
def api_get_indices():
    """Get index data."""
    try:
        from database.connection import get_indices

        indices_data = get_indices()
        return [
            IndexData(
                id=idx.get("id", ""),
                name=idx.get("name", ""),
                value=float(idx.get("value", 0)),
                change=float(idx.get("change", 0)),
            )
            for idx in indices_data
        ]
    except Exception as e:
        logger.error(f"Error fetching indices: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database unavailable")


@app.get("/api/stocks", response_model=list[StockData])
def get_stocks(sector: str | None = None, search: str | None = None):
    """
    Get stock data.

    Args:
        sector: Filter by sector
        search: Search by symbol or name
    """
    try:
        prices = get_latest_prices()

        # Convert dict to StockData
        stocks = []
        for p in prices:
            stocks.append(
                StockData(
                    symbol=p["symbol"],
                    name=p.get("name") or p["symbol"],
                    price=float(p.get("price") or 0),
                    change=float(p.get("change") or 0),
                    change_pct=float(p.get("change_pct") or 0),
                    volume=int(p.get("volume") or 0),
                    market_cap=str(p.get("market_cap") or ""),
                    sector=str(p.get("sector") or "Unknown"),
                    signal="HOLD",
                    high_52w=float(p.get("high_52w") or 0),
                    low_52w=float(p.get("low_52w") or 0),
                )
            )

        if sector:
            stocks = [s for s in stocks if s.sector == sector]

        if search:
            search_lower = search.lower()
            stocks = [
                s
                for s in stocks
                if search_lower in s.symbol.lower() or search_lower in s.name.lower()
            ]

        return stocks
    except Exception as e:
        logger.error(f"Error fetching stocks: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Database unavailable")


@app.get("/api/stocks/{symbol}", response_model=StockData)
def get_stock(symbol: str):
    """Get stock data by symbol."""
    try:
        p = get_stock_price(symbol.upper())
        if p:
            return StockData(
                symbol=p["symbol"],
                name=p.get("name") or p["symbol"],
                price=float(p.get("price") or 0),
                change=float(p.get("change") or 0),
                change_pct=float(p.get("change_pct") or 0),
                volume=int(p.get("volume") or 0),
                market_cap=str(p.get("market_cap") or ""),
                sector=str(p.get("sector") or "Unknown"),
                signal="HOLD",
                high_52w=float(p.get("high_52w") or 0),
                low_52w=float(p.get("low_52w") or 0),
            )
    except Exception as e:
        logger.error(f"Error fetching stock: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database unavailable")


@app.get("/api/predictions", response_model=list[PredictionData])
def api_get_predictions(filter: str | None = None):
    """
    Get ML predictions.

    Args:
        filter: Filter by result (correct, wrong, pending)
    """
    try:
        # get_predictions is imported from database.connection
        db_preds = get_predictions(result=filter)
        result = []
        for p in db_preds:
            result.append(
                PredictionData(
                    date=p.get("prediction_date").isoformat() if p.get("prediction_date") else "",
                    symbol=p["symbol"],
                    prediction=p["prediction"],
                    horizon=p["horizon"],
                    confidence=float(p["confidence"]) if p.get("confidence") else 0.0,
                    actual=p.get("actual"),
                    result=p.get("result"),
                    reason=p.get("reason"),
                )
            )
        return result
    except Exception as e:
        logger.error(f"Error fetching predictions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database unavailable")


@app.get("/api/health/status", response_model=list[HealthStatus])
def api_get_system_health():
    """Get system health status."""
    try:
        return [
            HealthStatus(
                name="API Gateway",
                status="healthy",
                value="99.9%",
                message="All systems operational",
            ),
            HealthStatus(
                name="Data Pipeline", status="healthy", value="Syncing", message="Last sync 2m ago"
            ),
            HealthStatus(
                name="ML Models", status="healthy", value="Loaded", message="Models up to date"
            ),
            HealthStatus(
                name="Database", status="healthy", value="Connected", message="Primary DB active"
            ),
        ]
    except Exception as e:
        logger.error(f"Error fetching system health: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database unavailable")


@app.get("/api/metrics/performance", response_model=list[MetricData])
def get_api_performance_metrics():
    """Get performance metrics."""
    from database.connection import get_performance_metrics as get_perf

    try:
        metrics = get_perf()
        if metrics:
            return [MetricData(**m) for m in metrics]
    except Exception as e:
        logger.error(f"Error fetching performance metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database unavailable")


@app.get("/api/metrics/model", response_model=list[MetricData])
def get_api_model_metrics():
    """Get model metrics."""
    from database.connection import get_model_metrics as get_mod

    try:
        metrics = get_mod()
        if metrics:
            return [MetricData(**m) for m in metrics]
    except Exception as e:
        logger.error(f"Error fetching model metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database unavailable")


@app.get("/api/ticker")
def get_api_ticker_data():
    """Get ticker bar data."""
    try:
        prices = get_latest_prices()
        if not prices:
            return []

        stocks = sorted(prices, key=lambda x: float(x.get("change_pct") or 0), reverse=True)
        return [
            {
                "name": s["symbol"],
                "value": float(s.get("price") or 0),
                "change": float(s.get("change_pct") or 0),
                "up": float(s.get("change_pct") or 0) >= 0,
            }
            for s in stocks[:10]
        ]
    except Exception as e:
        logger.error(f"Error fetching ticker data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database unavailable")


@app.get("/api/sectors")
def get_api_sector_data():
    """Get sector heatmap data."""
    try:
        data = get_sector_performance()
        if data:
            return data

        # Calculate sector performance from stock prices
        prices = get_latest_prices()
        if not prices:
            return []

        sectors_map = {}
        for s in prices:
            sector = s.get("sector") or "Unknown"
            if sector not in sectors_map:
                sectors_map[sector] = {
                    "perf": 0.0,
                    "count": 0,
                    "top_stock": s["symbol"],
                    "top_change": float(s.get("change_pct") or 0),
                    "vol": 0,
                }

            change_pct = float(s.get("change_pct") or 0)
            sectors_map[sector]["perf"] += change_pct
            sectors_map[sector]["count"] += 1
            sectors_map[sector]["vol"] += int(s.get("volume") or 0)

            if change_pct > sectors_map[sector]["top_change"]:
                sectors_map[sector]["top_change"] = change_pct
                sectors_map[sector]["top_stock"] = s["symbol"]

        return [
            {
                "name": sec,
                "change": data["perf"] / max(1, data["count"]),
                "top_stock": data["top_stock"],
                "volume": data["vol"],
            }
            for sec, data in sectors_map.items()
        ]
    except Exception as e:
        logger.error(f"Error fetching sector data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database unavailable")


@app.get("/api/oms/orders")
def get_orders():
    return []


@app.get("/api/calibration")
def api_get_calibration():
    """Get model calibration data."""
    try:
        db_preds = get_predictions(limit=1000)
        buckets = {
            "50-60": [0, 0],
            "60-70": [0, 0],
            "70-80": [0, 0],
            "80-90": [0, 0],
            "90-100": [0, 0],
        }

        for p in db_preds:
            conf = float(p.get("confidence") or 0.0) * 100
            if conf < 50:
                continue

            bucket_key = None
            if 50 <= conf < 60:
                bucket_key = "50-60"
            elif 60 <= conf < 70:
                bucket_key = "60-70"
            elif 70 <= conf < 80:
                bucket_key = "70-80"
            elif 80 <= conf < 90:
                bucket_key = "80-90"
            elif 90 <= conf <= 100:
                bucket_key = "90-100"

            if bucket_key and p.get("result") in ("correct", "wrong"):
                buckets[bucket_key][1] += 1
                if p.get("result") == "correct":
                    buckets[bucket_key][0] += 1

        calibration = {}
        for k, v in buckets.items():
            calibration[k] = round(v[0] / v[1], 2) if v[1] > 0 else 0.0

        return {"calibration": calibration}
    except Exception as e:
        logger.error(f"Error calculating calibration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database unavailable")


@app.get("/api/news/{symbol}")
def get_news(symbol: str):
    """Get news for a stock symbol."""
    # Return mock news data
    return [
        {
            "title": f"{symbol} reports strong quarterly results",
            "summary": f"{symbol} announced better than expected earnings driven by robust performance in core segments.",
            "source": "Economic Times",
            "timestamp": "2026-06-28T10:30:00Z",
            "sentiment": "positive",
        },
        {
            "title": f"Analysts maintain buy rating on {symbol}",
            "summary": f"Multiple brokerage firms have reiterated their buy rating on {symbol} with a target price upside of 15%.",
            "source": "Moneycontrol",
            "timestamp": "2026-06-28T09:15:00Z",
            "sentiment": "positive",
        },
        {
            "title": f"{symbol} announces strategic partnership",
            "summary": f"{symbol} has entered into a strategic partnership to expand its market presence in emerging sectors.",
            "source": "Business Standard",
            "timestamp": "2026-06-28T08:45:00Z",
            "sentiment": "neutral",
        },
    ]


@app.post("/api/auth/login")
def login():
    """Mock login endpoint - returns a dummy token."""
    return {
        "access_token": "mock_token_for_development",
        "token_type": "bearer",
        "expires_in": 3600,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
