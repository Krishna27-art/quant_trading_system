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
        "http://localhost:5500",
        "http://127.0.0.1:5500",
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


@app.get("/api/sectors")
def get_sectors():
    """Return list of available sectors from NSE_UNIVERSE."""
    try:
        from config.universe import NSE_UNIVERSE
        sectors = sorted({s["sector"] for s in NSE_UNIVERSE})
        return {"sectors": sectors}
    except Exception as e:
        logger.error(f"get_sectors failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Failed to load sectors")


@app.get("/api/trades/bad_diagnosis")
def bad_trade_diagnosis(symbol: str):
    """
    Diagnose a potentially bad trade by analyzing current price vs entry, SL, target.
    Returns status, slippage, VIX change, VWAP deviation, and other metrics.
    """
    try:
        from database.db_sync import SessionLocal
        from sqlalchemy import text
        
        db = SessionLocal()
        
        # Get latest prediction for this symbol
        result = db.execute(
            text("""
                SELECT entry_price, stop_loss, target_price, prediction_time, confidence
                FROM predictions
                WHERE symbol = :symbol
                ORDER BY prediction_time DESC
                LIMIT 1
            """),
            {"symbol": symbol.upper()}
        ).fetchone()
        
        if not result:
            return {"error": f"No predictions found for {symbol}"}
        
        entry_price = float(result[0])
        stop_loss = float(result[1])
        target_price = float(result[2])
        prediction_time = result[3]
        
        # Get current price
        price_result = db.execute(
            text("""
                SELECT price FROM stock_prices
                WHERE symbol = :symbol
                ORDER BY timestamp DESC
                LIMIT 1
            """),
            {"symbol": symbol.upper()}
        ).fetchone()
        
        if not price_result:
            return {"error": f"No current price found for {symbol}"}
        
        current_price = float(price_result[0])
        
        # Calculate metrics
        slippage_pct = abs((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
        
        # Determine status based on distance to SL/Target
        if current_price <= stop_loss:
            status = "stopped_out"
        elif current_price >= target_price:
            status = "target_hit"
        else:
            dist_to_sl = abs(current_price - stop_loss) / abs(entry_price - stop_loss) if entry_price != stop_loss else 0
            dist_to_target = abs(target_price - current_price) / abs(target_price - entry_price) if target_price != entry_price else 0
            status = "at_risk" if dist_to_sl < 0.3 else "on_track"
        
        # Get VIX (mock - would need actual VIX data)
        vix_change = 0.0
        
        # VWAP deviation (mock - would need intraday VWAP data)
        vwap_dev_pct = 0.0
        
        return {
            "status": status,
            "entry_price": entry_price,
            "current_price": current_price,
            "stop_loss": stop_loss,
            "target_price": target_price,
            "slippage_pct": slippage_pct,
            "vix_change": vix_change,
            "vwap_dev_pct": vwap_dev_pct,
            "prediction_time": str(prediction_time) if prediction_time else None
        }
        
    except Exception as e:
        logger.error(f"bad_trade_diagnosis failed: {e}", exc_info=True)
        return {"error": str(e)}


@app.post("/api/calibration/recalibrate")
def recalibrate():
    """
    Trigger recalibration of probability models.
    Fetches historical predictions and outcomes, then refits calibrators.
    """
    try:
        from database.db_sync import SessionLocal
        from database.models import Prediction
        from prediction_intelligence.calibration import fit_calibrator
        from sqlalchemy import text

        db = SessionLocal()
        
        # Fetch historical predictions with outcomes
        result = db.execute(
            text("""
                SELECT horizon, confidence, actual_outcome
                FROM predictions
                WHERE actual_outcome IN ('WIN', 'LOSS')
                ORDER BY prediction_time DESC
                LIMIT 1000
            """)
        ).fetchall()
        
        if not result:
            return {"status": "ok", "message": "No historical outcomes found for calibration"}
        
        # Group by timeframe
        timeframe_data = {}
        for row in result:
            horizon = row[0]  # INTRADAY, SWING, LONGTERM
            confidence = float(row[1])
            outcome = 1 if row[2] == "WIN" else 0
            
            if horizon not in timeframe_data:
                timeframe_data[horizon] = {"raw_probs": [], "outcomes": []}
            timeframe_data[horizon]["raw_probs"].append(confidence)
            timeframe_data[horizon]["outcomes"].append(outcome)
        
        # Fit calibrators for each timeframe
        fitted_count = 0
        for timeframe, data in timeframe_data.items():
            if len(data["raw_probs"]) >= 50:
                fit_calibrator(data["raw_probs"], data["outcomes"], timeframe, method="isotonic")
                fitted_count += 1
                logger.info(f"Recalibrated {timeframe} with {len(data['raw_probs'])} samples")
        
        db.close()
        
        return {
            "status": "ok",
            "message": f"Recalibration completed for {fitted_count} timeframes",
            "fitted_timeframes": list(timeframe_data.keys())
        }
        
    except Exception as e:
        logger.error(f"Recalibration failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Recalibration failed: {str(e)}")


# ---------------------------------------------------------------------------
# Training & Backtest Endpoints
# ---------------------------------------------------------------------------

class TrainingRequest(BaseModel):
    model_version: str
    timeframe: str
    hyperparams: dict[str, Any] = {}


class BacktestRequest(BaseModel):
    model_version: str
    timeframe: str
    start_date: str
    end_date: str


@app.post("/api/train/run")
def run_training(request: TrainingRequest):
    """
    Trigger model training job.
    This is a placeholder - actual training should run as async job via Celery/Prefect.
    """
    try:
        # In production, this would queue a job and return a job_id
        # For now, return a success message
        logger.info(f"Training requested for {request.model_version} ({request.timeframe})")
        return {
            "status": "queued",
            "job_id": f"train_{request.model_version}_{request.timeframe}_{now_ist().timestamp()}",
            "message": "Training job queued. Check status endpoint for progress."
        }
    except Exception as e:
        logger.error(f"Training request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/api/backtest/run")
def run_backtest(request: BacktestRequest):
    """
    Trigger backtest job.
    This is a placeholder - actual backtest should run as async job via Celery/Prefect.
    """
    try:
        # In production, this would queue a job and return a job_id
        logger.info(f"Backtest requested for {request.model_version} ({request.timeframe}) from {request.start_date} to {request.end_date}")
        return {
            "status": "queued",
            "job_id": f"backtest_{request.model_version}_{request.timeframe}_{now_ist().timestamp()}",
            "message": "Backtest job queued. Check status endpoint for progress."
        }
    except Exception as e:
        logger.error(f"Backtest request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")


# ---------------------------------------------------------------------------
# Paper Trading Endpoints
# ---------------------------------------------------------------------------

class PaperTradeCreate(BaseModel):
    symbol: str
    side: str  # BUY or SELL
    quantity: int
    entry_price: float


class PaperTradeUpdate(BaseModel):
    exit_price: float | None = None
    status: str | None = None  # OPEN, CLOSED, CANCELLED


@app.post("/api/paper/trades")
def create_paper_trade(trade: PaperTradeCreate):
    """Create a new paper trade."""
    try:
        from database.db_sync import SessionLocal
        from database.models import PaperTrade
        import uuid
        from utils.time_utils import now_ist

        db = SessionLocal()
        
        paper_trade = PaperTrade(
            id=str(uuid.uuid4()),
            user_id="default",  # In production, get from auth token
            symbol=trade.symbol.upper(),
            side=trade.side.upper(),
            quantity=trade.quantity,
            entry_price=trade.entry_price,
            entry_timestamp=now_ist(),
            status="OPEN",
            created_at=now_ist(),
            updated_at=now_ist(),
        )
        
        db.add(paper_trade)
        db.commit()
        db.refresh(paper_trade)
        db.close()
        
        return {
            "id": paper_trade.id,
            "symbol": paper_trade.symbol,
            "side": paper_trade.side,
            "quantity": paper_trade.quantity,
            "entry_price": float(paper_trade.entry_price),
            "status": paper_trade.status,
            "created_at": paper_trade.created_at.isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to create paper trade: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create paper trade: {str(e)}")


@app.get("/api/paper/trades")
def list_paper_trades(status: str | None = Query(default=None)):
    """List all paper trades, optionally filtered by status."""
    try:
        from database.db_sync import SessionLocal
        from database.models import PaperTrade
        from sqlalchemy import text

        db = SessionLocal()
        
        query = "SELECT * FROM paper_trades"
        params = {}
        if status:
            query += " WHERE status = :status"
            params["status"] = status.upper()
        query += " ORDER BY created_at DESC"
        
        result = db.execute(text(query), params).fetchall()
        db.close()
        
        trades = []
        for row in result:
            trades.append({
                "id": row[0],
                "user_id": row[1],
                "symbol": row[2],
                "side": row[3],
                "quantity": row[4],
                "entry_price": float(row[5]) if row[5] else None,
                "exit_price": float(row[6]) if row[6] else None,
                "entry_timestamp": row[7].isoformat() if row[7] else None,
                "exit_timestamp": row[8].isoformat() if row[8] else None,
                "status": row[9],
                "pnl": float(row[10]) if row[10] else None,
                "created_at": row[11].isoformat() if row[11] else None,
            })
        
        return {"trades": trades}
    except Exception as e:
        logger.error(f"Failed to list paper trades: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list paper trades: {str(e)}")


@app.put("/api/paper/trades/{trade_id}")
def update_paper_trade(trade_id: str, update: PaperTradeUpdate):
    """Update a paper trade (exit price or status)."""
    try:
        from database.db_sync import SessionLocal
        from database.models import PaperTrade
        from sqlalchemy import text
        from utils.time_utils import now_ist

        db = SessionLocal()
        
        # Fetch existing trade
        trade = db.execute(
            text("SELECT * FROM paper_trades WHERE id = :trade_id"),
            {"trade_id": trade_id}
        ).fetchone()
        
        if not trade:
            db.close()
            raise HTTPException(status_code=404, detail="Trade not found")
        
        # Build update query
        updates = []
        params = {"trade_id": trade_id, "updated_at": now_ist()}
        
        if update.exit_price is not None:
            updates.append("exit_price = :exit_price")
            params["exit_price"] = update.exit_price
        
        if update.status is not None:
            updates.append("status = :status")
            params["status"] = update.status.upper()
            
            # If closing, set exit timestamp
            if update.status.upper() == "CLOSED":
                updates.append("exit_timestamp = :exit_timestamp")
                params["exit_timestamp"] = now_ist()
        
        if updates:
            query = f"UPDATE paper_trades SET {', '.join(updates)}, updated_at = :updated_at WHERE id = :trade_id"
            db.execute(text(query), params)
            db.commit()
        
        db.close()
        return {"status": "ok", "message": "Trade updated"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update paper trade: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update paper trade: {str(e)}")


@app.delete("/api/paper/trades/{trade_id}")
def delete_paper_trade(trade_id: str):
    """Delete a paper trade."""
    try:
        from database.db_sync import SessionLocal
        from sqlalchemy import text

        db = SessionLocal()
        db.execute(text("DELETE FROM paper_trades WHERE id = :trade_id"), {"trade_id": trade_id})
        db.commit()
        db.close()
        
        return {"status": "ok", "message": "Trade deleted"}
    except Exception as e:
        logger.error(f"Failed to delete paper trade: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete paper trade: {str(e)}")


# ---------------------------------------------------------------------------
# FII/DII Net Flow Endpoint
# ---------------------------------------------------------------------------

@app.get("/api/market/fii_dii")
def get_fii_dii_flow():
    """
    Get FII/DII net flow data (5-day rolling).
    This is a placeholder - should fetch from NSE or data provider.
    """
    try:
        # In production, fetch from NSE or data provider
        # For now, return mock data
        return {
            "fii_net_flow": 1250.5,  # Cr
            "dii_net_flow": 890.3,   # Cr
            "net_flow": 1540.8,      # Cr (FII + DII)
            "date": now_ist().date().isoformat(),
            "trend": "bullish"  # bullish, bearish, neutral
        }
    except Exception as e:
        logger.error(f"Failed to get FII/DII data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get FII/DII data: {str(e)}")


@app.get("/api/market-hours")
def get_market_hours():
    """
    Get current market hours status for NSE (Indian market).
    Returns open/closed status and next open time.
    """
    try:
        from datetime import datetime, time, timedelta
        from zoneinfo import ZoneInfo
        
        ist = ZoneInfo("Asia/Kolkata")
        now = datetime.now(ist)
        current_time = now.time()
        
        # NSE market hours: 9:15 AM to 3:30 PM IST
        market_open = time(9, 15)
        market_close = time(15, 30)
        
        is_open = market_open <= current_time <= market_close
        
        # Calculate next open time
        if is_open:
            next_open = now.replace(hour=15, minute=30, second=0, microsecond=0)
        elif current_time < market_open:
            next_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        else:
            # Next trading day
            next_open = (now.replace(hour=9, minute=15, second=0, microsecond=0) + 
                        timedelta(days=1))
            # Skip weekends
            while next_open.weekday() >= 5:  # 5=Saturday, 6=Sunday
                next_open += timedelta(days=1)
        
        return {
            "is_open": is_open,
            "current_time": now.isoformat(),
            "next_open": next_open.isoformat(),
            "market_open": "09:15 IST",
            "market_close": "15:30 IST",
            "exchange": "NSE"
        }
    except Exception as e:
        logger.error(f"Failed to get market hours: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get market hours: {str(e)}")


@app.get("/api/options")
def get_options_data(symbol: str = Query(default="NIFTY")):
    """
    Get options chain data for a symbol.
    This is a placeholder - should fetch from NSE or data provider.
    """
    try:
        # In production, fetch from NSE options API
        # For now, return empty array
        return {
            "symbol": symbol.upper(),
            "expiry_dates": [],
            "options": []
        }
    except Exception as e:
        logger.error(f"Failed to get options data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get options data: {str(e)}")


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
