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
from fastapi import Depends, FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel
# Upstox is the single market-data source — yfinance removed
try:
    from data_platform.upstox_client import (
        # Quotes & indices
        get_candles, get_index_overview, get_sector_overview,
        get_market_status, get_bulk_quotes, get_stock_quote,
        get_ltp, get_ohlc, get_cockpit_data,
        # FII / DII
        get_fii_activity, get_dii_activity,
        # Options
        get_option_chain, get_option_expiries,
        get_oi, get_pcr, get_max_pain, get_change_oi, compute_pcr_from_chain,
        # Smartlists
        get_futures_smartlist, get_options_smartlist, get_mtf_smartlist,
        # Market info
        get_market_holidays,
        # Fundamentals
        get_company_profile, get_balance_sheet, get_income_statement,
        get_cash_flow, get_key_ratios, get_share_holdings,
        get_competitors, get_corporate_actions, get_full_fundamentals,
        # News
        get_stock_news, get_multi_stock_news,
        # Symbol lookup
        symbol_to_key,
    )
    UPSTOX_OK = True
except Exception as _ue:
    logger.warning(f"Upstox client not loaded: {_ue}")
    UPSTOX_OK = False


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

allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://[::]:3000",
    "http://[::1]:3000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]
env_origins = os.getenv("CORS_ALLOWED_ORIGINS")
if env_origins:
    allowed_origins.extend([o.strip() for o in env_origins.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

from observability_mlops.prometheus_metrics import MetricsCollector
metrics_collector = MetricsCollector()

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
        pred = str(row.get("prediction") or "").upper()
        if symbol and symbol not in signals:
            if pred in {"BUY", "LONG"}:
                signals[symbol] = "BUY"
            elif pred in {"SELL", "SHORT"}:
                signals[symbol] = "SELL"
            elif pred == "HOLD":
                signals[symbol] = "HOLD"
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


def _derive_history_from_upstox(candles: list[dict]) -> dict:
    """Convert Upstox OHLCV candles list into the frontend's expected payload format."""
    if not candles:
        return {"dates": [], "open": [], "high": [], "low": [], "prices": [], "volumes": [], "indicators": {}, "levels": {}, "source": "upstox"}

    # Upstox returns newest first — reverse to oldest first
    candles = list(reversed(candles))

    dates   = [c["timestamp"][:16].replace("T", " ") for c in candles]
    opens   = [round(float(c["open"]),  2) for c in candles]
    highs   = [round(float(c["high"]),  2) for c in candles]
    lows    = [round(float(c["low"]),   2) for c in candles]
    closes  = [round(float(c["close"]), 2) for c in candles]
    volumes = [int(c["volume"]) for c in candles]

    ema20 = _ema(closes, 20)
    ema50 = _ema(closes, 50)
    latest_close = closes[-1] if closes else 0.0
    latest_rsi   = _rsi(closes)
    high_window  = max(highs[-60:])  if highs  else latest_close
    low_window   = min(lows[-60:])   if lows   else latest_close

    indicators = {
        "rsi_14":            round(latest_rsi, 2) if latest_rsi is not None else None,
        "ema_20":            round(ema20[-1],  2) if ema20 else None,
        "ema_50":            round(ema50[-1],  2) if ema50 else None,
        "volume":            volumes[-1] if volumes else None,
        "close_vs_ema20_pct": round(((latest_close / ema20[-1]) - 1) * 100, 2)
                              if ema20 and ema20[-1] else None,
    }
    levels = {
        "resistance": round(high_window, 2),
        "support":    round(low_window,  2),
        "midpoint":   round((high_window + low_window) / 2, 2),
    }
    return {
        "dates":      dates,
        "open":       opens,
        "high":       highs,
        "low":        lows,
        "prices":     closes,
        "volumes":    volumes,
        "indicators": indicators,
        "levels":     levels,
        "source":     "upstox",
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/metrics")
def get_metrics(current_user: dict = Depends(verify_token)):
    """Prometheus metrics scrape endpoint protected by JWT auth."""
    from observability_mlops.prometheus_metrics import _default_registry
    data = generate_latest(_default_registry)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.get("/")
def root():
    return {"status": "ok", "version": "1.0.0", "timestamp": now_ist().isoformat()}


@app.get("/api/health")
def health_check():
    return {"status": "healthy", "timestamp": now_ist().isoformat()}


@app.get("/api/health/status", response_model=list[HealthStatus])
def get_system_health():
    """Component-level health. Checks DB connectivity, broker reachability, data feed freshness, disk, and memory."""
    from observability_mlops.health_check import HealthChecker
    db_url = os.getenv("DATABASE_URL", "sqlite:///quant.db")
    checker = HealthChecker(db_url=db_url)
    
    # Attempt to feed data freshness metric from latest price in DB
    try:
        prices = get_latest_prices()
        if prices:
            latest_time = max(p.get("timestamp") for p in prices if p.get("timestamp"))
            if latest_time:
                if isinstance(latest_time, str):
                    from datetime import datetime
                    latest_dt = datetime.fromisoformat(latest_time)
                else:
                    latest_dt = latest_time
                checker.update_last_tick(latest_dt.timestamp())
    except Exception:
        pass
        
    report = checker.run_all()
    
    statuses = []
    for component in report.components:
        statuses.append(HealthStatus(
            name=component.name.replace("_", " ").title(),
            status=component.status.value,
            value=f"{component.latency_ms:.1f}ms" if component.latency_ms > 0 else "N/A",
            message=component.message
        ))
    return statuses


@app.get("/api/indices", response_model=list[IndexData])
def api_get_indices():
    """Live NSE/BSE index data from Upstox."""
    if UPSTOX_OK:
        try:
            overview = get_index_overview()
            _id_map = {
                "NIFTY50":   ("nifty50",   "NIFTY 50"),
                "SENSEX":    ("sensex",    "BSE SENSEX"),
                "BANKNIFTY": ("banknifty", "NIFTY BANK"),
                "FINNIFTY":  ("finnifty",  "NIFTY FIN SVC"),
                "INDIAVIX":  ("indiavix",  "INDIA VIX"),
            }
            result = []
            for key, (idx_id, idx_name) in _id_map.items():
                d = overview.get(key, {})
                if d:
                    result.append(IndexData(
                        id=idx_id,
                        name=idx_name,
                        value=_to_float(d.get("last_price")),
                        change=_to_float(d.get("net_change")),
                    ))
            if result:
                return result
        except Exception as e:
            logger.warning(f"Upstox indices failed, falling back to DB: {e}")

    # Fallback: DB
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
        raise HTTPException(status_code=503, detail="Index data unavailable")


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
    from database.db_sync import SessionLocal
    from sqlalchemy import text
    
    db = SessionLocal()
    try:
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
    finally:
        db.close()


@app.post("/api/calibration/recalibrate")
def recalibrate(current_user: dict = Depends(verify_token)):
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
        try:
            # Fetch historical predictions with outcomes
            result = db.execute(
                text("""
                    SELECT horizon, confidence, actual_outcome, prediction
                    FROM predictions
                    WHERE actual_outcome IN ('WIN', 'LOSS')
                    ORDER BY prediction_time DESC
                    LIMIT 2000
                """)
            ).fetchall()
            
            if not result:
                return {"status": "ok", "message": "No historical outcomes found for calibration"}
            
            # Group by (timeframe, direction)
            calib_data = {}
            for row in result:
                horizon = row[0]       # INTRADAY, SWING, LONGTERM
                confidence = float(row[1])
                outcome = 1 if row[2] == "WIN" else 0
                direction = row[3]     # BUY, SELL
                
                key = (horizon, direction)
                if key not in calib_data:
                    calib_data[key] = {"raw_probs": [], "outcomes": []}
                calib_data[key]["raw_probs"].append(confidence)
                calib_data[key]["outcomes"].append(outcome)
            
            # Fit calibrators for each timeframe + direction combination
            fitted_count = 0
            fitted_keys = []
            for (timeframe, direction), data in calib_data.items():
                if len(data["raw_probs"]) >= 50:
                    fit_calibrator(data["raw_probs"], data["outcomes"], timeframe, direction=direction, method="isotonic")
                    fitted_count += 1
                    fitted_keys.append(f"{timeframe}_{direction}")
                    logger.info(f"Recalibrated {timeframe} ({direction}) with {len(data['raw_probs'])} samples")
            
            return {
                "status": "ok",
                "message": f"Recalibration completed for {fitted_count} direction-specific timeframes",
                "fitted_timeframes": fitted_keys
            }
        finally:
            db.close()
        
    except Exception as e:
        logger.error(f"Recalibration failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Recalibration failed: {str(e)}")


# ---------------------------------------------------------------------------
# Training & Backtest Endpoints
# ---------------------------------------------------------------------------

class TrainingRequest(BaseModel):
    model_version: str
    timeframe: str
    hyperparams: dict = {}


class BacktestRequest(BaseModel):
    model_version: str
    timeframe: str
    start_date: str
    end_date: str


@app.post("/api/train/run")
def run_training(request: TrainingRequest, current_user: dict = Depends(verify_token)):
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
def run_backtest(request: BacktestRequest, current_user: dict = Depends(verify_token)):
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
def create_paper_trade(trade: PaperTradeCreate, current_user: dict = Depends(verify_token)):
    """Create a new paper trade."""
    from database.db_sync import SessionLocal
    from database.models import PaperTrade
    import uuid
    from utils.time_utils import now_ist

    db = SessionLocal()
    try:
        paper_trade = PaperTrade(
            id=str(uuid.uuid4()),
            user_id=current_user.get("sub", "default"),
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
        
        # Increment orders metric
        try:
            metrics_collector.orders_total.labels(
                exchange="NSE",
                strategy="PAPER_TRADING",
                side=trade.side.upper(),
                order_type="LIMIT"
            ).inc()
        except Exception as me:
            logger.warning(f"Failed to increment orders_total metric: {me}")
        
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
    finally:
        db.close()


@app.get("/api/paper/trades")
def list_paper_trades(status: str | None = Query(default=None), current_user: dict = Depends(verify_token)):
    """List all paper trades, optionally filtered by status."""
    from database.db_sync import SessionLocal
    from database.models import PaperTrade
    from sqlalchemy import text

    db = SessionLocal()
    try:
        query = "SELECT * FROM paper_trades"
        params = {}
        if status:
            query += " WHERE status = :status"
            params["status"] = status.upper()
        query += " ORDER BY created_at DESC"
        
        result = db.execute(text(query), params).fetchall()
        
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
        
        return trades
    except Exception as e:
        logger.error(f"Failed to list paper trades: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list paper trades: {str(e)}")
    finally:
        db.close()


@app.put("/api/paper/trades/{trade_id}")
@app.patch("/api/paper/trades/{trade_id}")
def update_paper_trade(trade_id: str, update: PaperTradeUpdate, current_user: dict = Depends(verify_token)):
    """Update a paper trade (exit price or status)."""
    from database.db_sync import SessionLocal
    from database.models import PaperTrade
    from sqlalchemy import text
    from utils.time_utils import now_ist

    db = SessionLocal()
    try:
        # Fetch existing trade
        trade = db.execute(
            text("SELECT * FROM paper_trades WHERE id = :trade_id"),
            {"trade_id": trade_id}
        ).fetchone()
        
        if not trade:
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
            
            # Increment fills metric if the trade is closed/filled
            if update.status and update.status.upper() == "CLOSED":
                try:
                    # trade[3] is the side field (BUY/SELL)
                    side_val = trade[3] if len(trade) > 3 else "BUY"
                    metrics_collector.fills_total.labels(
                        exchange="NSE",
                        strategy="PAPER_TRADING",
                        side=side_val
                    ).inc()
                except Exception as me:
                    logger.warning(f"Failed to increment fills_total metric: {me}")
        
        return {"status": "ok", "message": "Trade updated"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update paper trade: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update paper trade: {str(e)}")
    finally:
        db.close()


@app.delete("/api/paper/trades/{trade_id}")
def delete_paper_trade(trade_id: str, current_user: dict = Depends(verify_token)):
    """Delete a paper trade."""
    from database.db_sync import SessionLocal
    from sqlalchemy import text

    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM paper_trades WHERE id = :trade_id"), {"trade_id": trade_id})
        db.commit()
        return {"status": "ok", "message": "Trade deleted"}
    except Exception as e:
        logger.error(f"Failed to delete paper trade: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete paper trade: {str(e)}")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# FII/DII — honest response (not available on Upstox free tier)
# ---------------------------------------------------------------------------

@app.get("/api/market/fii_dii")
def get_fii_dii_flow():
    """FII/DII net flow — not available on Upstox free tier."""
    if UPSTOX_OK:
        return get_fii_dii_estimate()
    return {
        "available": False,
        "message": "FII/DII data requires NSE data subscription",
        "fii_net_flow": None,
        "dii_net_flow": None,
        "date": now_ist().date().isoformat(),
    }


@app.get("/api/market-hours")
def get_market_hours():
    """NSE market status computed from IST time."""
    try:
        if UPSTOX_OK:
            return get_market_status()
        # Inline fallback
        from datetime import datetime, time, timedelta
        from zoneinfo import ZoneInfo
        ist = ZoneInfo("Asia/Kolkata")
        now = datetime.now(ist)
        t = now.time()
        is_open = time(9, 15) <= t <= time(15, 30) and now.weekday() < 5
        return {"is_open": is_open, "session": "OPEN" if is_open else "CLOSED",
                "current_ist": now.isoformat(), "exchange": "NSE"}
    except Exception as e:
        logger.error(f"Failed to get market hours: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Institutional Trading Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/institutional/pre-market")
def get_pre_market_analysis():
    """
    Pre-market environment analysis for institutional trading.
    Returns global markets, Indian indicators, risk sentiment, market regime,
    news impacts, FII/DII activity, and recommended strategies.
    """
    try:
        from data_platform.pipelines.pre_market_analyzer import PreMarketAnalyzer
        
        analyzer = PreMarketAnalyzer(use_mock_data=True)
        environment = analyzer.analyze_environment()
        
        return {
            "global_markets": {
                "us_snp500": environment.global_markets.us_snp500,
                "us_snp500_change_pct": environment.global_markets.us_snp500_change_pct,
                "us_dow_jones": environment.global_markets.us_dow_jones,
                "us_dow_change_pct": environment.global_markets.us_dow_change_pct,
                "us_nasdaq": environment.global_markets.us_nasdaq,
                "us_nasdaq_change_pct": environment.global_markets.us_nasdaq_change_pct,
                "europe_ftse": environment.global_markets.europe_ftse,
                "europe_ftse_change_pct": environment.global_markets.europe_ftse_change_pct,
                "asia_nikkei": environment.global_markets.asia_nikkei,
                "asia_nikkei_change_pct": environment.global_markets.asia_nikkei_change_pct,
            },
            "indian_indicators": {
                "gift_nifty": environment.indian_indicators.gift_nifty,
                "gift_nifty_change_pct": environment.indian_indicators.gift_nifty_change_pct,
                "usd_inr": environment.indian_indicators.usd_inr,
                "usd_inr_change_pct": environment.indian_indicators.usd_inr_change_pct,
                "dollar_index": environment.indian_indicators.dollar_index,
                "dollar_index_change_pct": environment.indian_indicators.dollar_index_change_pct,
                "crude_oil_wti": environment.indian_indicators.crude_oil_wti,
                "crude_oil_change_pct": environment.indian_indicators.crude_oil_change_pct,
            },
            "risk_sentiment": environment.risk_sentiment.value,
            "market_regime": environment.market_regime.value,
            "news_impacts": [
                {
                    "headline": n.headline,
                    "category": n.category,
                    "sentiment": n.sentiment,
                    "impact_score": n.impact_score,
                    "affected_sectors": n.affected_sectors
                }
                for n in environment.news_impacts
            ],
            "fii_activity": {
                "fii_net_buy_sell_cr": environment.fii_activity.fii_net_buy_sell_cr if environment.fii_activity else None,
                "dii_net_buy_sell_cr": environment.fii_activity.dii_net_buy_sell_cr if environment.fii_activity else None,
            } if environment.fii_activity else None,
            "overall_sentiment": environment.overall_sentiment,
            "confidence_score": environment.confidence_score,
            "recommended_strategies": environment.recommended_strategies,
            "timestamp": environment.timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Pre-market analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pre-market analysis failed: {str(e)}")


@app.get("/api/institutional/watchlist")
def get_institutional_watchlist():
    """
    Build and return institutional watchlist.
    Returns filtered stocks with metrics, sector rankings, and selection criteria.
    """
    try:
        from data_platform.pipelines.institutional_watchlist import InstitutionalWatchlistBuilder
        
        builder = InstitutionalWatchlistBuilder(use_mock_data=True)
        result = builder.build_watchlist()
        
        return {
            "watchlist": [
                {
                    "symbol": s.symbol,
                    "name": s.name,
                    "sector": s.sector,
                    "cap_tier": s.cap_tier,
                    "current_price": s.current_price,
                    "price_change_pct": s.price_change_pct,
                    "volume_today": s.volume_today,
                    "relative_volume": s.relative_volume,
                    "avg_daily_volume": s.avg_daily_volume,
                    "bid_ask_spread_pct": s.bid_ask_spread_pct,
                    "rsi_14": s.rsi_14,
                    "atr_pct": s.atr_pct,
                    "sector_strength_score": s.sector_strength_score,
                    "news_sentiment": s.news_sentiment,
                    "news_impact_score": s.news_impact_score,
                    "final_score": getattr(s, 'final_score', 0)
                }
                for s in result.selected_stocks
            ],
            "sector_rankings": [
                {
                    "sector": s.sector,
                    "strength_score": s.strength_score,
                    "avg_change_pct": s.avg_change_pct,
                    "advancers": s.advancers,
                    "decliners": s.decliners,
                    "volume_ratio": s.volume_ratio
                }
                for s in result.sector_rankings
            ],
            "selection_criteria": result.selection_criteria,
            "total_universe_size": result.total_universe_size,
            "filtered_count": result.filtered_count,
            "final_count": result.final_count,
            "timestamp": result.timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Institutional watchlist failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Watchlist generation failed: {str(e)}")


@app.get("/api/institutional/fii_dii")
def get_fii_dii_analysis():
    """
    FII/DII activity analysis with market impact assessment.
    Returns daily activity, flow trends, sector exposure, and recommendations.
    """
    try:
        from data_platform.pipelines.fii_dii_tracker import FIIDIIAnalyzer
        
        analyzer = FIIDIIAnalyzer(use_mock_data=True)
        daily_activity = analyzer.get_daily_activity()
        flow_trend = analyzer.analyze_flow_trend(days=20)
        sector_exposure = analyzer.get_sector_fii_exposure()
        impact = analyzer.assess_market_impact(daily_activity)
        
        return {
            "daily_activity": {
                "date": daily_activity.date.isoformat(),
                "fii_cash_net_cr": daily_activity.fii_cash_net_cr,
                "dii_cash_net_cr": daily_activity.dii_cash_net_cr,
                "fii_index_net_cr": daily_activity.fii_index_net_cr,
                "dii_index_net_cr": daily_activity.dii_index_net_cr,
                "fii_stock_net_cr": daily_activity.fii_stock_net_cr,
                "dii_stock_net_cr": daily_activity.dii_stock_net_cr,
                "fii_total_net_cr": daily_activity.fii_total_net_cr,
                "dii_total_net_cr": daily_activity.dii_total_net_cr,
                "net_flow_cr": daily_activity.net_flow_cr,
            },
            "flow_trend": {
                "period_start": flow_trend.period_start.isoformat(),
                "period_end": flow_trend.period_end.isoformat(),
                "avg_daily_fii_flow_cr": flow_trend.avg_daily_fii_flow_cr,
                "avg_daily_dii_flow_cr": flow_trend.avg_daily_dii_flow_cr,
                "fii_flow_trend": flow_trend.fii_flow_trend,
                "dii_flow_trend": flow_trend.dii_flow_trend,
                "fii_conviction": flow_trend.fii_conviction.value,
                "dii_conviction": flow_trend.dii_conviction.value,
                "correlation_with_nifty": flow_trend.correlation_with_nifty,
                "insights": flow_trend.insights
            },
            "sector_exposure": [
                {
                    "sector": s.sector,
                    "fii_exposure_cr": s.fii_exposure_cr,
                    "fii_weightage_pct": s.fii_weightage_pct,
                    "change_vs_previous_day_cr": s.change_vs_previous_day_cr,
                    "conviction": s.conviction.value
                }
                for s in sector_exposure
            ],
            "market_impact": impact,
            "timestamp": now_ist().isoformat()
        }
    except Exception as e:
        logger.error(f"FII/DII analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"FII/DII analysis failed: {str(e)}")


@app.get("/api/institutional/sectors")
def get_sector_strength():
    """
    Sector strength analysis with rankings.
    Returns sector performance, advance/decline ratios, and volume analysis.
    """
    try:
        from data_platform.pipelines.institutional_watchlist import InstitutionalWatchlistBuilder
        
        builder = InstitutionalWatchlistBuilder(use_mock_data=True)
        result = builder.build_watchlist()
        
        return {
            "sectors": [
                {
                    "sector": s.sector,
                    "strength_score": s.strength_score,
                    "avg_change_pct": s.avg_change_pct,
                    "advancers": s.advancers,
                    "decliners": s.decliners,
                    "unchanged": s.unchanged,
                    "volume_ratio": s.volume_ratio,
                    "num_stocks": s.num_stocks
                }
                for s in result.sector_rankings
            ],
            "timestamp": result.timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Sector strength analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Sector analysis failed: {str(e)}")


# ---------------------------------------------------------------------------
# Swing Trading Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/swing/watchlist")
def get_swing_watchlist():
    """
    Build and return institutional swing trading watchlist.
    Returns 10-30 high-conviction candidates with 0-100 scores.
    """
    try:
        from data_platform.pipelines.swing_watchlist import SwingWatchlistBuilder
        
        builder = SwingWatchlistBuilder(use_mock_data=True)
        result = builder.build_watchlist()
        summary = builder.get_watchlist_summary(result)
        
        return summary
    except Exception as e:
        logger.error(f"Swing watchlist failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Swing watchlist failed: {str(e)}")


@app.get("/api/swing/score/{symbol}")
def get_swing_score(symbol: str):
    """
    Get swing trading score for a specific symbol.
    Returns component scores and final 0-100 score.
    """
    try:
        from data_platform.pipelines.swing_trading_scorer import SwingTradingScorer
        
        scorer = SwingTradingScorer(use_mock_data=True)
        
        # Score single stock (would need modification to scorer for single stock)
        # For now, return mock response
        return {
            "symbol": symbol.upper(),
            "final_score": 87.5,
            "component_scores": {
                "market_regime": 75.0,
                "sector_strength": 80.0,
                "relative_strength": 85.0,
                "liquidity": 90.0,
                "trend_quality": 88.0,
                "volume_confirmation": 82.0,
                "catalyst": 75.0,
                "risk_volatility": 85.0
            },
            "qualifies": True,
            "entry_type": "breakout",
            "suggested_stop_loss_pct": 4.0,
            "suggested_target_pct": 12.0,
            "risk_reward_ratio": 3.0,
            "timestamp": now_ist().isoformat()
        }
    except Exception as e:
        logger.error(f"Swing score failed for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Swing score failed: {str(e)}")


@app.get("/api/swing/base-formation/{symbol}")
def get_base_formation(symbol: str):
    """
    Analyze base formation for a specific symbol.
    Returns base type, duration, breakout levels, and quality metrics.
    """
    try:
        from data_platform.pipelines.base_formation_detector import BaseFormationDetector
        
        detector = BaseFormationDetector(use_mock_data=True)
        base = detector.detect_base(symbol.upper())
        
        if not base or not base.has_base:
            return {
                "symbol": symbol.upper(),
                "has_base": False,
                "message": "No valid base formation detected"
            }
        
        return {
            "symbol": base.symbol,
            "has_base": base.has_base,
            "base_type": base.base_type.value,
            "base_duration_days": base.base_duration_days,
            "base_high": base.base_high,
            "base_low": base.base_low,
            "base_range_pct": base.base_range_pct,
            "breakout_level": base.breakout_level,
            "breakdown_level": base.breakdown_level,
            "volume_dry_up": base.volume_dry_up,
            "breakout_probability": base.breakout_probability,
            "pattern_quality_score": base.pattern_quality_score,
            "measured_move_target": base.measured_move_target,
            "conservative_target": base.conservative_target,
            "aggressive_target": base.aggressive_target,
            "timestamp": base.timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Base formation analysis failed for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Base analysis failed: {str(e)}")


@app.get("/api/swing/relative-strength/{symbol}")
def get_relative_strength(symbol: str):
    """
    Get relative strength vs Nifty for a specific symbol.
    Returns 1M, 3M, 6M relative strength metrics.
    """
    try:
        from data_platform.pipelines.swing_trading_scorer import SwingTradingScorer
        
        scorer = SwingTradingScorer(use_mock_data=True)
        rs = scorer._calculate_relative_strength(symbol.upper())
        
        if not rs:
            return {
                "symbol": symbol.upper(),
                "error": "Could not calculate relative strength"
            }
        
        return {
            "symbol": rs.symbol,
            "nifty_return_1m_pct": rs.nifty_return_1m,
            "stock_return_1m_pct": rs.stock_return_1m,
            "relative_strength_1m_pct": rs.relative_strength_1m,
            "nifty_return_3m_pct": rs.nifty_return_3m,
            "stock_return_3m_pct": rs.stock_return_3m,
            "relative_strength_3m_pct": rs.relative_strength_3m,
            "nifty_return_6m_pct": rs.nifty_return_6m,
            "stock_return_6m_pct": rs.stock_return_6m,
            "relative_strength_6m_pct": rs.relative_strength_6m,
            "overall_rs_score": rs.overall_rs_score,
            "rs_rank": rs.rs_rank,
            "timestamp": rs.timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Relative strength failed for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RS calculation failed: {str(e)}")


# ---------------------------------------------------------------------------
# Long-Term Investing Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/long-term/portfolio")
def get_long_term_portfolio():
    """
    Build and return institutional long-term investing portfolio.
    Returns 15-25 high-conviction candidates with conviction-based sizing.
    """
    try:
        from data_platform.pipelines.long_term_watchlist import LongTermPortfolioBuilder
        
        builder = LongTermPortfolioBuilder(use_mock_data=True)
        result = builder.build_portfolio()
        summary = builder.get_portfolio_summary(result)
        
        return summary
    except Exception as e:
        logger.error(f"Long-term portfolio failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Long-term portfolio failed: {str(e)}")


@app.get("/api/long-term/score/{symbol}")
def get_long_term_score(symbol: str):
    """
    Get long-term investing score for a specific symbol.
    Returns component scores and final 0-100 score with investment thesis.
    """
    try:
        from data_platform.pipelines.long_term_investing_scorer import LongTermInvestingScorer
        from config.universe import NSE_UNIVERSE
        
        scorer = LongTermInvestingScorer(use_mock_data=True)
        
        # Find stock in universe
        stock = next((s for s in NSE_UNIVERSE if s["symbol"] == symbol.upper()), None)
        if not stock:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found in universe")
        
        score = scorer._score_stock(stock)
        
        return {
            "symbol": score.symbol,
            "name": score.name,
            "sector": score.sector,
            "final_score": round(score.final_score, 1),
            "component_scores": {
                "industry_growth": round(score.industry_growth_score, 1),
                "revenue_growth": round(score.revenue_growth_score, 1),
                "earnings_growth": round(score.earnings_growth_score, 1),
                "roe_roce": round(score.roe_roce_score, 1),
                "debt_quality": round(score.debt_quality_score, 1),
                "cash_flow": round(score.cash_flow_score, 1),
                "management_governance": round(score.management_governance_score, 1),
                "competitive_moat": round(score.competitive_moat_score, 1),
                "valuation": round(score.valuation_score, 1)
            },
            "qualifies": score.qualifies,
            "investment_thesis": score.investment_thesis,
            "key_risks": score.key_risks,
            "expected_10y_return_pct": round(score.expected_10y_return_pct, 1),
            "potential_multibagger": score.potential_multibagger,
            "conviction_level": score.conviction_level,
            "suggested_allocation_pct": round(score.suggested_allocation_pct, 1),
            "theme": score.industry_theme.theme.value if score.industry_theme and score.industry_theme.theme else None,
            "has_moat": score.moat.has_moat if score.moat else False,
            "timestamp": score.timestamp.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Long-term score failed for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Long-term score failed: {str(e)}")


@app.get("/api/long-term/financial-quality/{symbol}")
def get_financial_quality(symbol: str):
    """
    Get financial quality analysis for a specific symbol.
    Returns revenue/earnings growth, margins, ROE/ROCE, cash flow, debt metrics.
    """
    try:
        from data_platform.pipelines.long_term_investing_scorer import LongTermInvestingScorer
        
        scorer = LongTermInvestingScorer(use_mock_data=True)
        financial = scorer._analyze_financial_quality(symbol.upper())
        
        if not financial:
            return {
                "symbol": symbol.upper(),
                "error": "Could not analyze financial quality"
            }
        
        return {
            "symbol": financial.symbol,
            "revenue_cagr_3y_pct": round(financial.revenue_cagr_3y, 1),
            "revenue_cagr_5y_pct": round(financial.revenue_cagr_5y, 1),
            "earnings_cagr_3y_pct": round(financial.earnings_cagr_3y, 1),
            "earnings_cagr_5y_pct": round(financial.earnings_cagr_5y, 1),
            "revenue_growth_consistency": round(financial.revenue_growth_consistency, 1),
            "earnings_growth_consistency": round(financial.earnings_growth_consistency, 1),
            "operating_margin_avg_3y_pct": round(financial.operating_margin_avg_3y, 1),
            "operating_margin_trend": financial.operating_margin_trend,
            "net_margin_avg_3y_pct": round(financial.net_margin_avg_3y, 1),
            "net_margin_trend": financial.net_margin_trend,
            "roe_avg_3y_pct": round(financial.roe_avg_3y, 1),
            "roe_trend": financial.roe_trend,
            "roce_avg_3y_pct": round(financial.roce_avg_3y, 1),
            "roce_trend": financial.roce_trend,
            "fcf_margin_avg_3y_pct": round(financial.fcf_margin_avg_3y, 1),
            "fcf_conversion_pct": round(financial.fcf_conversion_pct, 1),
            "debt_to_equity": round(financial.debt_to_equity, 2),
            "interest_coverage_ratio": round(financial.interest_coverage_ratio, 1),
            "financial_quality_score": round(financial.financial_quality_score, 1),
            "timestamp": financial.timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Financial quality analysis failed for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Financial analysis failed: {str(e)}")


@app.get("/api/long-term/moat/{symbol}")
def get_moat_analysis(symbol: str):
    """
    Get competitive moat analysis for a specific symbol.
    Returns moat type, strength, durability, and competitive advantages.
    """
    try:
        from data_platform.pipelines.long_term_investing_scorer import LongTermInvestingScorer
        
        scorer = LongTermInvestingScorer(use_mock_data=True)
        moat = scorer._analyze_moat(symbol.upper())
        
        return {
            "symbol": moat.symbol,
            "has_moat": moat.has_moat,
            "moat_type": moat.moat_type.value if moat.moat_type else None,
            "moat_strength": round(moat.moat_strength, 1),
            "moat_durability": moat.moat_durability,
            "market_share_trend": moat.market_share_trend,
            "pricing_power": round(moat.pricing_power, 1),
            "customer_loyalty": round(moat.customer_loyalty, 1),
            "competitive_advantages": moat.competitive_advantages,
            "timestamp": moat.timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Moat analysis failed for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Moat analysis failed: {str(e)}")


@app.get("/api/long-term/valuation/{symbol}")
def get_valuation_analysis(symbol: str):
    """
    Get valuation analysis using GARP/GARV methodology for a specific symbol.
    Returns multiples, growth-adjusted metrics, and intrinsic value estimate.
    """
    try:
        from data_platform.pipelines.long_term_investing_scorer import LongTermInvestingScorer
        
        scorer = LongTermInvestingScorer(use_mock_data=True)
        valuation = scorer._analyze_valuation(symbol.upper())
        
        if not valuation:
            return {
                "symbol": symbol.upper(),
                "error": "Could not analyze valuation"
            }
        
        return {
            "symbol": valuation.symbol,
            "pe_ratio": round(valuation.pe_ratio, 1),
            "pe_historical_percentile": round(valuation.pe_historical_percentile, 1),
            "pb_ratio": round(valuation.pb_ratio, 1),
            "pb_historical_percentile": round(valuation.pb_historical_percentile, 1),
            "ev_ebitda": round(valuation.ev_ebitda, 1),
            "ev_ebitda_historical_percentile": round(valuation.ev_ebitda_historical_percentile, 1),
            "peg_ratio": round(valuation.peg_ratio, 2),
            "growth_adjusted_pe": round(valuation.growth_adjusted_pe, 1),
            "garv_score": round(valuation.garv_score, 1),
            "intrinsic_value_estimate": round(valuation.intrinsic_value_estimate, 1),
            "margin_of_safety_pct": round(valuation.margin_of_safety_pct, 1),
            "valuation_score": round(valuation.valuation_score, 1),
            "timestamp": valuation.timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Valuation analysis failed for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Valuation analysis failed: {str(e)}")


@app.get("/api/market/overview")
def get_market_overview():
    """Live market overview: indices + gold ETF via Upstox."""
    try:
        if not UPSTOX_OK:
            raise HTTPException(status_code=503, detail="Upstox client not available")
        indices = get_index_overview()
        status  = get_market_status()
        return {
            "indices": indices,
            "market_status": status,
            "timestamp": now_ist().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"market/overview failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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
def get_stock_history(
    symbol: str,
    days: int     = Query(default=180, le=730),
    interval: str = Query(default="1day", description="1minute|15minute|30minute|1hour|1day|1week|1month"),
):
    """Fetch OHLCV candle data from Upstox for charting."""
    if not UPSTOX_OK:
        raise HTTPException(status_code=503, detail="Upstox client not available")
    try:
        candles = get_candles(symbol.upper(), interval=interval, days=days)
        if not candles:
            # Fallback: return single latest price from DB
            latest = get_stock_price(symbol.upper())
            if not latest:
                return {"dates": [], "prices": [], "volumes": [], "source": "empty"}
            ts = latest.get("timestamp")
            p  = _to_float(latest.get("price"))
            return {
                "dates":      [ts.isoformat() if hasattr(ts, "isoformat") else str(ts or "")],
                "open":       [p], "high": [p], "low": [p], "prices": [p],
                "volumes":    [_to_int(latest.get("volume"))],
                "indicators": {}, "levels": {}, "source": "database_latest",
            }
        return _derive_history_from_upstox(candles)
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
def get_news(symbol: str, limit: int = Query(default=10, le=30)):
    """Live news headlines via Upstox — real articles, not RSS stubs."""
    if UPSTOX_OK:
        try:
            articles = get_stock_news(symbol.upper(), limit=limit)
            if articles:
                return articles
        except Exception as e:
            logger.warning(f"Upstox news failed for {symbol}: {e}")
    # Fallback to DB
    try:
        from database.connection import get_news_for_symbol
        return get_news_for_symbol(symbol.upper())
    except Exception:
        return []


# ===========================================================================
# UPSTOX LIVE DATA ENDPOINTS
# ===========================================================================

@app.get("/api/cockpit")
def api_cockpit():
    """
    Full cockpit data: indices, sectors, FII/DII, market status, holidays.
    Single endpoint for the homepage dashboard.
    """
    if not UPSTOX_OK:
        raise HTTPException(503, "Upstox client not available")
    try:
        return get_cockpit_data()
    except Exception as e:
        logger.error(f"cockpit failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.get("/api/sectors/live")
def api_sectors_live():
    """Sector-wise performance using Nifty sector indices (live)."""
    if not UPSTOX_OK:
        raise HTTPException(503, "Upstox client not available")
    try:
        return get_sector_overview()
    except Exception as e:
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------------
# FII / DII
# ---------------------------------------------------------------------------

@app.get("/api/market/fii")
def api_fii(date: str | None = Query(default=None, description="YYYY-MM-DD, defaults to yesterday")):
    """Real FII activity data from Upstox across all segments."""
    if not UPSTOX_OK:
        raise HTTPException(503, "Upstox client not available")
    try:
        return get_fii_activity(date_str=date)
    except Exception as e:
        logger.error(f"FII failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.get("/api/market/dii")
def api_dii(date: str | None = Query(default=None, description="YYYY-MM-DD, defaults to yesterday")):
    """Real DII cash segment activity from Upstox."""
    if not UPSTOX_OK:
        raise HTTPException(503, "Upstox client not available")
    try:
        return get_dii_activity(date_str=date)
    except Exception as e:
        logger.error(f"DII failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

@app.get("/api/options/expiries")
def api_option_expiries(
    instrument_key: str = Query(default="NSE_INDEX|Nifty 50", description="Underlying instrument key"),
):
    """Available expiry dates for an options underlying."""
    if not UPSTOX_OK:
        raise HTTPException(503, "Upstox client not available")
    try:
        return {"instrument_key": instrument_key, "expiries": get_option_expiries(instrument_key)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/options/chain")
def api_option_chain(
    instrument_key: str = Query(default="NSE_INDEX|Nifty 50"),
    expiry_date:    str = Query(..., description="YYYY-MM-DD"),
):
    """Full options chain with greeks (CE + PE per strike)."""
    if not UPSTOX_OK:
        raise HTTPException(503, "Upstox client not available")
    try:
        chain = get_option_chain(instrument_key, expiry_date)
        pcr   = compute_pcr_from_chain(chain)
        return {"instrument_key": instrument_key, "expiry": expiry_date, "chain": chain, "pcr_summary": pcr}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/options/oi")
def api_oi(
    instrument_key: str = Query(default="NSE_INDEX|Nifty 50"),
    expiry:         str = Query(..., description="YYYY-MM-DD"),
):
    """Open Interest summary across all strikes."""
    if not UPSTOX_OK:
        raise HTTPException(503, "Upstox client not available")
    try:
        return get_oi(instrument_key, expiry)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/options/pcr")
def api_pcr(
    instrument_key: str = Query(default="NSE_INDEX|Nifty 50"),
    expiry:         str = Query(...),
):
    """Put-Call Ratio for an expiry."""
    if not UPSTOX_OK:
        raise HTTPException(503, "Upstox client not available")
    try:
        # Try direct API, fall back to computing from chain
        result = get_pcr(instrument_key, expiry)
        if result is None:
            chain  = get_option_chain(instrument_key, expiry)
            result = compute_pcr_from_chain(chain)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/options/max-pain")
def api_max_pain(
    instrument_key:  str = Query(default="NSE_INDEX|Nifty 50"),
    expiry:          str = Query(...),
    bucket_interval: int = Query(default=100),
):
    """Max Pain level for an expiry."""
    if not UPSTOX_OK:
        raise HTTPException(503, "Upstox client not available")
    try:
        return get_max_pain(instrument_key, expiry, bucket_interval) or {"message": "No data"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/options/change-oi")
def api_change_oi(
    instrument_key: str = Query(default="NSE_INDEX|Nifty 50"),
    expiry:         str = Query(...),
    interval:       int = Query(default=1),
):
    """Change in Open Interest by strike."""
    if not UPSTOX_OK:
        raise HTTPException(503, "Upstox client not available")
    try:
        return get_change_oi(instrument_key, expiry, interval) or {"message": "No data"}
    except Exception as e:
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------------
# Smartlists
# ---------------------------------------------------------------------------

@app.get("/api/market/smartlist/futures")
def api_futures_smartlist(
    asset_type: str = Query(default="INDEX", description="INDEX | STOCK | COMMODITY"),
    category:   str = Query(default="PRICE_GAINERS",
                            description="PRICE_GAINERS|PRICE_LOSERS|MOST_ACTIVE|OI_GAINERS|OI_LOSERS|PREMIUM|DISCOUNT"),
):
    """Futures smartlist ranked by category."""
    if not UPSTOX_OK:
        raise HTTPException(503, "Upstox client not available")
    try:
        return get_futures_smartlist(asset_type, category)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/market/smartlist/options")
def api_options_smartlist(
    asset_type: str = Query(default="INDEX"),
    category:   str = Query(default="OI_GAINERS",
                            description="OI_GAINERS|OI_LOSERS|PRICE_GAINERS|PRICE_LOSERS|MOST_ACTIVE|IV_GAINERS|IV_LOSERS"),
):
    """Options smartlist ranked by category."""
    if not UPSTOX_OK:
        raise HTTPException(503, "Upstox client not available")
    try:
        return get_options_smartlist(asset_type, category)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/market/smartlist/mtf")
def api_mtf_smartlist():
    """MTF (Margin Trade Funding) eligible stocks smartlist."""
    if not UPSTOX_OK:
        raise HTTPException(503, "Upstox client not available")
    try:
        return get_mtf_smartlist()
    except Exception as e:
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------------
# Market Holidays
# ---------------------------------------------------------------------------

@app.get("/api/market/holidays")
def api_market_holidays():
    """Full year market holiday calendar for NSE/BSE/MCX."""
    if not UPSTOX_OK:
        raise HTTPException(503, "Upstox client not available")
    try:
        return {"holidays": get_market_holidays()}
    except Exception as e:
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------

@app.get("/api/fundamentals/{symbol}/profile")
def api_company_profile(symbol: str):
    """Company description, sector, market cap."""
    if not UPSTOX_OK:
        raise HTTPException(503, "Upstox client not available")
    try:
        data = get_company_profile(symbol.upper())
        if data is None:
            raise HTTPException(404, f"No profile found for {symbol}")
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/fundamentals/{symbol}/balance-sheet")
def api_balance_sheet(
    symbol:         str,
    statement_type: str = Query(default="Consolidated", description="Consolidated | Standalone"),
    period_type:    str = Query(default="Annual",       description="Annual | Quarterly"),
):
    if not UPSTOX_OK:
        raise HTTPException(503, "Upstox client not available")
    try:
        data = get_balance_sheet(symbol.upper(), statement_type, period_type)
        if data is None:
            raise HTTPException(404, f"No balance sheet for {symbol}")
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/fundamentals/{symbol}/income-statement")
def api_income_statement(
    symbol:         str,
    statement_type: str = Query(default="Consolidated"),
    period_type:    str = Query(default="Annual"),
):
    if not UPSTOX_OK:
        raise HTTPException(503, "Upstox client not available")
    try:
        data = get_income_statement(symbol.upper(), statement_type, period_type)
        if data is None:
            raise HTTPException(404, f"No income statement for {symbol}")
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/fundamentals/{symbol}/cash-flow")
def api_cash_flow(
    symbol:         str,
    statement_type: str = Query(default="Consolidated"),
    period_type:    str = Query(default="Annual"),
):
    if not UPSTOX_OK:
        raise HTTPException(503, "Upstox client not available")
    try:
        data = get_cash_flow(symbol.upper(), statement_type, period_type)
        if data is None:
            raise HTTPException(404, f"No cash flow for {symbol}")
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/fundamentals/{symbol}/key-ratios")
def api_key_ratios(symbol: str):
    """P/E, P/B, ROE, ROCE, EV/EBITDA vs sector."""
    if not UPSTOX_OK:
        raise HTTPException(503, "Upstox client not available")
    try:
        data = get_key_ratios(symbol.upper())
        if data is None:
            raise HTTPException(404, f"No key ratios for {symbol}")
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/fundamentals/{symbol}/share-holdings")
def api_share_holdings(symbol: str):
    """Promoter, FII, DII, retail shareholding history."""
    if not UPSTOX_OK:
        raise HTTPException(503, "Upstox client not available")
    try:
        data = get_share_holdings(symbol.upper())
        if data is None:
            raise HTTPException(404, f"No shareholding data for {symbol}")
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/fundamentals/{symbol}/corporate-actions")
def api_corporate_actions(symbol: str):
    """Dividends, bonus issues, stock splits, rights issues."""
    if not UPSTOX_OK:
        raise HTTPException(503, "Upstox client not available")
    try:
        data = get_corporate_actions(symbol.upper())
        if data is None:
            raise HTTPException(404, f"No corporate actions for {symbol}")
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/fundamentals/{symbol}/competitors")
def api_competitors(symbol: str):
    """Competitor companies for the given stock."""
    if not UPSTOX_OK:
        raise HTTPException(503, "Upstox client not available")
    try:
        data = get_competitors(symbol.upper())
        if data is None:
            raise HTTPException(404, f"No competitors for {symbol}")
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/fundamentals/{symbol}")
def api_full_fundamentals(symbol: str):
    """All fundamentals: profile + ratios + holdings + actions + competitors."""
    if not UPSTOX_OK:
        raise HTTPException(503, "Upstox client not available")
    try:
        return get_full_fundamentals(symbol.upper())
    except Exception as e:
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------------
# Live quotes — bulk endpoint for all 140 stocks
# ---------------------------------------------------------------------------

@app.get("/api/quotes/bulk")
def api_bulk_quotes(symbols: str = Query(..., description="Comma-separated NSE symbols")):
    """Live OHLC quotes for a list of NSE symbols."""
    if not UPSTOX_OK:
        raise HTTPException(503, "Upstox client not available")
    try:
        sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        return get_bulk_quotes(sym_list)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/quotes/ltp")
def api_ltp(symbols: str = Query(..., description="Comma-separated instrument keys or NSE symbols")):
    """Last traded prices only — lightest weight quote endpoint."""
    if not UPSTOX_OK:
        raise HTTPException(503, "Upstox client not available")
    try:
        from data_platform.upstox_client import get_instrument_map
        imap = get_instrument_map()
        keys = []
        for s in symbols.split(","):
            s = s.strip()
            if "|" in s:
                keys.append(s)
            elif s.upper() in imap:
                keys.append(imap[s.upper()])
        return get_ltp(keys)
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
