import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session

from database.db_sync import SessionLocal
from database.models import Prediction
from utils.logger import get_logger

logger = get_logger("outcome_resolver")


# ---------------------------------------------------------------------------
# Data fetching — no dependency on data.upstox_historical (module missing)
# Falls back to yfinance with .NS suffix for NSE symbols.
# ---------------------------------------------------------------------------

def _fetch_intraday(symbol: str, from_date: str) -> pd.DataFrame:
    """Fetch 1-minute OHLCV candles for NSE symbol starting from from_date."""
    try:
        raw = yf.download(
            f"{symbol}.NS",
            start=from_date,
            interval="1m",
            progress=False,
            auto_adjust=True,
        )
        if raw.empty:
            return pd.DataFrame()

        # yfinance returns MultiIndex columns when auto_adjust=True sometimes
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw.reset_index().rename(
            columns={
                "Datetime": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        return df[["timestamp", "open", "high", "low", "close"]].copy()

    except Exception as exc:
        logger.error(f"yfinance intraday fetch failed for {symbol}: {exc}")
        return pd.DataFrame()


def _fetch_daily(symbol: str, from_date: str) -> pd.DataFrame:
    """Fetch daily OHLCV candles for NSE symbol starting from from_date."""
    try:
        raw = yf.download(
            f"{symbol}.NS",
            start=from_date,
            progress=False,
            auto_adjust=True,
        )
        if raw.empty:
            return pd.DataFrame()

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw.reset_index().rename(
            columns={
                "Date": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        return df[["timestamp", "open", "high", "low", "close"]].copy()

    except Exception as exc:
        logger.error(f"yfinance daily fetch failed for {symbol}: {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Resolution logic
# ---------------------------------------------------------------------------

# How many trading days before a prediction expires with TIMEOUT
TIMEFRAME_EXPIRY_DAYS = {
    "INTRADAY": 1,   # same day
    "SWING": 10,     # ~2 calendar weeks
    "LONGTERM": 60,  # ~3 calendar months
}


def _resolve_single(pred: Prediction) -> bool:
    """
    Resolve one open prediction.  Returns True if the prediction was resolved
    (outcome set), False if still OPEN (no data or not expired yet).

    Model columns used (from database/models.py + synonyms):
        pred.symbol          → symbol string  (e.g. "RELIANCE")
        pred.timeframe       → synonym for horizon  ("INTRADAY" / "SWING" / "LONGTERM")
        pred.generated_at    → synonym for prediction_time  (datetime, tz-naive)
        pred.direction       → synonym for prediction  ("BUY" / "SELL")
        pred.entry_price     → Numeric
        pred.stop_loss       → Numeric
        pred.target_price    → Numeric
        pred.outcome         → synonym for actual_outcome  (mutable)
        pred.target_hit      → Boolean
        pred.stop_hit        → Boolean
        pred.actual_return   → synonym for actual_return  (Numeric, fraction)
        pred.is_correct      → Boolean
    """

    sym = pred.symbol
    timeframe = (pred.timeframe or "SWING").upper()
    gen_time: datetime = pred.generated_at      # tz-naive IST assumed
    direction = (pred.direction or "BUY").upper()

    entry = float(pred.entry_price or 0)
    sl    = float(pred.stop_loss or 0)
    tp    = float(pred.target_price or 0)

    if entry == 0:
        logger.warning(f"Prediction {pred.id} ({sym}) has no entry_price — skipping")
        return False

    # ── fetch candles ───────────────────────────────────────────────────────
    from_date_str = gen_time.strftime("%Y-%m-%d")

    if timeframe == "INTRADAY":
        df = _fetch_intraday(sym, from_date_str)
    else:
        df = _fetch_daily(sym, from_date_str)

    if df.empty:
        logger.warning(f"No price data for {sym} from {from_date_str}")
        return False

    # Keep only candles at or after the prediction timestamp
    df = df[df["timestamp"] >= gen_time].sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        logger.warning(f"All candles pre-date prediction timestamp for {sym}")
        return False

    # ── bar-by-bar scan for SL / TP breach ─────────────────────────────────
    outcome   = "OPEN"
    exit_price: float | None = None
    exit_time: datetime | None = None

    for _, row in df.iterrows():
        bar_high = float(row["high"])
        bar_low  = float(row["low"])
        t_val    = row["timestamp"]

        if direction == "BUY":
            # SL breach: low touched below stop
            if sl > 0 and bar_low <= sl:
                outcome    = "LOSS"
                exit_price = sl
                exit_time  = t_val
                break
            # Target hit: high reached or exceeded target
            if tp > 0 and bar_high >= tp:
                outcome    = "WIN"
                exit_price = tp
                exit_time  = t_val
                break

        else:  # SELL / SHORT
            # For a short, SL is ABOVE entry price; high touching SL = loss
            if sl > 0 and bar_high >= sl:
                outcome    = "LOSS"
                exit_price = sl
                exit_time  = t_val
                break
            # Target is BELOW entry; low touching target = win
            if tp > 0 and bar_low <= tp:
                outcome    = "WIN"
                exit_price = tp
                exit_time  = t_val
                break

    # ── timeout / expiry check if no SL/TP hit ─────────────────────────────
    if outcome == "OPEN":
        expiry_bars = TIMEFRAME_EXPIRY_DAYS.get(timeframe, 10)

        if timeframe == "INTRADAY":
            latest = df["timestamp"].iloc[-1]
            market_closed = (
                latest.date() > gen_time.date()
                or latest.hour > 15
                or (latest.hour == 15 and latest.minute >= 30)
            )
            if market_closed:
                outcome    = "TIMEOUT"
                exit_price = float(df["close"].iloc[-1])
                exit_time  = latest

        else:
            # Swing / LongTerm: expire after N trading days
            if len(df) >= expiry_bars:
                outcome    = "TIMEOUT"
                exit_price = float(df["close"].iloc[expiry_bars - 1])
                exit_time  = df["timestamp"].iloc[expiry_bars - 1]

    if outcome == "OPEN":
        # Not yet expired; leave for next run
        return False

    # ── compute actual return ───────────────────────────────────────────────
    if direction == "BUY":
        actual_ret = (exit_price - entry) / entry
    else:
        actual_ret = (entry - exit_price) / entry

    # ── write back to model ─────────────────────────────────────────────────
    # Use only columns that exist in database/models.py
    pred.outcome        = outcome                          # → actual_outcome
    pred.actual_return  = round(actual_ret, 6)             # fraction, e.g. 0.023
    pred.target_hit     = outcome == "WIN"
    pred.stop_hit       = outcome == "LOSS"
    pred.is_correct     = outcome == "WIN"

    logger.info(
        f"Resolved {sym} | {timeframe} | {direction} | {outcome} | "
        f"entry={entry:.2f} exit={exit_price:.2f} ret={actual_ret * 100:.2f}%"
    )
    return True


# ---------------------------------------------------------------------------
# Win-rate summary helper
# ---------------------------------------------------------------------------

def _log_win_rate(db: Session) -> None:
    """Log overall and per-timeframe win percentages after a resolution run."""
    all_resolved = (
        db.query(Prediction)
        .filter(Prediction.actual_outcome.in_(["WIN", "LOSS", "TIMEOUT"]))
        .all()
    )
    if not all_resolved:
        return

    total = len(all_resolved)
    wins  = sum(1 for p in all_resolved if p.actual_outcome == "WIN")
    logger.info(f"Overall win rate: {wins}/{total} = {wins / total * 100:.1f}%")

    for tf in ("INTRADAY", "SWING", "LONGTERM"):
        subset = [p for p in all_resolved if (p.horizon or "").upper() == tf]
        if subset:
            w = sum(1 for p in subset if p.actual_outcome == "WIN")
            logger.info(
                f"  {tf}: {w}/{len(subset)} = {w / len(subset) * 100:.1f}% win rate"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def resolve_unresolved_predictions() -> None:
    """
    Fetch all OPEN predictions from DB, attempt resolution, commit results,
    then log aggregate win-rate statistics.
    """
    if SessionLocal is None:
        logger.error("Database session factory unavailable — check DATABASE_URL env var")
        return

    db: Session = SessionLocal()

    try:
        unresolved = (
            db.query(Prediction)
            .filter(Prediction.actual_outcome == "OPEN")
            .all()
        )

        if not unresolved:
            logger.info("No open predictions to resolve.")
            _log_win_rate(db)
            return

        logger.info(f"Resolving {len(unresolved)} open prediction(s)…")

        resolved_count = 0
        for pred in unresolved:
            try:
                if _resolve_single(pred):
                    resolved_count += 1
            except Exception as exc:
                logger.error(
                    f"Unhandled error resolving prediction {pred.id} "
                    f"({pred.symbol}): {exc}",
                    exc_info=True,
                )

        db.commit()
        logger.info(
            f"Resolution run complete: {resolved_count}/{len(unresolved)} resolved."
        )
        _log_win_rate(db)

    except Exception as exc:
        db.rollback()
        logger.error(f"Fatal error during resolution run: {exc}", exc_info=True)
        raise

    finally:
        db.close()


if __name__ == "__main__":
    resolve_unresolved_predictions()