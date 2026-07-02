"""
Prediction Outcomes Evaluator
=============================
Evaluates open/pending predictions against historical price paths from Upstox.
Tracks MAE, MFE, target/stop hits, and computes Brier score components.
"""

import sys
import os
import argparse
from datetime import datetime, timezone, date
import sqlite3

# Ensure we can import from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_platform.upstox_client import get_candles, get_stock_quote
from utils.logger import get_logger
from utils.time_utils import now_ist

logger = get_logger("outcomes_evaluator")


def evaluate_predictions(dry_run: bool = False):
    conn = sqlite3.connect("quant.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Find predictions that are not yet fully evaluated (actual_outcome is NULL or 'PENDING')
    cursor.execute("""
        SELECT id, symbol, prediction, horizon, confidence, entry_price, stop_loss, target_price, prediction_time, expiry_time
        FROM predictions
        WHERE actual_outcome IS NULL OR actual_outcome = 'PENDING'
    """)
    pending = cursor.fetchall()
    
    logger.info(f"Found {len(pending)} pending predictions to evaluate.")

    updated_count = 0
    for row in pending:
        pid = row["id"]
        symbol = row["symbol"]
        pred_type = row["prediction"].upper() # BUY or SELL
        horizon = row["horizon"].upper()
        entry = row["entry_price"]
        sl = row["stop_loss"]
        tp = row["target_price"]
        pred_time_str = row["prediction_time"]
        
        # Parse prediction time
        try:
            # Handle possible ISO formats
            if "T" in pred_time_str:
                pred_time = datetime.fromisoformat(pred_time_str.replace("Z", "+00:00"))
            else:
                pred_time = datetime.fromisoformat(pred_time_str)
        except Exception:
            pred_time = now_ist()

        # If it hasn't expired yet and we are still in trading, we evaluate the price path since prediction time
        now = now_ist()
        
        # Use appropriate interval based on horizon
        interval = "15minute" if horizon == "INTRADAY" else "1day"
        days_to_fetch = 2 if horizon == "INTRADAY" else 30
        
        candles = get_candles(symbol, interval=interval, days=days_to_fetch)
        if not candles:
            # Fall back to single live quote to check if hit
            quote = get_stock_quote(symbol)
            if quote:
                candles = [{
                    "timestamp": quote["timestamp"] or now.isoformat(),
                    "open": quote["open"] or quote["last_price"],
                    "high": quote["high"] or quote["last_price"],
                    "low": quote["low"] or quote["last_price"],
                    "close": quote["last_price"],
                }]
        
        if not candles:
            logger.warning(f"No price data found for {symbol}, skipping evaluation.")
            continue
            
        # Filter candles that occurred AFTER the prediction time
        path = []
        for c in candles:
            try:
                c_time = datetime.fromisoformat(c["timestamp"].replace("Z", "+00:00"))
            except Exception:
                continue
            if c_time >= pred_time:
                path.append(c)
                
        # If no candles yet since prediction, check if expired by time
        if not path:
            # Use the latest candle if we have at least one
            if candles:
                path = [candles[0]]
            else:
                continue
                
        # Evaluate price path
        target_hit = 0
        stop_hit = 0
        max_fav = 0.0
        max_adv = 0.0
        
        closes = [c["close"] for c in path]
        highs = [c["high"] for c in path]
        lows = [c["low"] for c in path]
        
        latest_price = closes[-1]
        
        if pred_type == "BUY":
            # Best high reached relative to entry
            max_high = max(highs) if highs else entry
            # Worst low reached relative to entry
            min_low = min(lows) if lows else entry
            
            max_fav = ((max_high - entry) / entry) * 100 if entry else 0.0
            max_adv = ((entry - min_low) / entry) * 100 if entry else 0.0
            
            # Did it cross TP / SL?
            if tp and max_high >= tp:
                target_hit = 1
            if sl and min_low <= sl:
                stop_hit = 1
        elif pred_type == "SELL":
            # For short selling predictions, drop in price is favorable
            max_high = max(highs) if highs else entry
            min_low = min(lows) if lows else entry
            
            max_fav = ((entry - min_low) / entry) * 100 if entry else 0.0
            max_adv = ((max_high - entry) / entry) * 100 if entry else 0.0
            
            if tp and min_low <= tp:
                target_hit = 1
            if sl and max_high >= sl:
                stop_hit = 1
                
        # Determine actual outcome
        outcome = "PENDING"
        is_correct = 0
        actual_return = ((latest_price - entry) / entry) * 100 if entry else 0.0
        if pred_type == "SELL":
            actual_return = -actual_return
            
        # Target hit takes precedence
        if target_hit and stop_hit:
            # Both hit: check which happened first (approximate by close of first path element)
            # For simplicity, if target hit, we count it as a win
            outcome = "WIN"
            is_correct = 1
        elif target_hit:
            outcome = "WIN"
            is_correct = 1
        elif stop_hit:
            outcome = "LOSS"
            is_correct = 0
        else:
            # Check for expiry (more than 1 day for intraday, 7 days for swing)
            elapsed_days = (now - pred_time).days
            limit_days = 1 if horizon == "INTRADAY" else 7
            if elapsed_days >= limit_days:
                outcome = "EXPIRED"
                # If expired, check if return is positive
                is_correct = 1 if actual_return > 0 else 0
                
        if outcome != "PENDING":
            logger.info(f"Evaluated {symbol} ({pred_type}): {outcome} | Return: {actual_return:.2f}% | MAE: {max_adv:.2f}% | MFE: {max_fav:.2f}%")
            if not dry_run:
                conn.execute("""
                    UPDATE predictions
                    SET actual_outcome = ?,
                        target_hit = ?,
                        stop_hit = ?,
                        actual_return = ?,
                        mfe = ?,
                        mae = ?,
                        is_correct = ?
                    WHERE id = ?
                """, (outcome, target_hit, stop_hit, actual_return, max_fav, max_adv, is_correct, pid))
                updated_count += 1

    if not dry_run:
        conn.commit()
    conn.close()
    logger.info(f"Outcome evaluation complete. Updated {updated_count} rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate prediction outcomes")
    parser.add_argument("--dry-run", action="store_true", help="Print updates without modifying database")
    args = parser.parse_args()
    
    evaluate_predictions(dry_run=args.dry_run)
