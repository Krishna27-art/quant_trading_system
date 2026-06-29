import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime

import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session

from data.upstox_historical import download_historical_candles
from database.db_sync import SessionLocal
from database.models import Prediction
from utils.logger import get_logger

logger = get_logger("outcome_resolver")


def resolve_unresolved_predictions():
    """
    Finds all predictions with outcome = 'OPEN' and resolves them:
    - Intraday resolves at 3:30 PM same day.
    - Swing resolves on 10th trading day.
    - Longterm resolves on 60th trading day.
    - Breach of SL or Target resolves immediately.
    """
    db: Session = SessionLocal()

    try:
        unresolved = db.query(Prediction).filter(Prediction.outcome == "OPEN").all()

        if not unresolved:
            logger.info("No unresolved predictions found.")
            return

        logger.info(f"Found {len(unresolved)} open predictions to resolve.")

        datetime.utcnow()

        for pred in unresolved:
            sym = pred.symbol
            timeframe = pred.timeframe
            gen_time = pred.generated_at

            logger.info(f"Resolving {timeframe} prediction for {sym} generated at {gen_time}")

            try:
                # 1. Fetch historical prices from the generation time onwards
                if timeframe == "INTRADAY":
                    # Fetch 1m candles for the generation day
                    df = download_historical_candles(
                        sym, interval="1minute", from_date=gen_time.strftime("%Y-%m-%d")
                    )
                    if df.empty:
                        df = yf.download(
                            f"{sym}.NS",
                            start=gen_time.strftime("%Y-%m-%d"),
                            interval="1m",
                            progress=False,
                        )
                        if not df.empty:
                            df = df.reset_index().rename(
                                columns={
                                    "Datetime": "timestamp",
                                    "Open": "open",
                                    "High": "high",
                                    "Low": "low",
                                    "Close": "close",
                                }
                            )
                            # localise / drop tz
                            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
                else:
                    # Daily candles for Swing / Long-term
                    df = download_historical_candles(
                        sym, interval="day", from_date=gen_time.strftime("%Y-%m-%d")
                    )
                    if df.empty:
                        df = yf.download(
                            f"{sym}.NS", start=gen_time.strftime("%Y-%m-%d"), progress=False
                        )
                        if not df.empty:
                            df = df.reset_index().rename(
                                columns={
                                    "Date": "timestamp",
                                    "Open": "open",
                                    "High": "high",
                                    "Low": "low",
                                    "Close": "close",
                                }
                            )
                            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

                if df.empty:
                    logger.warning(f"No price data found to resolve {sym}")
                    continue

                # Filter to only ticks after the prediction generation time
                df = df[df["timestamp"] >= gen_time].sort_values("timestamp").reset_index(drop=True)
                if df.empty:
                    logger.warning(f"No price data after prediction timestamp {gen_time} for {sym}")
                    continue

                resolved = False
                outcome = "OPEN"
                exit_price = None
                exit_time = None

                # Check for target/stop loss breach first
                for _, row in df.iterrows():
                    high = float(row["high"])
                    low = float(row["low"])
                    t_val = row["timestamp"]

                    if pred.direction == "BUY":
                        if low <= pred.stop_loss:
                            outcome = "LOSS"
                            exit_price = pred.stop_loss
                            exit_time = t_val
                            resolved = True
                            break
                        if high >= pred.target_price:
                            outcome = "WIN"
                            exit_price = pred.target_price
                            exit_time = t_val
                            resolved = True
                            break
                    else:  # SELL
                        if (
                            high <= pred.stop_loss
                        ):  # wait, for SELL stop loss is ABOVE entry, so high >= stop_loss
                            pass
                        if high >= pred.stop_loss:
                            outcome = "LOSS"
                            exit_price = pred.stop_loss
                            exit_time = t_val
                            resolved = True
                            break
                        if low <= pred.target_price:
                            outcome = "WIN"
                            exit_price = pred.target_price
                            exit_time = t_val
                            resolved = True
                            break

                if not resolved:
                    # Check for expiration/timeout
                    if timeframe == "INTRADAY":
                        # If we have reached the end of the day or are past 15:30 IST on the same day
                        # Note: Indian market closes at 15:30 IST
                        # Let's see if the latest tick is past 3:30 PM (or if the date has changed)
                        latest_tick_time = df["timestamp"].iloc[-1]
                        if (
                            latest_tick_time.date() > gen_time.date()
                            or (latest_tick_time.hour == 15 and latest_tick_time.minute >= 30)
                            or latest_tick_time.hour > 15
                        ):
                            outcome = "TIMEOUT"
                            exit_price = float(df["close"].iloc[-1])
                            exit_time = latest_tick_time
                            resolved = True

                    elif timeframe == "SWING":
                        # Resolves on 10th trading day (roughly 10 rows in daily candles)
                        if len(df) >= 10:
                            outcome = "TIMEOUT"
                            exit_price = float(df["close"].iloc[9])
                            exit_time = df["timestamp"].iloc[9]
                            resolved = True

                    elif timeframe == "LONGTERM":
                        # Resolves on 60th trading day
                        if len(df) >= 60:
                            outcome = "TIMEOUT"
                            exit_price = float(df["close"].iloc[59])
                            exit_time = df["timestamp"].iloc[59]
                            resolved = True

                if resolved:
                    # Calculate actual return percentage
                    if pred.direction == "BUY":
                        actual_ret = (exit_price - pred.entry_price) / pred.entry_price
                    else:
                        actual_ret = (pred.entry_price - exit_price) / pred.entry_price

                    pred.outcome = outcome
                    pred.exit_price = exit_price
                    pred.exit_time = exit_time
                    pred.actual_return_pct = actual_ret
                    pred.resolved_at = datetime.utcnow()

                    logger.info(
                        f"Resolved {sym} {timeframe} {pred.direction}: {outcome} at {exit_price} (Return: {actual_ret * 100:.2f}%)"
                    )

            except Exception as e:
                logger.error(f"Error resolving prediction {pred.id} for {sym}: {e}")

        db.commit()
        logger.info("Outcomes resolution run complete.")

    finally:
        db.close()


if __name__ == "__main__":
    resolve_unresolved_predictions()
