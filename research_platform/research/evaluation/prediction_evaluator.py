from datetime import datetime
from typing import Any

from database.connection import execute_read, update_prediction
from utils.logger import get_logger

logger = get_logger("prediction_evaluator")


class PredictionEvaluator:
    """
    Evaluates trading predictions out-of-sample by walking forward through the tick database.
    Updates the predictions table with outcome, MFE, MAE, and hit status.
    """

    def __init__(self):
        self.batch_size = 500

    def _get_pending_predictions(self) -> list[dict[str, Any]]:
        """Fetch predictions that haven't been evaluated yet."""
        query = """
            SELECT id, symbol, prediction, entry_price, stop_loss, target_price,
                   prediction_time, expiry_time
            FROM predictions
            WHERE is_correct IS NULL
            AND expiry_time < CURRENT_TIMESTAMP
            ORDER BY prediction_time ASC
            LIMIT %s
        """
        rows = execute_read(query, (self.batch_size,))
        if not rows:
            return []

        columns = [
            "id",
            "symbol",
            "prediction",
            "entry_price",
            "stop_loss",
            "target_price",
            "prediction_time",
            "expiry_time",
        ]
        return [dict(zip(columns, row, strict=False)) for row in rows]

    def _get_price_path(self, symbol: str, start_time: datetime, end_time: datetime) -> list[float]:
        """Fetch the price path for a symbol between start and end times."""
        query = """
            SELECT price
            FROM stock_prices
            WHERE symbol = %s AND timestamp >= %s AND timestamp <= %s
            ORDER BY timestamp ASC
        """
        rows = execute_read(query, (symbol, start_time, end_time))
        return [float(row[0]) for row in rows] if rows else []

    def evaluate_pending(self):
        """Evaluate all pending predictions that have expired."""
        pending = self._get_pending_predictions()
        if not pending:
            logger.info("No pending predictions to evaluate.")
            return

        logger.info(f"Evaluating {len(pending)} pending predictions...")

        for pred in pending:
            try:
                self._evaluate_single(pred)
            except Exception as e:
                logger.error(f"Error evaluating prediction {pred['id']}: {e}")

        logger.info("Evaluation complete.")

    def _evaluate_single(self, pred: dict[str, Any]):
        symbol = pred["symbol"]
        direction = pred["prediction"].upper()
        entry = float(pred["entry_price"]) if pred["entry_price"] else None
        stop = float(pred["stop_loss"]) if pred["stop_loss"] else None
        target = float(pred["target_price"]) if pred["target_price"] else None

        # We need an entry price to evaluate
        if not entry:
            update_prediction(
                pred["id"], {"is_correct": False, "reason": "No entry price provided"}
            )
            return

        path = self._get_price_path(symbol, pred["prediction_time"], pred["expiry_time"])

        if not path:
            update_prediction(
                pred["id"], {"is_correct": False, "reason": "No price data found in horizon"}
            )
            return

        high_price = max(path)
        low_price = min(path)
        final_price = path[-1]

        target_hit = False
        stop_hit = False
        mfe = 0.0
        mae = 0.0
        actual_return = 0.0

        # Calculate for LONG
        if direction in ("BUY", "LONG"):
            mfe = (high_price - entry) / entry
            mae = (low_price - entry) / entry
            actual_return = (final_price - entry) / entry

            # Did it hit stop or target first? We'd have to walk the path to know order
            for price in path:
                if stop and price <= stop:
                    stop_hit = True
                    actual_return = (stop - entry) / entry
                    break
                if target and price >= target:
                    target_hit = True
                    actual_return = (target - entry) / entry
                    break

        # Calculate for SHORT
        elif direction in ("SELL", "SHORT"):
            mfe = (entry - low_price) / entry
            mae = (entry - high_price) / entry
            actual_return = (entry - final_price) / entry

            for price in path:
                if stop and price >= stop:
                    stop_hit = True
                    actual_return = (entry - stop) / entry
                    break
                if target and price <= target:
                    target_hit = True
                    actual_return = (entry - target) / entry
                    break

        is_correct = target_hit or (not stop_hit and actual_return > 0)
        actual_outcome = "PROFIT" if is_correct else "LOSS"

        updates = {
            "target_hit": target_hit,
            "stop_hit": stop_hit,
            "mfe": mfe,
            "mae": mae,
            "actual_return": actual_return,
            "is_correct": is_correct,
            "actual_outcome": actual_outcome,
        }

        update_prediction(pred["id"], updates)
        logger.debug(
            f"Evaluated {symbol} {direction}: {actual_outcome} (Ret: {actual_return * 100:.2f}%)"
        )


if __name__ == "__main__":
    evaluator = PredictionEvaluator()
    evaluator.evaluate_pending()
