"""
Signal Engine (Alpha Factory).
Generates predictions and exports them to disk/DB without executing trades.
"""

import os
from datetime import datetime
from typing import Any

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


class SignalEngine:
    def __init__(self, export_path: str = "./data/signals", export_format: str = "parquet"):
        self.export_path = export_path
        self.export_format = export_format.lower()
        self._signal_buffer: list[dict[str, Any]] = []

        if not os.path.exists(self.export_path):
            os.makedirs(self.export_path)

    def process_tick(
        self, timestamp: pd.Timestamp, symbol: str, features: dict[str, float], ml_predictor=None
    ):
        """
        Receives raw features for a symbol at a specific time, applies ML prediction,
        and buffers the raw signal score.
        """
        # If an ML predictor is provided, use it to ensemble features into a single score
        if ml_predictor:
            predicted_return = ml_predictor.predict(features)
        else:
            # Fallback: simple sum or heuristic if no model is loaded
            predicted_return = sum(features.values()) / (len(features) or 1)

        signal_record = {
            "timestamp": timestamp,
            "symbol": symbol,
            "predicted_return": predicted_return,
        }
        # Add raw features for explainability/debugging
        signal_record.update({f"feat_{k}": v for k, v in features.items()})

        self._signal_buffer.append(signal_record)

        # Flush if buffer gets large to manage memory
        if len(self._signal_buffer) > 10000:
            self.flush()

    def flush(self):
        """
        Writes the buffered signals to disk in the specified format.
        """
        if not self._signal_buffer:
            return

        df = pd.DataFrame(self._signal_buffer)

        # Generate a unique batch filename
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.export_format == "csv":
            filepath = os.path.join(self.export_path, f"signals_{batch_id}.csv")
            df.to_csv(filepath, index=False)
        else:
            # Default to parquet for speed and compression
            filepath = os.path.join(self.export_path, f"signals_{batch_id}.parquet")
            df.to_parquet(filepath, index=False)

        logger.info(f"Flushed {len(df)} signals to {filepath}")
        self._signal_buffer.clear()
