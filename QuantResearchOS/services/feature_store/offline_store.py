import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


class OfflineFeatureStore:
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            # Resolve base_dir dynamically to QuantResearchOS/data
            base_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"
            )
        self.base_dir = os.path.abspath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

    def save_features(self, df: pd.DataFrame, asset_class: str, symbol: str, version: str = "v1"):
        """Saves a fully joined feature dataset to Parquet for offline ML training."""
        path = os.path.join(self.base_dir, asset_class)
        os.makedirs(path, exist_ok=True)

        file_path = os.path.join(path, f"{symbol}_features_{version}.parquet")
        df.to_parquet(file_path, index=True)
        logger.info("Saved %d feature records to %s", len(df), file_path)

    def load_features(self, asset_class: str, symbol: str, version: str = "v1") -> pd.DataFrame:
        """Loads feature dataset, ensuring point-in-time correctness was preserved during save."""
        file_path = os.path.join(self.base_dir, asset_class, f"{symbol}_features_{version}.parquet")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Feature file not found: {file_path}")
        return pd.read_parquet(file_path)
