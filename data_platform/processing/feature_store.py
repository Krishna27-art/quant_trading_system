import json
import os
from datetime import datetime

import pandas as pd
import redis


class FeatureStore:
    """
    Point-in-Time (PIT) Feature Store for Research OS v2.
    Prevents lookahead bias by enforcing strict 'as_of_time' queries.

    Offline storage: Parquet files.
    Online storage: Redis key-value store.
    """

    def __init__(self, offline_dir: str = "data/features", redis_url: str | None = None):
        self.offline_dir = offline_dir
        os.makedirs(self.offline_dir, exist_ok=True)

        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            self.redis_client = redis.Redis.from_url(self.redis_url, decode_responses=True)
        except Exception:
            self.redis_client = None

    def push_offline(self, symbol: str, df: pd.DataFrame):
        """
        Store features in the offline Parquet store.
        df must contain a datetime index or a 'timestamp' column, plus feature columns.
        """
        if "timestamp" not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a 'timestamp' column or DatetimeIndex")

        df = df.copy()
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()

        file_path = os.path.join(self.offline_dir, f"{symbol.upper()}_features.parquet")

        if os.path.exists(file_path):
            existing_df = pd.read_parquet(file_path)
            # Combine and deduplicate on timestamp
            combined = pd.concat([existing_df, df]).drop_duplicates(
                subset=["timestamp"], keep="last"
            )
            combined.to_parquet(file_path, index=False)
        else:
            df.to_parquet(file_path, index=False)

    def push_online(self, symbol: str, feature_name: str, value: float, timestamp: datetime):
        """
        Store a feature value in the online Redis store for paper/live simulation.
        """
        if not self.redis_client:
            return

        key = f"feature:{symbol.upper()}:{feature_name}"
        score = timestamp.timestamp()
        val_str = json.dumps({"value": value, "timestamp": timestamp.isoformat()})
        self.redis_client.zadd(key, {val_str: score})

    def get_feature_pit(self, symbol: str, feature_name: str, as_of_time: datetime) -> float | None:
        """
        Get the latest feature value strictly on or before as_of_time (Time Travel query).
        """
        # First check online store if connected
        if self.redis_client:
            key = f"feature:{symbol.upper()}:{feature_name}"
            max_score = as_of_time.timestamp()
            # Get latest item before or at max_score
            results = self.redis_client.zrevrangebyscore(key, max_score, "-inf", start=0, num=1)
            if results:
                data = json.loads(results[0])
                return float(data["value"])

        # Fall back to offline Parquet store
        file_path = os.path.join(self.offline_dir, f"{symbol.upper()}_features.parquet")
        if not os.path.exists(file_path):
            return None

        try:
            df = pd.read_parquet(file_path)
            # Convert timestamp column to datetime if it isn't
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Filter strictly on or before as_of_time
            df_pit = df[df["timestamp"] <= as_of_time]
            if df_pit.empty:
                return None

            # Get latest record
            latest_record = df_pit.sort_values("timestamp").iloc[-1]
            if feature_name in latest_record:
                val = latest_record[feature_name]
                return float(val) if pd.notna(val) else None
        except Exception:
            pass

        return None

    def get_features_df_pit(
        self, symbol: str, feature_names: list[str], as_of_time: datetime
    ) -> dict[str, float | None]:
        """
        Query multiple features strictly as of a point in time.
        """
        return {name: self.get_feature_pit(symbol, name, as_of_time) for name in feature_names}
