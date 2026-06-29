import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class OnlineFeatureStore:
    def __init__(self, redis_client, ttl: int = 3600):
        """
        Args:
            redis_client: An async Redis client instance.
            ttl: Time-to-live in seconds for cached features (default 3600).
        """
        self.redis = redis_client
        self.ttl = ttl

    async def set_features(self, symbol: str, features: dict[str, Any]):
        """Pushes the latest computed features to Redis for low-latency inference."""
        key = f"features:online:{symbol}"
        try:
            # Store as JSON string in Redis with TTL
            await self.redis.set(key, json.dumps(features), ex=self.ttl)
        except Exception as e:
            logger.error("Redis SET failed for %s: %s", key, e)
            raise

    async def get_features(self, symbol: str) -> dict[str, Any] | None:
        """Retrieves the latest features for immediate model prediction."""
        key = f"features:online:{symbol}"
        try:
            data = await self.redis.get(key)
        except Exception as e:
            logger.error("Redis GET failed for %s: %s", key, e)
            raise
        if data:
            return json.loads(data)
        return None
