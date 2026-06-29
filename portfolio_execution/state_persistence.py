"""
State Persistence

Provides Redis-backed state persistence for crash recovery and distributed architecture.
Includes a bounded LRU cache for in-memory operations and heartbeat mechanisms
for primary/standby failover.
"""

import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any

from utils.logger import get_logger

# Optional Redis import
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = get_logger(__name__)


class LRUCache:
    """A bounded LRU cache for memory safety. Thread-safe."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: Any) -> Any | None:
        with self.lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: Any, value: Any) -> None:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

    def values(self) -> list[Any]:
        with self.lock:
            return list(self.cache.values())


@dataclass
class StatePersistenceConfig:
    """Configuration for state persistence."""

    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    use_redis: bool = os.getenv("USE_REDIS", "false").lower() == "true"
    instance_id: str = os.getenv("INSTANCE_ID", "primary_node")
    heartbeat_interval_seconds: int = 5
    standby_timeout_seconds: int = 15
    local_fallback_file: str = "logs/state_fallback.json"
    max_cache_size: int = 10000


class RedisStateStore:
    """
    Manages order, position, and session state persistence using Redis.
    Falls back to a local JSON file if Redis is unavailable.
    """

    def __init__(self, config: StatePersistenceConfig | None = None):
        self.config = config or StatePersistenceConfig()
        self.client: Any | None = None

        # Bounded in-memory caches
        self.order_cache = LRUCache(self.config.max_cache_size)
        self.position_cache = LRUCache(self.config.max_cache_size)
        self.session_cache: dict[str, Any] = {}

        if self.config.use_redis and REDIS_AVAILABLE:
            try:
                self.client = redis.from_url(self.config.redis_url, decode_responses=True)
                # Test connection
                self.client.ping()
                logger.info(f"Connected to Redis at {self.config.redis_url}")
            except Exception as e:
                logger.critical(
                    f"Failed to connect to Redis: {e}. Strict mode enabled, shutting down."
                )
                raise RuntimeError(
                    "Redis connection required for institutional state persistence"
                ) from e
        else:
            if not REDIS_AVAILABLE:
                logger.critical("Redis library not installed but required for state persistence.")
                raise RuntimeError("Redis library required")

        # Load initial state
        self._load_all_state()

    def _get_key(self, namespace: str, key: str) -> str:
        return f"quant:state:{namespace}:{key}"

    def save_order_state(self, order: Any) -> None:
        """Persist an order to Redis/Cache."""
        # Assume order can be serialized
        order_dict = (
            order
            if isinstance(order, dict)
            else (asdict(order) if hasattr(order, "__dataclass_fields__") else order.__dict__)
        )

        order_id = str(order_dict.get("order_id", "unknown"))
        self.order_cache.put(order_id, order_dict)

        if self.client:
            try:
                self.client.hset(
                    self._get_key("orders", "active"), order_id, json.dumps(order_dict)
                )
            except Exception as e:
                logger.critical(f"Redis save error for order {order_id}: {e}")
                raise RuntimeError("Distributed State Failure") from e
        else:
            pass  # Fallback to local cache

    def load_order_states(self) -> list[dict[str, Any]]:
        """Load all orders from cache/Redis."""
        if self.client:
            try:
                orders_json = self.client.hgetall(self._get_key("orders", "active"))
                orders = []
                for _, order_str in orders_json.items():
                    order_data = json.loads(order_str)
                    orders.append(order_data)
                    self.order_cache.put(str(order_data.get("order_id")), order_data)
                return orders
            except Exception as e:
                logger.error(f"Redis load error for orders: {e}")

        return self.order_cache.values()

    def save_position(self, position: Any) -> None:
        """Persist a position."""
        pos_dict = (
            position
            if isinstance(position, dict)
            else (
                asdict(position) if hasattr(position, "__dataclass_fields__") else position.__dict__
            )
        )
        symbol = str(pos_dict.get("symbol", "unknown"))

        self.position_cache.put(symbol, pos_dict)

        if self.client:
            try:
                self.client.hset(self._get_key("positions", "active"), symbol, json.dumps(pos_dict))
            except Exception as e:
                logger.critical(f"Redis save error for position {symbol}: {e}")
                raise RuntimeError("Distributed State Failure") from e
        else:
            pass  # Fallback to local cache

    def load_positions(self) -> list[dict[str, Any]]:
        """Load all positions."""
        if self.client:
            try:
                pos_json = self.client.hgetall(self._get_key("positions", "active"))
                positions = []
                for _, pos_str in pos_json.items():
                    pos_data = json.loads(pos_str)
                    positions.append(pos_data)
                    self.position_cache.put(str(pos_data.get("symbol")), pos_data)
                return positions
            except Exception as e:
                logger.error(f"Redis load error for positions: {e}")

        return self.position_cache.values()

    def save_session_state(self, state: dict[str, Any]) -> None:
        """Persist general session state."""
        self.session_cache.update(state)

        if self.client:
            try:
                self.client.set(self._get_key("session", "data"), json.dumps(self.session_cache))
            except Exception as e:
                logger.critical(f"Redis save error for session: {e}")
                raise RuntimeError("Distributed State Failure") from e
        else:
            pass  # Fallback to local cache

    def heartbeat(self) -> None:
        """Write a heartbeat to Redis to signal liveness for primary/standby architecture."""
        if self.client:
            try:
                key = f"quant:heartbeat:{self.config.instance_id}"
                self.client.setex(key, self.config.standby_timeout_seconds, str(time.time()))

                # Also assert leadership if primary
                if "primary" in self.config.instance_id:
                    self.client.setex(
                        "quant:leader", self.config.standby_timeout_seconds, self.config.instance_id
                    )
            except Exception as e:
                logger.error(f"Failed to write heartbeat to Redis: {e}")

    def is_primary(self) -> bool:
        """Check if this instance is the primary leader."""
        if not self.client:
            return True  # Standalone mode

        try:
            leader = self.client.get("quant:leader")
            # We are leader or there is no leader, so we claim it on next heartbeat
            return not leader or leader == self.config.instance_id
        except Exception:
            return True

    def _load_all_state(self) -> None:
        """Load state from Redis on startup."""
        if not self.client:
            pass
        self.load_order_states()
        self.load_positions()
