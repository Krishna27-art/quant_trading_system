import asyncio
import json
import time
from collections.abc import Callable
from typing import Any

from utils.logger import get_logger

logger = get_logger("event_bus")


class EventBus:
    """
    Deterministic async Event Bus using Redis Pub/Sub or local asyncio queues.
    Guarantees monotonic sequence IDs for strict ordering.
    """

    def __init__(self, use_redis: bool = False, redis_url: str = "redis://localhost:6379/0"):
        self.use_redis = use_redis
        self.redis_url = redis_url
        self.redis_client = None
        self.pubsub = None
        self._local_queues: dict[str, asyncio.Queue] = {}
        self._handlers: dict[str, list] = {}
        self._global_sequence: int = int(time.time() * 1000)  # Epoch ms start
        self._lock = asyncio.Lock()

    async def connect(self):
        """Connect to Redis if enabled."""
        if self.use_redis:
            try:
                import redis.asyncio as redis

                self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
                self.pubsub = self.redis_client.pubsub()
                await self.pubsub.psubscribe("quant:events:*")
                asyncio.create_task(self._redis_listener())
                logger.info("Connected to Redis Event Bus")
            except Exception as e:
                logger.error(f"Failed to connect to Redis Event Bus: {e}. Falling back to local.")
                self.use_redis = False

    async def _get_next_sequence(self) -> int:
        async with self._lock:
            self._global_sequence += 1
            return self._global_sequence

    async def publish(self, topic: str, event_type: str, payload: dict[str, Any]):
        """Publish an event to a topic."""
        seq = await self._get_next_sequence()
        message = {"seq": seq, "type": event_type, "timestamp": time.time(), "payload": payload}

        if self.use_redis and self.redis_client:
            await self.redis_client.publish(f"quant:events:{topic}", json.dumps(message))
        else:
            if topic not in self._local_queues:
                self._local_queues[topic] = asyncio.Queue()
            await self._local_queues[topic].put(message)
            asyncio.create_task(self._process_local(topic))

    def subscribe(self, topic: str, handler: Callable):
        """Subscribe a handler to a topic."""
        if topic not in self._handlers:
            self._handlers[topic] = []
        self._handlers[topic].append(handler)

    async def _process_local(self, topic: str):
        if topic in self._local_queues:
            msg = await self._local_queues[topic].get()
            await self._dispatch(topic, msg)
            self._local_queues[topic].task_done()

    async def _redis_listener(self):
        if not self.pubsub:
            return
        async for message in self.pubsub.listen():
            if message["type"] == "pmessage":
                channel = message["channel"].split(":")[-1]
                data = json.loads(message["data"])
                await self._dispatch(channel, data)

    async def _dispatch(self, topic: str, message: dict[str, Any]):
        if topic in self._handlers:
            for handler in self._handlers[topic]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message)
                    else:
                        handler(message)
                except Exception as e:
                    logger.error(f"Event handler error on topic {topic}: {e}")


# Global singleton
event_bus = EventBus()
