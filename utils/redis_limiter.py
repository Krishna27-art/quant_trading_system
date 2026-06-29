import os
import time

import redis.asyncio as redis

from utils.logger import get_logger

logger = get_logger("redis_limiter")

# Fallback to local memory if Redis is unavailable
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
except Exception as e:
    logger.warning(f"Failed to connect to Redis: {e}")
    redis_client = None

LUA_RATE_LIMIT = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local requested = tonumber(ARGV[4])

local bucket = redis.call("HMGET", key, "tokens", "last_update")
local tokens = tonumber(bucket[1])
local last_update = tonumber(bucket[2])

if not tokens then
    tokens = capacity
    last_update = now
end

local elapsed = math.max(0, now - last_update)
tokens = math.min(capacity, tokens + (elapsed * refill_rate))

if tokens >= requested then
    tokens = tokens - requested
    redis.call("HMSET", key, "tokens", tokens, "last_update", now)
    redis.call("EXPIRE", key, 3600)
    return 1
end
return 0
"""


async def check_rate_limit(client_id: str, capacity: int = 60, refill_rate: float = 1.0) -> bool:
    if not redis_client:
        return True  # Fallback if no redis
    try:
        now = time.time()
        result = await redis_client.eval(
            LUA_RATE_LIMIT, 1, f"ratelimit:{client_id}", capacity, refill_rate, now, 1
        )
        return result == 1
    except Exception as e:
        logger.error(f"Redis rate limit error: {e}")
        return True  # Fail open


async def is_circuit_breaker_open(route: str) -> bool:
    if not redis_client:
        return False
    try:
        state = await redis_client.get(f"circuit_breaker:{route}:state")
        return state == "open"
    except Exception:
        return False


async def record_circuit_failure(
    route: str, failure_threshold: int = 5, recovery_timeout: int = 60
):
    if not redis_client:
        return
    try:
        key = f"circuit_breaker:{route}:failures"
        failures = await redis_client.incr(key)
        if failures == 1:
            await redis_client.expire(key, recovery_timeout)
        if failures >= failure_threshold:
            await redis_client.setex(f"circuit_breaker:{route}:state", recovery_timeout, "open")
    except Exception:
        pass
