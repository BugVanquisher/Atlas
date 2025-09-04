import time

from redis.asyncio import Redis

# Lua script for token bucket
_LUA = """
local key      = KEYS[1]
local rate     = tonumber(ARGV[1])  -- tokens/sec
local capacity = tonumber(ARGV[2])  -- bucket size
local now      = tonumber(ARGV[3])  -- current time
local need     = tonumber(ARGV[4])  -- tokens needed

local vals = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(vals[1]) or capacity
local ts     = tonumber(vals[2]) or now

-- Refill
local delta = math.max(0, now - ts)
tokens = math.min(capacity, tokens + delta * rate)

local allowed = 0
if tokens >= need then
  tokens = tokens - need
  allowed = 1
end

redis.call('HMSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, math.ceil((capacity / rate) * 2))

return allowed
"""


class RateLimiter:
    def __init__(self, redis: Redis):
        self.redis = redis
        self._sha = None

    async def _ensure_script(self):
        if not self._sha:
            self._sha = await self.redis.script_load(_LUA)

    async def allow(self, key: str, rate_per_sec: float, burst: int) -> bool:
        await self._ensure_script()
        now = int(time.time())
        res = await self.redis.evalsha(
            self._sha, 1, f"rl:{key}", rate_per_sec, burst, now, 1
        )
        return res == 1
