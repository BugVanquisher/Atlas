from redis.asyncio import Redis
from fastapi import HTTPException
from .config import settings
from .utils import ymd_now

LIMITS_KEY = "limits:{api_key}"              # hash: daily_limit, monthly_limit, rate_per_sec, burst
USAGE_D_KEY = "usage:d:{api_key}:{ymd}"     # int counter for daily usage
USAGE_M_KEY = "usage:m:{api_key}:{ym}"      # int counter for monthly usage


class QuotaManager:
    def __init__(self, redis: Redis):
        self.redis = redis

    async def get_limits(self, api_key: str):
        key = LIMITS_KEY.format(api_key=api_key)
        data = await self.redis.hgetall(key)
        if not data:
            # Defaults if key not configured
            return {
                "daily_limit": settings.DEFAULT_DAILY_LIMIT,
                "monthly_limit": settings.DEFAULT_MONTHLY_LIMIT,
                "rate_per_sec": settings.DEFAULT_RATE_PER_SEC,
                "burst": settings.DEFAULT_BURST,
            }

        def to_int(v, d): return int(v) if v is not None else d
        def to_float(v, d): return float(v) if v is not None else d

        return {
            "daily_limit": to_int(data.get(b"daily_limit"), settings.DEFAULT_DAILY_LIMIT),
            "monthly_limit": to_int(data.get(b"monthly_limit"), settings.DEFAULT_MONTHLY_LIMIT),
            "rate_per_sec": to_float(data.get(b"rate_per_sec"), settings.DEFAULT_RATE_PER_SEC),
            "burst": to_int(data.get(b"burst"), settings.DEFAULT_BURST),
        }

    async def set_limits(self, api_key: str, daily: int, monthly: int, rate_per_sec: float, burst: int):
        key = LIMITS_KEY.format(api_key=api_key)
        await self.redis.hset(key, mapping={
            "daily_limit": daily,
            "monthly_limit": monthly,
            "rate_per_sec": rate_per_sec,
            "burst": burst,
        })

    async def get_usage(self, api_key: str):
        ymd, ym = ymd_now()
        d = await self.redis.get(USAGE_D_KEY.format(api_key=api_key, ymd=ymd))
        m = await self.redis.get(USAGE_M_KEY.format(api_key=api_key, ym=ym))
        return int(d or 0), int(m or 0)

    async def add_tokens(self, api_key: str, tokens: int):
        ymd, ym = ymd_now()
        d_key = USAGE_D_KEY.format(api_key=api_key, ymd=ymd)
        m_key = USAGE_M_KEY.format(api_key=api_key, ym=ym)
        pipe = self.redis.pipeline()
        pipe.incrby(d_key, tokens)
        pipe.expire(d_key, 60 * 60 * 24 * 40)   # ~40 days
        pipe.incrby(m_key, tokens)
        pipe.expire(m_key, 60 * 60 * 24 * 500)  # ~500 days
        await pipe.execute()

    async def enforce_after_response_or_raise(self, api_key: str, actual_tokens: int, limits: dict):
        used_d, used_m = await self.get_usage(api_key)
        if used_d + actual_tokens > limits["daily_limit"]:
            await self.add_tokens(api_key, actual_tokens)
            raise HTTPException(status_code=429, detail="Daily quota exceeded")
        if used_m + actual_tokens > limits["monthly_limit"]:
            await self.add_tokens(api_key, actual_tokens)
            raise HTTPException(status_code=429, detail="Monthly quota exceeded")
        await self.add_tokens(api_key, actual_tokens)