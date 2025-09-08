from typing import Any, Dict, Optional

from fastapi import HTTPException
from redis.asyncio import Redis
from redis.asyncio import from_url as redis_from_url

from .config import settings
from .utils import ymd_now

LIMITS_KEY = "limits:{api_key}"  # hash: daily_limit, monthly_limit, rate_per_sec, burst
USAGE_D_KEY = "usage:d:{api_key}:{ymd}"  # int counter for daily usage
USAGE_M_KEY = "usage:m:{api_key}:{ym}"  # int counter for monthly usage

ALLOWED_PRIORITIES = {"low", "normal", "high", "critical"}


def _get_str(d: Dict[Any, Any], key: str, default: str | None = None) -> str | None:
    v = d.get(key)
    if v is None:
        v = d.get(key.encode())
    if isinstance(v, bytes):
        v = v.decode()
    return v if v is not None else default


def _get_int(d: Dict[Any, Any], key: str, default: int) -> int:
    s = _get_str(d, key, None)
    try:
        return int(s) if s is not None else default
    except Exception:
        return default


def _get_float(d: Dict[Any, Any], key: str, default: float) -> float:
    s = _get_str(d, key, None)
    try:
        return float(s) if s is not None else default
    except Exception:
        return default


def _normalize_priority(p: str | None) -> str:
    if not p:
        return "normal"
    p = p.lower().strip()
    return p if p in ALLOWED_PRIORITIES else "normal"


class QuotaManager:
    def __init__(self, redis: Optional[Redis] = None):
        if redis is None:
            self.redis = redis_from_url(settings.REDIS_URL, decode_responses=False)
        else:
            self.redis = redis

    async def get_limits(self, api_key: str) -> dict:
        key = LIMITS_KEY.format(api_key=api_key)
        data = await self.redis.hgetall(key)
        if not data:
            return {
                "daily_limit": settings.DEFAULT_DAILY_LIMIT,
                "monthly_limit": settings.DEFAULT_MONTHLY_LIMIT,
                "rate_per_sec": settings.DEFAULT_RATE_PER_SEC,
                "burst": settings.DEFAULT_BURST,
                "priority": "normal",
            }
        return {
            "daily_limit": _get_int(data, "daily_limit", settings.DEFAULT_DAILY_LIMIT),
            "monthly_limit": _get_int(data, "monthly_limit", settings.DEFAULT_MONTHLY_LIMIT),
            "rate_per_sec": _get_float(data, "rate_per_sec", settings.DEFAULT_RATE_PER_SEC),
            "burst": _get_int(data, "burst", settings.DEFAULT_BURST),
            "priority": _normalize_priority(_get_str(data, "priority", "normal")),
        }

    async def set_limits(
        self,
        api_key: str,
        daily: int,
        monthly: int,
        rate_per_sec: float,
        burst: int,
        priority: str = "normal",
    ):
        key = LIMITS_KEY.format(api_key=api_key)
        await self.redis.hset(
            key,
            mapping={
                "daily_limit": daily,
                "monthly_limit": monthly,
                "rate_per_sec": rate_per_sec,
                "burst": burst,
                "priority": _normalize_priority(priority),
            },
        )

    async def get_usage(self, api_key: str) -> tuple[int, int]:
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
        pipe.expire(d_key, 60 * 60 * 24 * 40)  # ~40 days
        pipe.incrby(m_key, tokens)
        pipe.expire(m_key, 60 * 60 * 24 * 500)  # ~500 days
        await pipe.execute()

    async def reserve_tokens(self, api_key: str, tokens: int):
        """
        Reserve tokens for usage (same as add_tokens).
        """
        await self.add_tokens(api_key, tokens)

    async def refund_tokens(self, api_key: str, tokens: int):
        """
        Refund previously reserved tokens by decrementing usage.
        """
        ymd, ym = ymd_now()
        d_key = USAGE_D_KEY.format(api_key=api_key, ymd=ymd)
        m_key = USAGE_M_KEY.format(api_key=api_key, ym=ym)
        pipe = self.redis.pipeline()
        pipe.incrby(d_key, -tokens)
        pipe.expire(d_key, 60 * 60 * 24 * 40)  # ~40 days
        pipe.incrby(m_key, -tokens)
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
