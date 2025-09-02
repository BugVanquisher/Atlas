from .logging import setup_logging

setup_logging()

import json
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from redis.asyncio import from_url as redis_from_url

from .auth import extract_api_key
from .config import settings
from .metrics import (
    quota_rejections_total,
    rate_limit_rejections_total,
    requests_total,
    route_name_from_path,
    setup_metrics,
    tokens_used_total,
)
from .quota import ALLOWED_PRIORITIES, QuotaManager
from .rate_limit import RateLimiter
from .vllm_client import UpstreamClient

logger = logging.getLogger(__name__)

app = FastAPI(title="Atlas Gateway", version="0.1.0")
setup_metrics(app)

redis = redis_from_url(settings.REDIS_URL, decode_responses=False)
quota = QuotaManager(redis)
rl = RateLimiter(redis)
upstream = UpstreamClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    yield
    # shutdown
    await upstream.close()
    await redis.aclose()


app.router.lifespan_context = (
    lifespan  # use FastAPI lifespan (avoids deprecation warning)
)


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.post("/v1/admin/keys")
async def create_or_update_key(
    payload: dict,
    x_admin_key: str = Header(None),
):
    if x_admin_key != settings.ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Admin key invalid")

    api_key = payload.get("api_key")
    if not api_key:
        raise HTTPException(status_code=400, detail="api_key required")

    daily = int(payload.get("daily_limit", settings.DEFAULT_DAILY_LIMIT))
    monthly = int(payload.get("monthly_limit", settings.DEFAULT_MONTHLY_LIMIT))
    rate = float(payload.get("rate_per_sec", settings.DEFAULT_RATE_PER_SEC))
    burst = int(payload.get("burst", settings.DEFAULT_BURST))
    priority = str(payload.get("priority", "normal")).lower()
    if priority not in ALLOWED_PRIORITIES:
        raise HTTPException(
            status_code=400,
            detail=f"priority must be one of {sorted(ALLOWED_PRIORITIES)}",
        )

    await quota.set_limits(api_key, daily, monthly, rate, burst, priority)

    return {
        "ok": True,
        "api_key": api_key,
        "daily_limit": daily,
        "monthly_limit": monthly,
        "rate_per_sec": rate,
        "burst": burst,
        "priority": priority,
    }


@app.get("/v1/usage")
async def get_usage(api_key: str = Depends(extract_api_key)):
    used_d, used_m = await quota.get_usage(api_key)
    limits = await quota.get_limits(api_key)
    return {
        "daily_used": used_d,
        "daily_limit": limits["daily_limit"],
        "monthly_used": used_m,
        "monthly_limit": limits["monthly_limit"],
        "priority": limits.get("priority", "normal"),
    }


@app.api_route("/v1/{full_path:path}", methods=["GET", "POST"])
async def proxy(
    full_path: str,
    request: Request,
    api_key: str = Depends(extract_api_key),
):
    # Load limits (includes default priority)
    limits = await quota.get_limits(api_key)

    # Priority: header override if provided; else use key default
    req_priority = request.headers.get(
        "x-request-priority", limits.get("priority", "normal")
    ).lower()
    if req_priority not in ALLOWED_PRIORITIES:
        req_priority = limits.get("priority", "normal")

    route_name = route_name_from_path(full_path)
    requests_total.labels(
        api_key=api_key, priority=req_priority, route=route_name
    ).inc()

    # Rate limiting
    allowed = await rl.allow(api_key, limits["rate_per_sec"], limits["burst"])
    if not allowed:
        rate_limit_rejections_total.labels(api_key=api_key, priority=req_priority).inc()
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Forward upstream
    body = await request.body()
    upstream_resp = await upstream.forward(
        request.method, full_path, dict(request.headers), body
    )
    raw = upstream_resp.content

    # Token usage (if present)
    actual_tokens = 0
    if "application/json" in upstream_resp.headers.get("content-type", "") and raw:
        try:
            jr = json.loads(raw)
            if isinstance(jr, dict) and "usage" in jr:
                actual_tokens = int(jr["usage"].get("total_tokens", 0))
        except Exception:
            pass

    # Quota enforcement happens after upstream reply; record tokens metric
    if actual_tokens > 0:
        tokens_used_total.labels(api_key=api_key, priority=req_priority).inc(
            actual_tokens
        )

    try:
        if actual_tokens > 0:
            await quota.enforce_after_response_or_raise(api_key, actual_tokens, limits)
    except HTTPException as e:
        # Distinguish daily vs monthly for metrics
        scope = (
            "daily"
            if "Daily" in str(e.detail)
            else ("monthly" if "Monthly" in str(e.detail) else "unknown")
        )
        quota_rejections_total.labels(
            api_key=api_key, priority=req_priority, scope=scope
        ).inc()
        raise

    return Response(
        content=raw,
        status_code=upstream_resp.status_code,
        headers=dict(upstream_resp.headers),
        media_type=upstream_resp.headers.get("content-type"),
    )
