from .logging import setup_logging
setup_logging()

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, HTTPException, Depends, Header
from redis.asyncio import from_url as redis_from_url
from starlette.responses import StreamingResponse

from .config import settings
from .metrics import (
    setup_metrics,
    requests_total,
    tokens_used_total,
    quota_rejections_total,
    rate_limit_rejections_total,
    route_name_from_path,
)
from .auth import extract_api_key
from .quota import QuotaManager, ALLOWED_PRIORITIES
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


app.router.lifespan_context = lifespan  # use FastAPI lifespan (avoids deprecation warning)


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
        raise HTTPException(status_code=400, detail=f"priority must be one of {sorted(ALLOWED_PRIORITIES)}")

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
    limits = await quota.get_limits(api_key)
    logger.info("incoming request")

    req_priority = request.headers.get("x-request-priority", limits.get("priority", "normal")).lower()
    if req_priority not in ALLOWED_PRIORITIES:
        req_priority = limits.get("priority", "normal")

    route_name = route_name_from_path(full_path)
    requests_total.labels(api_key=api_key, priority=req_priority, route=route_name).inc()

    allowed = await rl.allow(api_key, limits["rate_per_sec"], limits["burst"])
    if not allowed:
        rate_limit_rejections_total.labels(api_key=api_key, priority=req_priority).inc()
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    payload = None
    body_bytes = await request.body()
    if request.headers.get("content-type", "").startswith("application/json"):
        try:
            payload = json.loads(body_bytes) if body_bytes else {}
        except Exception:
            payload = {}

    is_stream = bool(payload and payload.get("stream") is True)
    logger.info(f"is_stream={is_stream}")
    # --- Streaming ---
    if is_stream:
        reservation = int(payload.get("max_tokens") or settings.DEFAULT_STREAM_RESERVATION or 0)

        if reservation > 0:
            used_d, used_m = await quota.get_usage(api_key)
            if used_d + reservation > limits["daily_limit"]:
                quota_rejections_total.labels(api_key=api_key, priority=req_priority, scope="daily").inc()
                raise HTTPException(status_code=429, detail="Daily quota would be exceeded")
            if used_m + reservation > limits["monthly_limit"]:
                quota_rejections_total.labels(api_key=api_key, priority=req_priority, scope="monthly").inc()
                raise HTTPException(status_code=429, detail="Monthly quota would be exceeded")

        if payload is not None and "stream" not in payload:
            payload["stream"] = True
            body_bytes = json.dumps(payload).encode("utf-8")

        async def iter_stream():
            nonlocal reservation
            status_ok = False
            try:
                async with upstream.stream(request.method, full_path, dict(request.headers), body_bytes) as resp:
                    async for chunk in resp.aiter_raw():
                        if chunk:
                            yield chunk
                    status_ok = (200 <= resp.status_code < 300)
            finally:
                if reservation > 0 and status_ok:
                    tokens_used_total.labels(api_key=api_key, priority=req_priority).inc(reservation)
                    await quota.add_tokens(api_key, reservation)

        return StreamingResponse(
            iter_stream(),
            status_code=200,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # --- Non-stream (existing path) ---
    upstream_resp = await upstream.forward(request.method, full_path, dict(request.headers), body_bytes)
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
        tokens_used_total.labels(api_key=api_key, priority=req_priority).inc(actual_tokens)

    try:
        if actual_tokens > 0:
            await quota.enforce_after_response_or_raise(api_key, actual_tokens, limits)
    except HTTPException as e:
        # Distinguish daily vs monthly for metrics
        scope = "daily" if "Daily" in str(e.detail) else ("monthly" if "Monthly" in str(e.detail) else "unknown")
        quota_rejections_total.labels(api_key=api_key, priority=req_priority, scope=scope).inc()
        raise

    return Response(
        content=raw,
        status_code=upstream_resp.status_code,
        headers=dict(upstream_resp.headers),
        media_type=upstream_resp.headers.get("content-type"),
    )