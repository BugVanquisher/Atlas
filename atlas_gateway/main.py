import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from redis.asyncio import from_url as redis_from_url
from .config import settings
from .metrics import setup_metrics
from .auth import extract_api_key
from .quota import QuotaManager
from .rate_limit import RateLimiter
from .vllm_client import UpstreamClient

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    yield
    # shutdown
    await upstream.close()
    await redis.aclose()

app = FastAPI(title="Atlas Gateway", version="0.1.0", lifespan=lifespan)
setup_metrics(app)

redis = redis_from_url(settings.REDIS_URL, decode_responses=False)
quota = QuotaManager(redis)
rl = RateLimiter(redis)
upstream = UpstreamClient()

@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.post("/v1/admin/keys")
async def create_or_update_key(payload: dict, request: Request):
    admin_key = request.headers.get("x-admin-key")
    if admin_key != settings.ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Admin key invalid")
    api_key = payload.get("api_key")
    if not api_key:
        raise HTTPException(status_code=400, detail="api_key required")
    await quota.set_limits(
        api_key,
        int(payload.get("daily_limit", settings.DEFAULT_DAILY_LIMIT)),
        int(payload.get("monthly_limit", settings.DEFAULT_MONTHLY_LIMIT)),
        float(payload.get("rate_per_sec", settings.DEFAULT_RATE_PER_SEC)),
        int(payload.get("burst", settings.DEFAULT_BURST)),
    )
    return {"ok": True, "api_key": api_key}


@app.get("/v1/usage")
async def get_usage(api_key: str = Depends(extract_api_key)):
    used_d, used_m = await quota.get_usage(api_key)
    limits = await quota.get_limits(api_key)
    return {
        "daily_used": used_d,
        "daily_limit": limits["daily_limit"],
        "monthly_used": used_m,
        "monthly_limit": limits["monthly_limit"],
    }


@app.api_route("/v1/{full_path:path}", methods=["GET", "POST"])
async def proxy(full_path: str, request: Request, api_key: str = Depends(extract_api_key)):
    limits = await quota.get_limits(api_key)
    allowed = await rl.allow(api_key, limits["rate_per_sec"], limits["burst"])
    if not allowed:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    body = await request.body()

    upstream_resp = await upstream.forward(request.method, full_path, dict(request.headers), body)
    raw = upstream_resp.content

    actual_tokens = 0
    if "application/json" in upstream_resp.headers.get("content-type", "") and raw:
        try:
            jr = json.loads(raw)
            if isinstance(jr, dict) and "usage" in jr:
                actual_tokens = int(jr["usage"].get("total_tokens", 0))
        except Exception:
            pass

    if actual_tokens > 0:
        await quota.enforce_after_response_or_raise(api_key, actual_tokens, limits)

    return Response(
        content=raw,
        status_code=upstream_resp.status_code,
        headers=dict(upstream_resp.headers),
        media_type=upstream_resp.headers.get("content-type"),
    )