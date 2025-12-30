import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response

from .auth import extract_api_key
from .config import settings
from .logging import setup_logging
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
from .sentinel_client import SentinelClient, Verdict
from .streaming import StreamingHandler  # New enhanced streaming handler
from .vllm_client import UpstreamClient

try:
    from forecasting.metrics_collector import ForecastingIntegration

    from .forecasting_api import forecasting_router

    FORECASTING_ENABLED = True
except ImportError:
    # Forecasting modules not available, disable forecasting features
    ForecastingIntegration = None
    forecasting_router = None
    FORECASTING_ENABLED = False


setup_logging()

X_ADMIN_HEADER = Header(None)
API_KEY_DEP = Depends(extract_api_key)

USE_FAKEREDIS = os.getenv("USE_FAKEREDIS", "0") == "1"

if USE_FAKEREDIS:
    import fakeredis

    redis = fakeredis.FakeAsyncRedis()
else:
    from redis.asyncio import from_url as redis_from_url

    redis = redis_from_url(settings.REDIS_URL, decode_responses=False)

logger = logging.getLogger(__name__)

app = FastAPI(title="Atlas Gateway", version="0.1.0")
setup_metrics(app)

quota = QuotaManager(redis)
rl = RateLimiter(redis)
upstream = UpstreamClient()

# Initialize Sentinel client if enabled
sentinel: SentinelClient | None = None
if settings.SENTINEL_ENABLED:
    sentinel = SentinelClient()
    logger.info(f"Sentinel integration enabled: {settings.SENTINEL_URL}")

# Initialize forecasting integration
if FORECASTING_ENABLED:
    forecasting = ForecastingIntegration(redis)
    # Include the forecasting router
    app.include_router(forecasting_router)
else:
    forecasting = None

# Initialize enhanced streaming handler
streaming_handler = StreamingHandler(
    quota_manager=quota,
    upstream_client=upstream,
    metrics=type(
        "Metrics",
        (),
        {"quota_rejections_total": quota_rejections_total, "tokens_used_total": tokens_used_total},
    )(),
    forecasting_integration=forecasting,  # Pass forecasting integration
)


@asynccontextmanager
async def lifespan_context(app: FastAPI):
    # startup
    yield
    # shutdown
    if sentinel:
        await sentinel.close()
    await upstream.close()
    await redis.aclose()


app.router.lifespan_context = lifespan_context  # use FastAPI lifespan (avoids deprecation warning)


@app.post("/v1/admin/keys")
async def create_or_update_key(
    payload: dict,
    x_admin_key: str = X_ADMIN_HEADER,
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
async def get_usage(api_key: str = API_KEY_DEP):
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
    api_key: str = API_KEY_DEP,
):
    limits = await quota.get_limits(api_key)
    logger.info(f"Incoming request to /{full_path}")

    # Priority: header override if provided; else use key default
    req_priority = request.headers.get(
        "x-request-priority", limits.get("priority", "normal")
    ).lower()
    if req_priority not in ALLOWED_PRIORITIES:
        req_priority = limits.get("priority", "normal")

    route_name = route_name_from_path(full_path)
    requests_total.labels(api_key=api_key, priority=req_priority, route=route_name).inc()

    # Rate limiting check
    allowed = await rl.allow(api_key, limits["rate_per_sec"], limits["burst"])
    if not allowed:
        rate_limit_rejections_total.labels(api_key=api_key, priority=req_priority).inc()
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Parse request body
    payload = None
    body_bytes = await request.body()
    if request.headers.get("content-type", "").startswith("application/json"):
        try:
            payload = json.loads(body_bytes) if body_bytes else {}
        except Exception:
            payload = {}

    # Extract model from request if available
    model = payload.get("model") if payload else None

    is_stream = bool(payload and payload.get("stream") is True)
    logger.info(f"Processing request: stream={is_stream}, priority={req_priority}")

    # --- Enhanced Streaming Path ---
    if is_stream:
        try:
            response = await streaming_handler.handle_streaming_request(
                request=request,
                full_path=full_path,
                api_key=api_key,
                limits=limits,
                req_priority=req_priority,
                payload=payload,
                body_bytes=body_bytes,
            )

            # Record streaming metrics after successful response
            # Note: Token usage will be recorded in the streaming handler cleanup
            return response

        except HTTPException:
            # HTTPExceptions are properly formatted, re-raise them
            raise
        except Exception as e:
            logger.error(f"Unexpected error in streaming handler: {e}")
            raise HTTPException(status_code=502, detail="Internal streaming error")

    # --- Non-streaming Path (enhanced with forecasting metrics) ---
    upstream_resp = await upstream.forward(
        request.method, full_path, dict(request.headers), body_bytes
    )

    raw = upstream_resp.content
    response_json = None

    # Token usage parsing for non-streaming
    actual_tokens = 0
    if "application/json" in upstream_resp.headers.get("content-type", "") and raw:
        try:
            response_json = json.loads(raw)
            if isinstance(response_json, dict) and "usage" in response_json:
                actual_tokens = int(response_json["usage"].get("total_tokens", 0))
        except Exception:
            pass

    # --- Sentinel Safety Supervision ---
    if sentinel and response_json:
        try:
            # Extract the LLM output for supervision
            llm_output = ""
            if "choices" in response_json and response_json["choices"]:
                choice = response_json["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    llm_output = choice["message"]["content"] or ""
                elif "text" in choice:
                    llm_output = choice["text"] or ""

            if llm_output:
                # Get the original prompt from the request
                original_prompt = ""
                if payload:
                    if "messages" in payload and payload["messages"]:
                        # Chat completion format
                        for msg in payload["messages"]:
                            if msg.get("role") == "user":
                                original_prompt = msg.get("content", "")
                                break
                    elif "prompt" in payload:
                        # Completion format
                        original_prompt = payload["prompt"]

                # Call Sentinel for supervision
                supervision_result = await sentinel.supervise(
                    prompt=original_prompt,
                    draft=llm_output,
                    context={"api_key": api_key, "model": model, "route": route_name},
                )

                # Track Tier 3 invocations for safety compute budget
                if supervision_result.used_deep_ml:
                    await quota.add_safety_tier3_invocation(api_key)
                    logger.info(f"Tier 3 invocation recorded for {api_key}")

                # Handle supervision verdict
                if supervision_result.is_blocked:
                    logger.warning(f"Request blocked by Sentinel: {supervision_result.reasons}")
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "error": "Content blocked by safety filter",
                            "reasons": supervision_result.reasons,
                            "confidence": supervision_result.confidence,
                        },
                    )

                # If FIX verdict, replace the content with redacted version
                if supervision_result.verdict == Verdict.FIX and supervision_result.fixed_output:
                    logger.info(f"Content redacted by Sentinel: {supervision_result.reasons}")
                    # Update the response with redacted content
                    if "choices" in response_json and response_json["choices"]:
                        if "message" in response_json["choices"][0]:
                            response_json["choices"][0]["message"][
                                "content"
                            ] = supervision_result.fixed_output
                        elif "text" in response_json["choices"][0]:
                            response_json["choices"][0]["text"] = supervision_result.fixed_output
                    # Re-encode the modified response
                    raw = json.dumps(response_json).encode()

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Sentinel supervision error: {e}")
            if not settings.SENTINEL_FAIL_OPEN:
                raise HTTPException(status_code=503, detail="Safety supervision unavailable")
            # Fail-open: continue without blocking

    # Record token usage metrics
    if actual_tokens > 0:
        tokens_used_total.labels(api_key=api_key, priority=req_priority).inc(actual_tokens)

        # Record forecasting metrics for successful requests
        if FORECASTING_ENABLED and forecasting:
            try:
                await forecasting.record_request_metrics(
                    api_key=api_key,
                    tokens_used=actual_tokens,
                    priority=req_priority,
                    route=route_name,
                    model=model,
                )
            except Exception as metrics_error:
                # Don't fail the request if metrics recording fails
                logger.warning(f"Failed to record forecasting metrics: {metrics_error}")

    # Quota enforcement for non-streaming
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
        quota_rejections_total.labels(api_key=api_key, priority=req_priority, scope=scope).inc()
        raise

    return Response(
        content=raw,
        status_code=upstream_resp.status_code,
        headers=dict(upstream_resp.headers),
        media_type=upstream_resp.headers.get("content-type"),
    )


# Add periodic maintenance task


async def periodic_maintenance():
    """Run periodic maintenance tasks"""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            if FORECASTING_ENABLED and forecasting:
                await forecasting.run_maintenance()
                logger.info("Periodic maintenance completed")
        except Exception as e:
            logger.error(f"Periodic maintenance failed: {e}")


# Start background maintenance on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    # Start the maintenance task
    maintenance_task = asyncio.create_task(periodic_maintenance())

    yield

    # shutdown
    maintenance_task.cancel()
    try:
        await maintenance_task
    except asyncio.CancelledError:
        pass

    if sentinel:
        await sentinel.close()
    await upstream.close()
    await redis.aclose()


# Add a health check that includes forecasting and sentinel status
@app.get("/healthz")
async def healthz():
    """Enhanced health check including forecasting and sentinel status"""
    try:
        # Test Redis connectivity
        await redis.ping()
        redis_status = "healthy"
    except Exception:
        redis_status = "unhealthy"

    forecasting_status = "disabled"
    if FORECASTING_ENABLED and forecasting:
        try:
            # Test forecasting system
            test_date = datetime.now(timezone.utc) - timedelta(days=1)
            await forecasting.collector.get_top_users(test_date, limit=1)
            forecasting_status = "healthy"
        except Exception:
            forecasting_status = "unhealthy"

    sentinel_status = "disabled"
    if settings.SENTINEL_ENABLED and sentinel:
        try:
            if await sentinel.health_check():
                sentinel_status = "healthy"
            else:
                sentinel_status = "unhealthy"
        except Exception:
            sentinel_status = "unhealthy"

    # Determine overall status
    critical_components_ok = redis_status == "healthy"
    optional_components_ok = forecasting_status in ["healthy", "disabled"] and sentinel_status in [
        "healthy",
        "disabled",
    ]

    if critical_components_ok and optional_components_ok:
        overall_status = "healthy"
    elif critical_components_ok:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"

    components = {"redis": redis_status}
    if FORECASTING_ENABLED:
        components["forecasting"] = forecasting_status
    if settings.SENTINEL_ENABLED:
        components["sentinel"] = sentinel_status

    return {
        "ok": critical_components_ok,  # OK if critical components (Redis) are healthy
        "status": overall_status,
        "components": components,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
