from pydantic_settings import BaseSettings, SettingsConfigDict
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter


class Settings(BaseSettings):
    UPSTREAM_BASE_URL: str = "http://mock-upstream:9000"
    REDIS_URL: str = "redis://redis:6379/0"
    ADMIN_API_KEY: str = "my-admin-key"

    DEFAULT_DAILY_LIMIT: int = 100_000
    DEFAULT_MONTHLY_LIMIT: int = 1_000_000
    DEFAULT_RATE_PER_SEC: float = 2.0
    DEFAULT_BURST: int = 5

    PORT: int = 8080

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


# Per-request counters, split by API key and priority.
requests_total = Counter(
    "atlas_requests_total",
    "Total requests received by Atlas",
    ["api_key", "priority", "route"],
)

# Count tokens consumed (as reported by upstream) by api_key/priority.
tokens_used_total = Counter(
    "atlas_tokens_used_total",
    "Total tokens used (as reported by upstream Usage.total_tokens)",
    ["api_key", "priority"],
)

# Rejections due to quotas (daily/monthly); scope label = 'daily' or 'monthly'.
quota_rejections_total = Counter(
    "atlas_quota_rejections_total",
    "Requests rejected due to quota limits",
    ["api_key", "priority", "scope"],
)

# Rejections due to rate limiting.
rate_limit_rejections_total = Counter(
    "atlas_rate_limit_rejections_total",
    "Requests rejected due to rate limiting",
    ["api_key", "priority"],
)


def setup_metrics(app):
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")


def route_name_from_path(full_path: str) -> str:
    # Low-cardinality route labeling
    # We only care to split /v1/chat/completions from "other".
    p = full_path.lower()
    if p.startswith("chat/completions") or "chat/completions" in p:
        return "chat_completions"
    return "other"


settings = Settings()
