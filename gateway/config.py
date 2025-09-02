import os
from pydantic_settings import BaseSettings, SettingsConfigDict


def detect_redis_url() -> str:
    """
    Detect whether we're running inside Docker Compose or locally.

    - If running in Docker (REDIS_HOST is set or /.dockerenv exists):
        use redis://redis:6379/0
    - Otherwise (local dev / pytest):
        use redis://localhost:6379/0
    """
    if os.getenv("REDIS_URL"):
        return os.getenv("REDIS_URL")

    if os.getenv("REDIS_HOST") or os.path.exists("/.dockerenv"):
        return "redis://redis:6379/0"

    return "redis://localhost:6379/0"


class Settings(BaseSettings):
    UPSTREAM_BASE_URL: str = "http://mock-upstream:9000"
    REDIS_URL: str = detect_redis_url()
    ADMIN_API_KEY: str = "my-admin-key"

    DEFAULT_DAILY_LIMIT: int = 100_000
    DEFAULT_MONTHLY_LIMIT: int = 1_000_000
    DEFAULT_RATE_PER_SEC: float = 2.0
    DEFAULT_BURST: int = 5

    PORT: int = 8080

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


settings = Settings()
