from pydantic_settings import BaseSettings, SettingsConfigDict
from prometheus_fastapi_instrumentator import Instrumentator


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

def setup_metrics(app):
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

settings = Settings()


