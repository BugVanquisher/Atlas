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

    # Enhanced streaming configuration
    DEFAULT_STREAM_RESERVATION: int = 256  # tokens to pre-authorize for stream requests

    # Dynamic reservation multipliers by priority (as percentage of daily limit)
    STREAM_RESERVATION_MULTIPLIERS: dict = {
        "low": 0.05,  # 5% of daily limit
        "normal": 0.10,  # 10% of daily limit
        "high": 0.20,  # 20% of daily limit
        "critical": 0.30,  # 30% of daily limit
    }

    # Maximum reservation cap (prevents excessive reservations)
    MAX_STREAM_RESERVATION: int = 2000

    # Streaming timeout settings
    STREAM_CONNECT_TIMEOUT: int = 30  # seconds to wait for initial connection
    STREAM_READ_TIMEOUT: int = 300  # seconds to wait between chunks
    STREAM_TOTAL_TIMEOUT: int = 1800  # maximum total streaming time (30 min)

    # Chunk processing settings
    STREAM_CHUNK_RATE_LIMIT: int = 1000  # max chunks per second per stream
    STREAM_BUFFER_SIZE: int = 8192  # bytes to buffer before yielding

    # Error handling
    STREAM_MAX_RETRIES: int = 2  # retry attempts for transient failures
    STREAM_RETRY_DELAY: float = 0.5  # seconds between retries

    # Monitoring and alerting
    STREAM_METRICS_ENABLED: bool = True  # enable detailed streaming metrics
    STREAM_ALERT_THRESHOLD: float = 0.8  # alert when error rate exceeds this

    # Quota management
    QUOTA_REFUND_ENABLED: bool = True  # enable quota refunding
    QUOTA_GRACE_PERIOD: int = 60  # seconds to allow quota overrun during streams

    # Development and debugging
    STREAM_DEBUG_LOGGING: bool = False  # enable verbose streaming logs
    STREAM_SAVE_RESPONSES: bool = False  # save streaming responses for debugging

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    def get_stream_reservation_for_priority(self, priority: str, daily_limit: int) -> int:
        """Calculate dynamic stream reservation based on priority and daily limit"""
        multiplier = self.STREAM_RESERVATION_MULTIPLIERS.get(priority.lower(), 0.10)
        calculated = int(daily_limit * multiplier)
        return min(calculated, self.MAX_STREAM_RESERVATION)


settings = Settings()


# Example usage in streaming handler:
# reservation = settings.get_stream_reservation_for_priority(
#     req_priority, limits["daily_limit"]
# ) if not max_tokens else max_tokens
