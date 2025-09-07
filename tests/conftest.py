import asyncio

import fakeredis
import pytest

import gateway.main as main
from gateway.quota import QuotaManager

# Force local/fake Redis URL so config never defaults to "redis:6379"
# os.environ["REDIS_URL"] = "redis://localhost:6379/0"


class DummyRateLimiter:
    async def allow(self, key, rate_per_sec, burst):
        return True  # always allow in unit tests


@pytest.fixture(autouse=True, scope="function")
async def fake_redis_fixture():
    fake = fakeredis.FakeAsyncRedis()
    main.redis = fake
    main.quota = QuotaManager(fake)
    main.rl = DummyRateLimiter()
    yield
    await fake.flushall()
    await fake.aclose()  # use aclose() for async client


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the entire session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    pending = asyncio.all_tasks(loop)
    for task in pending:
        task.cancel()
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()


def pytest_sessionfinish(session, exitstatus):
    try:
        loop = asyncio.get_event_loop()
        if hasattr(main, "redis"):
            loop.run_until_complete(main.redis.close())
    except Exception:
        pass
