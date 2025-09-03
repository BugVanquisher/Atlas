import os
import asyncio
import pytest
import fakeredis

# Force local/fake Redis URL so config never defaults to "redis:6379"
# os.environ["REDIS_URL"] = "redis://localhost:6379/0"

import gateway.main as main
from gateway.quota import QuotaManager



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
    loop.close()