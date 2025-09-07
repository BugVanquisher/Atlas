# tests/test_streaming_quota.py
import json

import pytest
import pytest_asyncio
from httpx import AsyncClient

import gateway.main as main
from gateway.config import settings

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture(autouse=True)
async def clear_quota():
    # reset state before each test
    keys = await main.quota.redis.keys("*")
    if keys:
        await main.quota.redis.delete(*keys)
    yield
    # keys = await quota.redis.keys("*")
    # if keys:
    #     await quota.redis.delete(*keys)


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    async with AsyncClient(app=main.app, base_url="http://test") as c:
        yield c


async def register_key(client: AsyncClient, api_key="test-stream", daily=100, monthly=1000):
    resp = await client.post(
        "/v1/admin/keys",
        headers={"x-admin-key": settings.ADMIN_API_KEY},
        json={"api_key": api_key, "daily_limit": daily, "monthly_limit": monthly},
    )
    assert resp.status_code == 200
    return api_key


async def test_reservation_and_refund_success(client: AsyncClient, monkeypatch):
    api_key = await register_key(client)

    # Mock upstream.stream to yield JSON with usage
    class DummyResp:
        status_code = 200

        def aiter_raw(self):
            async def gen():
                yield json.dumps({"usage": {"total_tokens": 20}}).encode()

            return gen()

    def fake_stream(*args, **kwargs):
        class DummyCtx:
            async def __aenter__(self):
                return DummyResp()

            async def __aexit__(self, exc_type, exc, tb):
                return False

        return DummyCtx()

    monkeypatch.setattr("gateway.main.upstream.stream", fake_stream)

    resp = await client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "content-type": "application/json"},
        json={
            "model": "mock",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 50,
            "stream": True,
        },
    )
    assert resp.status_code == 200

    await resp.aread()

    used_d, _ = await main.quota.get_usage(api_key)
    # Reservation was 50, actual usage was 20 → refund of 30 → net = 20
    assert used_d == 20


async def test_full_reservation_if_no_usage(client: AsyncClient, monkeypatch):
    api_key = await register_key(client)

    class DummyResp:
        status_code = 200

        def aiter_raw(self):
            async def gen():
                yield b"chunk"

            return gen()

    def fake_stream(*args, **kwargs):
        class DummyCtx:
            async def __aenter__(self):
                return DummyResp()

            async def __aexit__(self, exc_type, exc, tb):
                return False

        return DummyCtx()

    monkeypatch.setattr("gateway.main.upstream.stream", fake_stream)

    resp = await client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "content-type": "application/json"},
        json={
            "model": "mock",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 40,
            "stream": True,
        },
    )
    assert resp.status_code == 200
    await resp.aread()

    used_d, _ = await main.quota.get_usage(api_key)
    assert used_d == 40  # charged full reservation


async def test_refund_all_on_failure(client: AsyncClient, monkeypatch):
    api_key = await register_key(client)

    class DummyResp:
        status_code = 200

        def aiter_raw(self):
            async def gen():
                raise RuntimeError("upstream failed")
                yield  # to make this an async generator

            return gen()

    def fake_stream(*args, **kwargs):
        class DummyCtx:
            async def __aenter__(self):
                return DummyResp()

            async def __aexit__(self, exc_type, exc, tb):
                return False

        return DummyCtx()

    monkeypatch.setattr("gateway.main.upstream.stream", fake_stream)

    resp = await client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "content-type": "application/json"},
        json={
            "model": "mock",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 30,
            "stream": True,
        },
    )
    assert resp.status_code in (500, 502, 503)  # upstream error propagates

    used_d, _ = await main.quota.get_usage(api_key)
    assert used_d == 0  # refunded all
