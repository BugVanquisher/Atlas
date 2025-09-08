from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from httpx import AsyncClient

import gateway.main as main
from gateway.config import settings
from gateway.streaming import StreamingHandler, TokenUsageParser

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture(autouse=True)
async def clear_quota():
    keys = await main.quota.redis.keys("*")
    if keys:
        await main.quota.redis.delete(*keys)
    yield


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    async with AsyncClient(app=main.app, base_url="http://test") as c:
        yield c


async def register_key(client: AsyncClient, api_key="test-stream", daily=100000, monthly=1000000):
    resp = await client.post(
        "/v1/admin/keys",
        headers={"x-admin-key": settings.ADMIN_API_KEY},
        json={"api_key": api_key, "daily_limit": daily, "monthly_limit": monthly},
    )
    assert resp.status_code == 200
    return api_key


class TestTokenUsageParser:
    """Test the token usage parsing functionality"""

    def test_parse_openai_chunk_with_usage(self):
        # Test OpenAI-style chunk with usage
        chunk_data = (
            b'data: {"id":"test","object":"chat.completion.chunk","usage":'
            b'{"total_tokens":42}}\n\n'
        )
        parser = TokenUsageParser()
        tokens = parser.parse_openai_chunk(chunk_data)
        assert tokens == 42

    def test_parse_openai_chunk_without_usage(self):
        # Test chunk without usage data
        chunk_data = (
            b'data: {"id":"test","object":"chat.completion.chunk","choices":'
            b'[{"delta":{"content":"hello"}}]}\n\n'
        )
        parser = TokenUsageParser()
        tokens = parser.parse_openai_chunk(chunk_data)
        assert tokens is None

    def test_parse_openai_chunk_done_marker(self):
        # Test [DONE] marker
        chunk_data = b"data: [DONE]\n\n"
        parser = TokenUsageParser()
        tokens = parser.parse_openai_chunk(chunk_data)
        assert tokens is None

    def test_parse_accumulated_response_streaming(self):
        # Test parsing complete streaming response
        response_body = b"""data: {"id":"test","choices":[{"delta":{"content":"hi"}}]}

data: {"id":"test","choices":[{"delta":{"content":" there"}}]}

data: {"id":"test","usage":{"total_tokens":25}}

data: [DONE]

"""
        parser = TokenUsageParser()
        tokens = parser.parse_accumulated_response(response_body)
        assert tokens == 25

    def test_parse_accumulated_response_json(self):
        # Test parsing complete JSON response (non-streaming fallback)
        response_body = (
            b'{"id":"test","choices":[{"message":{"content":"hello"}}],'
            b'"usage":{"total_tokens":30}}'
        )
        parser = TokenUsageParser()
        tokens = parser.parse_accumulated_response(response_body)
        assert tokens == 30


class TestStreamingHandler:
    """Test the streaming handler functionality"""

    @pytest_asyncio.fixture
    async def streaming_handler(self):
        # Mock metrics
        metrics = MagicMock()
        metrics.quota_rejections_total = MagicMock()
        metrics.quota_rejections_total.labels.return_value.inc = MagicMock()
        metrics.tokens_used_total = MagicMock()
        metrics.tokens_used_total.labels.return_value.inc = MagicMock()

        return StreamingHandler(main.quota, main.upstream, metrics)

    def test_calculate_reservation_with_max_tokens(self, streaming_handler):
        payload = {"max_tokens": 100}
        limits = {"priority": "normal", "daily_limit": 10000}

        reservation = streaming_handler._calculate_reservation(payload, limits)
        assert reservation == 100

    def test_calculate_reservation_without_max_tokens(self, streaming_handler):
        payload = {}
        limits = {"priority": "high", "daily_limit": 10000}

        reservation = streaming_handler._calculate_reservation(payload, limits)
        # Should be 30% of daily limit for high priority, capped at 1000
        expected = min(int(10000 * 0.3), 1000)
        assert reservation == expected

    async def test_check_and_reserve_quota_success(self, streaming_handler):
        api_key = "test-quota"
        reservation = 100
        limits = {"daily_limit": 10000, "monthly_limit": 100000}
        req_priority = "normal"

        # Should not raise exception
        await streaming_handler._check_and_reserve_quota(api_key, reservation, limits, req_priority)

        # Verify tokens were reserved
        used_d, used_m = await main.quota.get_usage(api_key)
        assert used_d == 100
        assert used_m == 100

    async def test_check_and_reserve_quota_daily_exceeded(self, streaming_handler):
        api_key = "test-quota-daily"

        # Set usage close to limit
        await main.quota.add_tokens(api_key, 9950)

        reservation = 100
        limits = {"daily_limit": 10000, "monthly_limit": 100000}
        req_priority = "normal"

        # Should raise HTTPException
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await streaming_handler._check_and_reserve_quota(
                api_key, reservation, limits, req_priority
            )

        assert exc_info.value.status_code == 429
        assert "Daily quota would be exceeded" in str(exc_info.value.detail)

    async def test_check_and_reserve_quota_monthly_exceeded(self, streaming_handler):
        api_key = "test-quota-monthly"

        # Set usage that exceeds monthly but not daily limit
        # Daily usage will be much less than daily limit (10000)
        # But monthly usage will be close to monthly limit (100000)
        await main.quota.add_tokens(api_key, 99950)

        reservation = 100
        limits = {
            "daily_limit": 200000,
            "monthly_limit": 100000,
        }  # High daily limit, low monthly limit
        req_priority = "normal"

        # Should raise HTTPException
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await streaming_handler._check_and_reserve_quota(
                api_key, reservation, limits, req_priority
            )

        assert exc_info.value.status_code == 429
        assert "Monthly quota would be exceeded" in str(exc_info.value.detail)


class TestStreamingIntegration:
    """Integration tests for streaming with mocked upstream"""

    async def test_streaming_with_usage_tracking(self, client: AsyncClient, monkeypatch):
        """Test streaming request with proper usage tracking and refunding"""
        api_key = await register_key(client, daily=1000, monthly=10000)

        # Mock upstream that returns usage data
        class MockStreamingResp:
            status_code = 200

            def aiter_raw(self):
                async def gen():
                    # Yield some content chunks
                    hello_data = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
                    world_data = b'data: {"choices":[{"delta":{"content":" hi"}}]}\n\n'
                    yield hello_data
                    yield world_data
                    # Final chunk with usage
                    yield b'data: {"id":"test","usage":{"total_tokens":15}}\n\n'
                    yield b"data: [DONE]\n\n"

                return gen()

        def mock_stream(*args, **kwargs):
            class MockCtx:
                async def __aenter__(self):
                    return MockStreamingResp()

                async def __aexit__(self, exc_type, exc, tb):
                    return False

            return MockCtx()

        monkeypatch.setattr("gateway.main.upstream.stream", mock_stream)

        # Make streaming request
        resp = await client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
            json={
                "model": "mock",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 50,
                "stream": True,
            },
        )

        assert resp.status_code == 200

        # Read the full response
        response_content = await resp.aread()
        assert b"hi" in response_content

        # Check quota usage: should be 15 (actual) not 50 (reserved)
        used_d, used_m = await main.quota.get_usage(api_key)
        assert used_d == 15
        assert used_m == 15

    async def test_streaming_no_usage_data(self, client: AsyncClient, monkeypatch):
        """Test streaming when no usage data is available"""
        api_key = await register_key(client, daily=1000, monthly=10000)

        class MockStreamingResp:
            status_code = 200

            def aiter_raw(self):
                async def gen():
                    hello_data = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
                    yield hello_data
                    yield b"data: [DONE]\n\n"

                return gen()

        def mock_stream(*args, **kwargs):
            class MockCtx:
                async def __aenter__(self):
                    return MockStreamingResp()

                async def __aexit__(self, exc_type, exc, tb):
                    return False

            return MockCtx()

        monkeypatch.setattr("gateway.main.upstream.stream", mock_stream)

        resp = await client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
            json={
                "model": "mock",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 30,
                "stream": True,
            },
        )

        assert resp.status_code == 200
        await resp.aread()

        # Should keep full reservation when no usage data
        used_d, used_m = await main.quota.get_usage(api_key)
        assert used_d == 30
        assert used_m == 30

    async def test_streaming_upstream_error(self, client: AsyncClient, monkeypatch):
        """Test streaming when upstream returns error"""
        api_key = await register_key(client, daily=1000, monthly=10000)

        class MockStreamingResp:
            status_code = 500  # Upstream error

            def aiter_raw(self):
                async def gen():
                    raise RuntimeError("Upstream failed")
                    yield  # Make it a generator

                return gen()

        def mock_stream(*args, **kwargs):
            class MockCtx:
                async def __aenter__(self):
                    return MockStreamingResp()

                async def __aexit__(self, exc_type, exc, tb):
                    return False

            return MockCtx()

        monkeypatch.setattr("gateway.main.upstream.stream", mock_stream)

        resp = await client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
            json={
                "model": "mock",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 40,
                "stream": True,
            },
        )

        # Should return error status
        assert resp.status_code in (500, 502, 503)

        # Should refund all tokens on error
        used_d, used_m = await main.quota.get_usage(api_key)
        assert used_d == 0
        assert used_m == 0

    async def test_streaming_quota_exceeded(self, client: AsyncClient):
        """Test streaming request when quota would be exceeded"""
        api_key = await register_key(client, daily=100, monthly=1000)

        resp = await client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
            json={
                "model": "mock",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 150,  # Exceeds daily limit of 100
                "stream": True,
            },
        )

        assert resp.status_code == 429
        data = resp.json()
        assert "Daily quota would be exceeded" in data["detail"]

        # Should not have charged any tokens
        used_d, used_m = await main.quota.get_usage(api_key)
        assert used_d == 0
        assert used_m == 0

    async def test_streaming_dynamic_reservation(self, client: AsyncClient, monkeypatch):
        """Test streaming with dynamic reservation based on priority"""
        api_key = await register_key(client, daily=10000, monthly=100000)

        # Set key to high priority
        await client.post(
            "/v1/admin/keys",
            headers={"x-admin-key": settings.ADMIN_API_KEY},
            json={
                "api_key": api_key,
                "daily_limit": 10000,
                "monthly_limit": 100000,
                "priority": "high",
            },
        )

        class MockStreamingResp:
            status_code = 200

            def aiter_raw(self):
                async def gen():
                    yield b'data: {"id":"test","usage":{"total_tokens":20}}\n\n'
                    yield b"data: [DONE]\n\n"

                return gen()

        def mock_stream(*args, **kwargs):
            class MockCtx:
                async def __aenter__(self):
                    return MockStreamingResp()

                async def __aexit__(self, exc_type, exc, tb):
                    return False

            return MockCtx()

        monkeypatch.setattr("gateway.main.upstream.stream", mock_stream)

        # Request without max_tokens - should use dynamic reservation
        resp = await client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "content-type": "application/json",
                "x-request-priority": "high",
            },
            json={
                "model": "mock",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                # No max_tokens specified
            },
        )

        assert resp.status_code == 200
        await resp.aread()

        # Should charge actual usage (20 tokens)
        used_d, used_m = await main.quota.get_usage(api_key)
        assert used_d == 20
        assert used_m == 20
