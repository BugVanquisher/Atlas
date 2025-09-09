import pytest
from httpx import AsyncClient

from gateway.main import app


@pytest.mark.asyncio
async def test_healthz():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert "status" in data
    assert "components" in data
    assert "timestamp" in data
