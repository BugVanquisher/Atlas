import pytest
from httpx import AsyncClient
from atlas_gateway.main import app

@pytest.mark.asyncio
async def test_usage_endpoint():
    api_key = "test-key-usage"
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # First, set some limits for this key so we know what to expect.
        admin_headers = {"x-admin-key": "my-admin-key"}
        limits_payload = {"api_key": api_key, "daily_limit": 12345}
        await ac.post("/v1/admin/keys", json=limits_payload, headers=admin_headers)


        headers = {"Authorization": f"Bearer {api_key}"}
        response = await ac.get("/v1/usage", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["daily_used"] == 0
    assert data["monthly_used"] == 0
    assert data["daily_limit"] == 12345