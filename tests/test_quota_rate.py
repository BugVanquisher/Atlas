import pytest
from httpx import AsyncClient
from gateway.main import app, settings

@pytest.mark.asyncio
async def test_admin_set_get_limits():
    api_key = "test-key-limits"
    headers = {"x-admin-key": settings.ADMIN_API_KEY, "content-type": "application/json"}
    payload = {
        "api_key": api_key,
        "daily_limit": 50000,
        "monthly_limit": 500000,
        "rate_per_sec": 10.0,
        "burst": 20,
    }

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/v1/admin/keys", json=payload, headers=headers)
        assert response.status_code == 200
        assert response.json() == {"ok": True, "api_key": api_key}

        # Verify that the limits are set correctly
        response = await ac.get(f"/v1/usage", headers={"Authorization": f"Bearer {api_key}"})
        assert response.status_code == 200
        data = response.json()
        assert data["daily_limit"] == 50000
        assert data["monthly_limit"] == 500000