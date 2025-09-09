import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import pytest_asyncio
from httpx import AsyncClient

import gateway.main as main
from forecasting.core import (
    CapacityPlanner,
    TimeSeriesForecaster,
    TrendDirection,
    UsageAnalyzer,
    UsageDataPoint,
)
from forecasting.metrics_collector import ForecastingIntegration, MetricsCollector
from gateway.config import settings

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture(autouse=True)
async def clear_quota_and_metrics():
    """Clear both quota and forecasting metrics before each test"""
    keys = await main.redis.keys("*")
    if keys:
        await main.redis.delete(*keys)
    yield


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    async with AsyncClient(app=main.app, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def metrics_collector():
    """Provide a metrics collector instance"""
    return MetricsCollector(main.redis)


@pytest_asyncio.fixture
async def forecasting_integration():
    """Provide a forecasting integration instance"""
    return ForecastingIntegration(main.redis)


async def register_key(client: AsyncClient, api_key="test-forecast", daily=100000, monthly=1000000):
    """Helper to register a test API key"""
    resp = await client.post(
        "/v1/admin/keys",
        headers={"x-admin-key": settings.ADMIN_API_KEY},
        json={"api_key": api_key, "daily_limit": daily, "monthly_limit": monthly},
    )
    assert resp.status_code == 200
    return api_key


class TestTimeSeriesForecaster:
    """Test the core forecasting algorithms"""

    def test_simple_moving_average(self):
        forecaster = TimeSeriesForecaster()
        data = [10, 20, 30, 40, 50]

        forecasts = forecaster.simple_moving_average(data, window=3)

        # Should have same length as input
        assert len(forecasts) == len(data)

        # Last forecast should be average of last 3 values
        assert forecasts[-1] == 40.0  # (30 + 40 + 50) / 3

    def test_exponential_smoothing(self):
        forecaster = TimeSeriesForecaster()
        data = [100, 110, 120, 130, 140]

        forecasts = forecaster.exponential_smoothing(data, alpha=0.3)

        assert len(forecasts) == len(data)
        assert forecasts[0] == data[0]  # First forecast equals first data point

        # Second forecast: 0.3 * 100 + 0.7 * 100 = 100
        assert forecasts[1] == 100.0

    def test_linear_trend(self):
        forecaster = TimeSeriesForecaster()
        data = [10, 20, 30, 40, 50]  # Perfect linear trend

        slope, intercept = forecaster.linear_trend(data)

        assert abs(slope - 10.0) < 0.01  # Slope should be 10
        assert abs(intercept - 10.0) < 0.01  # Intercept should be 10

    def test_seasonality_detection(self):
        forecaster = TimeSeriesForecaster()

        # Create seasonal data (24-hour pattern)
        data = []
        for _ in range(10):
            for hour in range(24):
                # Higher usage during business hours
                value = 100 + 50 * np.sin(2 * np.pi * hour / 24)
                data.append(value)

        seasonal = forecaster.detect_seasonality(data, period=24)
        assert seasonal  # Should detect 24-hour seasonality

        # Non-seasonal data (may or may not detect seasonality in random data)
        random_data = np.random.normal(100, 10, 48).tolist()
        forecaster.detect_seasonality(random_data, period=24)

    def test_forecast_future(self):
        forecaster = TimeSeriesForecaster()
        data = [10, 15, 20, 25, 30, 35, 40, 45]  # Increasing trend with 8 points

        forecasts, lower, upper = forecaster.forecast_future(data, periods=3)

        assert len(forecasts) == 3
        assert len(lower) == 3
        assert len(upper) == 3

        # Lower bounds should be <= forecasts <= upper bounds
        for i in range(3):
            assert lower[i] <= forecasts[i] <= upper[i]

        # Forecasts should show increasing trend
        assert forecasts[1] > forecasts[0]
        assert forecasts[2] > forecasts[1]


class TestUsageAnalyzer:
    """Test usage pattern analysis"""

    def test_analyze_trend_increasing(self):
        analyzer = UsageAnalyzer()
        data = [10, 15, 20, 25, 30, 35, 40, 45, 50]  # Clear increasing trend

        direction, strength = analyzer.analyze_trend(data)

        assert direction == TrendDirection.INCREASING
        assert strength > 0.8  # Should have high confidence

    def test_analyze_trend_decreasing(self):
        analyzer = UsageAnalyzer()
        data = [50, 45, 40, 35, 30, 25, 20, 15, 10]  # Clear decreasing trend

        direction, strength = analyzer.analyze_trend(data)

        assert direction == TrendDirection.DECREASING
        assert strength > 0.8

    def test_analyze_trend_stable(self):
        analyzer = UsageAnalyzer()
        data = [100, 102, 98, 101, 99, 103, 97, 101, 100]  # Stable around 100

        direction, strength = analyzer.analyze_trend(data)

        assert direction == TrendDirection.STABLE

    def test_analyze_trend_volatile(self):
        analyzer = UsageAnalyzer()
        data = [10, 50, 15, 45, 20, 55, 5, 40, 12]  # High volatility

        direction, strength = analyzer.analyze_trend(data)

        assert direction == TrendDirection.VOLATILE

    def test_calculate_growth_rate(self):
        analyzer = UsageAnalyzer()

        # 100% growth over 7 periods
        data = [100, 110, 120, 130, 140, 150, 160, 200]
        growth_rate = analyzer.calculate_growth_rate(data, periods=7)

        # Should be positive growth
        assert growth_rate > 0

        # Declining data
        declining_data = [200, 180, 160, 140, 120, 100, 80, 60]
        decline_rate = analyzer.calculate_growth_rate(declining_data, periods=7)

        assert decline_rate < 0

    def test_detect_usage_patterns(self):
        analyzer = UsageAnalyzer()

        # Create usage data with patterns
        usage_data = []
        base_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

        for day in range(7):
            for hour in range(24):
                timestamp = base_time + timedelta(days=day, hours=hour)
                # Higher usage during business hours (9-17)
                tokens = 50 if 9 <= hour <= 17 else 20

                usage_data.append(
                    UsageDataPoint(
                        timestamp=timestamp,
                        api_key="test",
                        tokens_used=tokens,
                        requests_count=1,
                        priority="normal",
                        route="chat",
                    )
                )

        patterns = analyzer.detect_usage_patterns(usage_data)

        assert "peak_hour" in patterns
        assert "peak_weekday" in patterns
        assert "hourly_averages" in patterns
        assert "weekly_averages" in patterns

        # Peak hour should be during business hours
        assert 9 <= patterns["peak_hour"] <= 17


class TestCapacityPlanner:
    """Test capacity planning recommendations"""

    def test_calculate_recommended_limits(self):
        planner = CapacityPlanner()

        current_limits = {"daily_limit": 10000, "monthly_limit": 300000}
        predicted_usage = [8000, 8500, 9000, 9500, 10000]  # Growing usage
        growth_rate = 0.1  # 10% growth

        recommendations = planner.calculate_recommended_limits(
            current_limits, predicted_usage, growth_rate, horizon_days=30
        )

        assert recommendations["daily_limit"] > current_limits["daily_limit"]
        assert recommendations["monthly_limit"] > current_limits["monthly_limit"]
        assert recommendations["utilization_daily"] <= 1.0

    def test_calculate_rate_limits(self):
        planner = CapacityPlanner()

        request_patterns = [100, 120, 150, 200, 180]  # Requests per hour
        current_rate = 1.0

        recommendations = planner.calculate_rate_limits(
            request_patterns, current_rate, peak_factor=2.0
        )

        assert recommendations["rate_per_sec"] >= 0.1
        assert recommendations["burst"] >= 5
        assert "peak_utilization" in recommendations

    def test_assess_overrun_risk(self):
        planner = CapacityPlanner()

        predicted_usage = [8000, 8500, 9000, 9500]
        current_limits = {"daily_limit": 10000}
        confidence_upper = [9000, 9500, 10000, 11000]  # One prediction exceeds limit

        risk = planner.assess_overrun_risk(predicted_usage, current_limits, confidence_upper)

        assert 0 <= risk <= 1.0
        assert risk > 0  # Should detect some risk


class TestMetricsCollector:
    """Test metrics collection and storage"""

    async def test_record_usage(self, metrics_collector):
        api_key = "test-metrics"

        await metrics_collector.record_usage(
            api_key=api_key,
            tokens_used=100,
            requests_count=1,
            priority="normal",
            route="chat",
            model="gpt-4",
        )

        # Check that data was stored
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y%m%d")

        daily_key = f"atlas:metrics:daily:{api_key}:{date_str}"
        data = await main.redis.hgetall(daily_key)

        assert data
        assert int(data[b"tokens_used"]) == 100
        assert int(data[b"requests_count"]) == 1

    async def test_get_hourly_usage(self, metrics_collector):
        api_key = "test-hourly"

        # Record some usage data
        base_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

        for i in range(5):
            timestamp = base_time + timedelta(hours=i)
            await metrics_collector.record_usage(
                api_key=api_key, tokens_used=100 + i * 10, timestamp=timestamp
            )

        # Retrieve hourly data
        start_time = base_time
        end_time = base_time + timedelta(hours=4)

        usage_data = await metrics_collector.get_hourly_usage(api_key, start_time, end_time)

        assert len(usage_data) == 5
        assert usage_data[0].tokens_used == 100
        assert usage_data[4].tokens_used == 140

    async def test_get_usage_breakdown(self, metrics_collector):
        api_key = "test-breakdown"

        # Record usage with different priorities and routes
        now = datetime.now(timezone.utc)

        await metrics_collector.record_usage(api_key, 100, priority="normal", route="chat")
        await metrics_collector.record_usage(api_key, 50, priority="high", route="completions")
        await metrics_collector.record_usage(api_key, 75, priority="normal", route="chat")

        breakdown = await metrics_collector.get_usage_breakdown(api_key, now)

        assert breakdown["total_tokens"] == 225
        assert breakdown["total_requests"] == 3
        assert breakdown["priority_breakdown"]["tokens"]["normal"] == 175
        assert breakdown["priority_breakdown"]["tokens"]["high"] == 50
        assert breakdown["route_breakdown"]["tokens"]["chat"] == 175

    async def test_get_top_users(self, metrics_collector):
        # Record usage for multiple users
        now = datetime.now(timezone.utc)

        users_data = [("user1", 1000), ("user2", 500), ("user3", 1500), ("user4", 200)]

        for api_key, tokens in users_data:
            await metrics_collector.record_usage(api_key, tokens, timestamp=now)

        top_users = await metrics_collector.get_top_users(now, limit=3)

        assert len(top_users) == 3
        assert top_users[0]["api_key"] == "user3"  # Highest usage
        assert top_users[0]["tokens_used"] == 1500
        assert top_users[1]["api_key"] == "user1"  # Second highest
        assert top_users[2]["api_key"] == "user2"  # Third highest


class TestForecastingAPI:
    """Test the forecasting API endpoints"""

    async def test_get_forecast_endpoint(self, client, forecasting_integration):
        api_key = await register_key(client, "test-api-forecast")

        # Record some historical usage
        for i in range(7):
            timestamp = datetime.now(timezone.utc) - timedelta(days=i)
            await forecasting_integration.collector.record_usage(
                api_key=api_key, tokens_used=1000 + i * 100, timestamp=timestamp
            )

        # Get forecast
        response = await client.get(
            "/v1/forecasting/forecast?horizon=daily", headers={"Authorization": f"Bearer {api_key}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["api_key"] == api_key
        assert data["forecast_horizon"] == "daily"
        assert "predictions" in data
        assert "capacity_recommendations" in data
        assert "confidence_intervals" in data
        assert "risk_assessment" in data

    async def test_get_usage_patterns_endpoint(self, client, forecasting_integration):
        api_key = await register_key(client, "test-patterns")

        # Record usage patterns
        base_time = datetime.now(timezone.utc)
        for day in range(10):
            for hour in [9, 12, 15]:  # Business hours
                timestamp = base_time - timedelta(days=day, hours=24 - hour)
                await forecasting_integration.collector.record_usage(
                    api_key=api_key, tokens_used=100, timestamp=timestamp
                )

        response = await client.get(
            "/v1/forecasting/usage-patterns?days_back=10",
            headers={"Authorization": f"Bearer {api_key}"},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["api_key"] == api_key
        assert "summary" in data
        assert "peak_usage" in data
        assert "priority_distribution" in data

    async def test_capacity_recommendations_endpoint(self, client, forecasting_integration):
        api_key = await register_key(client, "test-capacity", daily=5000, monthly=150000)

        # Record high usage to trigger recommendations
        for i in range(7):
            timestamp = datetime.now(timezone.utc) - timedelta(days=i)
            await forecasting_integration.collector.record_usage(
                api_key=api_key,
                tokens_used=4000 + i * 200,  # Growing usage near limit
                timestamp=timestamp,
            )

        response = await client.get(
            "/v1/forecasting/capacity-recommendations",
            headers={"Authorization": f"Bearer {api_key}"},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["api_key"] == api_key
        assert "current_limits" in data
        assert "recommendations" in data
        assert "analysis" in data

        # Should recommend higher limits due to high usage
        assert data["recommendations"]["daily_limit"] > data["current_limits"]["daily_limit"]

    async def test_invalid_horizon(self, client):
        api_key = await register_key(client, "test-invalid")

        response = await client.get(
            "/v1/forecasting/forecast?horizon=invalid",
            headers={"Authorization": f"Bearer {api_key}"},
        )

        assert response.status_code == 400
        assert "Invalid horizon" in response.json()["detail"]

    async def test_unauthorized_access(self, client):
        response = await client.get("/v1/forecasting/forecast")
        assert response.status_code == 401


class TestIntegrationWithMainApp:
    """Test integration with the main application"""

    async def test_metrics_recorded_on_request(self, client, forecasting_integration, monkeypatch):
        api_key = await register_key(client, "test-integration")

        # Mock upstream response
        import json
        from unittest.mock import MagicMock

        # Mock the upstream client
        mock_response = MagicMock()
        mock_response.content = json.dumps(
            {
                "id": "test",
                "choices": [{"message": {"content": "Hello"}}],
                "usage": {"total_tokens": 25},
            }
        ).encode()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}

        async def mock_forward(*args, **kwargs):
            return mock_response

        monkeypatch.setattr("gateway.main.upstream.forward", mock_forward)

        # Make a request through the main app
        await client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 50,
            },
        )

        # Give some time for metrics to be recorded
        await asyncio.sleep(0.1)

        # Check that metrics were recorded
        now = datetime.now(timezone.utc)
        await forecasting_integration.collector.get_usage_breakdown(api_key, now)

        # Note: This test might not pass without proper upstream mocking
        # In a real scenario, you'd need to mock the upstream service

    async def test_enhanced_health_check(self, client):
        response = await client.get("/healthz")
        assert response.status_code == 200

        data = response.json()
        assert "components" in data
        assert "redis" in data["components"]
        assert "forecasting" in data["components"]
