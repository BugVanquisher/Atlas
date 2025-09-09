import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from redis.asyncio import Redis

from .core import ForecastHorizon, UsageDataPoint

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collect and store detailed usage metrics for forecasting"""

    def __init__(self, redis: Redis):
        self.redis = redis
        self.metrics_key_prefix = "atlas:metrics"
        self.batch_size = 1000

    def _get_metrics_key(self, api_key: str, date: str) -> str:
        """Generate Redis key for storing daily metrics"""
        return f"{self.metrics_key_prefix}:daily:{api_key}:{date}"

    def _get_hourly_key(self, api_key: str, date_hour: str) -> str:
        """Generate Redis key for hourly metrics"""
        return f"{self.metrics_key_prefix}:hourly:{api_key}:{date_hour}"

    async def record_usage(
        self,
        api_key: str,
        tokens_used: int,
        requests_count: int = 1,
        priority: str = "normal",
        route: str = "unknown",
        model: Optional[str] = None,
        cost_estimate: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ):
        """Record a single usage event"""

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        data_point = UsageDataPoint(
            timestamp=timestamp,
            api_key=api_key,
            tokens_used=tokens_used,
            requests_count=requests_count,
            priority=priority,
            route=route,
            model=model,
            cost_estimate=cost_estimate,
        )

        # Store in both hourly and daily aggregations
        await self._store_hourly_metric(data_point)
        await self._store_daily_metric(data_point)

    async def _store_hourly_metric(self, data_point: UsageDataPoint):
        """Store hourly aggregated metrics"""
        date_hour = data_point.timestamp.strftime("%Y%m%d:%H")
        key = self._get_hourly_key(data_point.api_key, date_hour)

        # Use Redis hash to accumulate hourly data
        pipe = self.redis.pipeline()
        pipe.hincrby(key, "tokens_used", data_point.tokens_used)
        pipe.hincrby(key, "requests_count", data_point.requests_count)
        pipe.hincrbyfloat(key, "cost_estimate", data_point.cost_estimate or 0.0)

        # Store metadata if not exists
        pipe.hsetnx(key, "timestamp", data_point.timestamp.isoformat())
        pipe.hsetnx(key, "api_key", data_point.api_key)
        pipe.hsetnx(key, "priority", data_point.priority)
        pipe.hsetnx(key, "route", data_point.route)
        if data_point.model:
            pipe.hsetnx(key, "model", data_point.model)

        # Set expiration (keep hourly data for 90 days)
        pipe.expire(key, 90 * 24 * 3600)

        await pipe.execute()

    async def _store_daily_metric(self, data_point: UsageDataPoint):
        """Store daily aggregated metrics"""
        date = data_point.timestamp.strftime("%Y%m%d")
        key = self._get_metrics_key(data_point.api_key, date)

        # Accumulate daily totals
        pipe = self.redis.pipeline()
        pipe.hincrby(key, "tokens_used", data_point.tokens_used)
        pipe.hincrby(key, "requests_count", data_point.requests_count)
        pipe.hincrbyfloat(key, "cost_estimate", data_point.cost_estimate or 0.0)

        # Track priority breakdown
        pipe.hincrby(key, f"tokens_by_priority:{data_point.priority}", data_point.tokens_used)
        pipe.hincrby(key, f"requests_by_priority:{data_point.priority}", data_point.requests_count)

        # Track route breakdown
        pipe.hincrby(key, f"tokens_by_route:{data_point.route}", data_point.tokens_used)
        pipe.hincrby(key, f"requests_by_route:{data_point.route}", data_point.requests_count)

        # Store metadata
        pipe.hsetnx(key, "date", date)
        pipe.hsetnx(key, "api_key", data_point.api_key)

        # Set expiration (keep daily data for 1 year)
        pipe.expire(key, 365 * 24 * 3600)

        await pipe.execute()

    async def get_hourly_usage(
        self, api_key: str, start_time: datetime, end_time: datetime
    ) -> List[UsageDataPoint]:
        """Retrieve hourly usage data for time range"""

        usage_points = []
        current_time = start_time.replace(minute=0, second=0, microsecond=0)

        while current_time <= end_time:
            date_hour = current_time.strftime("%Y%m%d:%H")
            key = self._get_hourly_key(api_key, date_hour)

            data = await self.redis.hgetall(key)
            if data:
                # Convert bytes to strings
                data = {
                    k.decode() if isinstance(k, bytes) else k: (
                        v.decode() if isinstance(v, bytes) else v
                    )
                    for k, v in data.items()
                }

                usage_point = UsageDataPoint(
                    timestamp=current_time,
                    api_key=api_key,
                    tokens_used=int(data.get("tokens_used", 0)),
                    requests_count=int(data.get("requests_count", 0)),
                    priority=data.get("priority", "normal"),
                    route=data.get("route", "unknown"),
                    model=data.get("model"),
                    cost_estimate=float(data.get("cost_estimate", 0.0)),
                )
                usage_points.append(usage_point)

            current_time += timedelta(hours=1)

        return usage_points

    async def get_daily_usage(
        self, api_key: str, start_date: datetime, end_date: datetime
    ) -> List[UsageDataPoint]:
        """Retrieve daily usage data for date range"""

        usage_points = []
        current_date = start_date.date()

        while current_date <= end_date.date():
            date_str = current_date.strftime("%Y%m%d")
            key = self._get_metrics_key(api_key, date_str)

            data = await self.redis.hgetall(key)
            if data:
                # Convert bytes to strings
                data = {
                    k.decode() if isinstance(k, bytes) else k: (
                        v.decode() if isinstance(v, bytes) else v
                    )
                    for k, v in data.items()
                }

                usage_point = UsageDataPoint(
                    timestamp=datetime.combine(current_date, datetime.min.time()).replace(
                        tzinfo=timezone.utc
                    ),
                    api_key=api_key,
                    tokens_used=int(data.get("tokens_used", 0)),
                    requests_count=int(data.get("requests_count", 0)),
                    priority="normal",  # Daily aggregates don't have single priority
                    route="all",  # Daily aggregates combine all routes
                    cost_estimate=float(data.get("cost_estimate", 0.0)),
                )
                usage_points.append(usage_point)

            current_date += timedelta(days=1)

        return usage_points

    async def get_usage_breakdown(self, api_key: str, date: datetime) -> Dict[str, Any]:
        """Get detailed usage breakdown for a specific date"""

        date_str = date.strftime("%Y%m%d")
        key = self._get_metrics_key(api_key, date_str)

        data = await self.redis.hgetall(key)
        if not data:
            return {}

        # Convert bytes to strings
        data = {
            k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v
            for k, v in data.items()
        }

        # Parse priority breakdown
        priority_tokens = {}
        priority_requests = {}
        route_tokens = {}
        route_requests = {}

        for key_name, value in data.items():
            if key_name.startswith("tokens_by_priority:"):
                priority = key_name.split(":", 1)[1]
                priority_tokens[priority] = int(value)
            elif key_name.startswith("requests_by_priority:"):
                priority = key_name.split(":", 1)[1]
                priority_requests[priority] = int(value)
            elif key_name.startswith("tokens_by_route:"):
                route = key_name.split(":", 1)[1]
                route_tokens[route] = int(value)
            elif key_name.startswith("requests_by_route:"):
                route = key_name.split(":", 1)[1]
                route_requests[route] = int(value)

        return {
            "date": date_str,
            "api_key": api_key,
            "total_tokens": int(data.get("tokens_used", 0)),
            "total_requests": int(data.get("requests_count", 0)),
            "total_cost": float(data.get("cost_estimate", 0.0)),
            "priority_breakdown": {"tokens": priority_tokens, "requests": priority_requests},
            "route_breakdown": {"tokens": route_tokens, "requests": route_requests},
        }

    async def get_top_users(self, date: datetime, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top users by token usage for a specific date"""

        date_str = date.strftime("%Y%m%d")
        pattern = f"{self.metrics_key_prefix}:daily:*:{date_str}"

        keys = await self.redis.keys(pattern)
        user_usage = []

        for key in keys:
            data = await self.redis.hgetall(key)
            if data:
                # Extract API key from Redis key
                api_key = (
                    key.decode().split(":")[-2] if isinstance(key, bytes) else key.split(":")[-2]
                )

                data = {
                    k.decode() if isinstance(k, bytes) else k: (
                        v.decode() if isinstance(v, bytes) else v
                    )
                    for k, v in data.items()
                }

                user_usage.append(
                    {
                        "api_key": api_key,
                        "tokens_used": int(data.get("tokens_used", 0)),
                        "requests_count": int(data.get("requests_count", 0)),
                        "cost_estimate": float(data.get("cost_estimate", 0.0)),
                    }
                )

        # Sort by token usage and return top users
        user_usage.sort(key=lambda x: x["tokens_used"], reverse=True)
        return user_usage[:limit]

    async def cleanup_old_metrics(self, days_to_keep: int = 90):
        """Clean up metrics older than specified days"""

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)

        # Clean hourly metrics
        current_date = cutoff_date
        deleted_count = 0

        while current_date >= cutoff_date - timedelta(days=7):  # Look back 7 more days
            for hour in range(24):
                date_hour = current_date.strftime(f"%Y%m%d:{hour:02d}")
                pattern = f"{self.metrics_key_prefix}:hourly:*:{date_hour}"

                keys = await self.redis.keys(pattern)
                if keys:
                    await self.redis.delete(*keys)
                    deleted_count += len(keys)

            current_date -= timedelta(days=1)

        logger.info(f"Cleaned up {deleted_count} old hourly metric keys")
        return deleted_count


class UsagePredictor:
    """Enhanced predictor that integrates with the metrics collector"""

    def __init__(self, metrics_collector: MetricsCollector):
        self.collector = metrics_collector

    async def collect_usage_data(
        self, api_key: str, hours_back: int = 168  # 7 days
    ) -> List[UsageDataPoint]:
        """Collect real usage data from metrics collector"""

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours_back)

        # Get hourly data for detailed forecasting
        if hours_back <= 720:  # Up to 30 days - use hourly data
            return await self.collector.get_hourly_usage(api_key, start_time, end_time)
        else:  # More than 30 days - use daily data
            return await self.collector.get_daily_usage(api_key, start_time, end_time)

    async def get_usage_patterns(self, api_key: str, days_back: int = 30) -> Dict[str, Any]:
        """Get detailed usage patterns for an API key"""

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)

        # Collect daily breakdowns
        daily_breakdowns = []
        current_date = start_date

        while current_date <= end_date:
            breakdown = await self.collector.get_usage_breakdown(api_key, current_date)
            if breakdown:
                daily_breakdowns.append(breakdown)
            current_date += timedelta(days=1)

        if not daily_breakdowns:
            return {}

        # Analyze patterns
        total_tokens = sum(day["total_tokens"] for day in daily_breakdowns)
        total_requests = sum(day["total_requests"] for day in daily_breakdowns)
        total_cost = sum(day["total_cost"] for day in daily_breakdowns)

        # Calculate averages
        avg_daily_tokens = total_tokens / len(daily_breakdowns)
        avg_daily_requests = total_requests / len(daily_breakdowns)
        avg_daily_cost = total_cost / len(daily_breakdowns)

        # Find peak and low usage days
        peak_day = max(daily_breakdowns, key=lambda x: x["total_tokens"])
        low_day = min(daily_breakdowns, key=lambda x: x["total_tokens"])

        # Aggregate priority usage
        priority_totals = {}
        route_totals = {}

        for day in daily_breakdowns:
            for priority, tokens in day.get("priority_breakdown", {}).get("tokens", {}).items():
                priority_totals[priority] = priority_totals.get(priority, 0) + tokens

            for route, tokens in day.get("route_breakdown", {}).get("tokens", {}).items():
                route_totals[route] = route_totals.get(route, 0) + tokens

        return {
            "summary": {
                "days_analyzed": len(daily_breakdowns),
                "total_tokens": total_tokens,
                "total_requests": total_requests,
                "total_cost": total_cost,
                "avg_daily_tokens": avg_daily_tokens,
                "avg_daily_requests": avg_daily_requests,
                "avg_daily_cost": avg_daily_cost,
            },
            "peak_usage": {
                "date": peak_day["date"],
                "tokens": peak_day["total_tokens"],
                "requests": peak_day["total_requests"],
            },
            "low_usage": {
                "date": low_day["date"],
                "tokens": low_day["total_tokens"],
                "requests": low_day["total_requests"],
            },
            "priority_distribution": priority_totals,
            "route_distribution": route_totals,
            "daily_breakdown": daily_breakdowns,
        }


class ForecastingIntegration:
    """Integration layer that connects forecasting with the main gateway"""

    def __init__(self, redis: Redis):
        self.redis = redis
        self.collector = MetricsCollector(redis)
        self.predictor = UsagePredictor(self.collector)

    async def record_request_metrics(
        self,
        api_key: str,
        tokens_used: int,
        priority: str = "normal",
        route: str = "chat_completions",
        model: Optional[str] = None,
    ):
        """Record metrics for a completed request"""

        # Estimate cost (this should come from your pricing model)
        cost_per_token = 0.00001  # $0.01 per 1000 tokens
        cost_estimate = tokens_used * cost_per_token

        await self.collector.record_usage(
            api_key=api_key,
            tokens_used=tokens_used,
            requests_count=1,
            priority=priority,
            route=route,
            model=model,
            cost_estimate=cost_estimate,
        )

    async def get_forecast_for_api_key(
        self,
        api_key: str,
        horizon: ForecastHorizon = ForecastHorizon.DAILY,
        current_limits: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get comprehensive forecast and recommendations for an API key"""

        from .core import TrafficForecaster

        # Create forecaster with real data collection
        forecaster = TrafficForecaster(self.redis)

        # Override the collect_usage_data method to use our real collector
        forecaster.collect_usage_data = self.predictor.collect_usage_data

        # Generate forecast
        forecast = await forecaster.generate_forecast(api_key, horizon, current_limits)

        # Get additional usage patterns
        patterns = await self.predictor.get_usage_patterns(api_key)

        # Combine results
        result = forecast.to_dict()
        result["usage_patterns"] = patterns

        return result

    async def get_system_wide_metrics(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get system-wide usage metrics"""

        if date is None:
            date = datetime.now(timezone.utc)

        top_users = await self.collector.get_top_users(date, limit=20)

        # Calculate system totals
        total_tokens = sum(user["tokens_used"] for user in top_users)
        total_requests = sum(user["requests_count"] for user in top_users)
        total_cost = sum(user["cost_estimate"] for user in top_users)

        return {
            "date": date.strftime("%Y-%m-%d"),
            "system_totals": {
                "total_tokens": total_tokens,
                "total_requests": total_requests,
                "total_cost": total_cost,
                "active_users": len(top_users),
            },
            "top_users": top_users,
        }

    async def run_maintenance(self):
        """Run periodic maintenance tasks"""

        # Clean up old metrics
        deleted = await self.collector.cleanup_old_metrics(days_to_keep=90)
        logger.info(f"Maintenance: Deleted {deleted} old metric keys")

        return {"deleted_keys": deleted}
