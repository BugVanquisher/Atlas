import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ForecastHorizon(Enum):
    HOURLY = "1h"
    DAILY = "1d"
    WEEKLY = "7d"
    MONTHLY = "30d"


class TrendDirection(Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


@dataclass
class UsageDataPoint:
    """Single usage measurement"""

    timestamp: datetime
    api_key: str
    tokens_used: int
    requests_count: int
    priority: str
    route: str
    model: Optional[str] = None
    cost_estimate: Optional[float] = None


@dataclass
class ForecastResult:
    """Result of a forecasting operation"""

    api_key: str
    horizon: ForecastHorizon
    forecast_start: datetime
    forecast_end: datetime

    # Predictions
    predicted_tokens: List[int]
    predicted_requests: List[int]
    predicted_costs: List[float]
    timestamps: List[datetime]

    # Confidence intervals
    tokens_lower_bound: List[int]
    tokens_upper_bound: List[int]

    # Trend analysis
    trend_direction: TrendDirection
    trend_strength: float  # 0-1 scale
    seasonality_detected: bool

    # Capacity recommendations
    recommended_daily_limit: int
    recommended_monthly_limit: int
    recommended_rate_limit: float
    capacity_utilization: float  # Current usage as % of limits

    # Risk assessment
    overrun_probability: float  # Probability of exceeding current limits
    growth_rate: float  # % growth rate

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        # Convert datetime objects to ISO strings
        result["forecast_start"] = self.forecast_start.isoformat()
        result["forecast_end"] = self.forecast_end.isoformat()
        result["timestamps"] = [ts.isoformat() for ts in self.timestamps]
        result["trend_direction"] = self.trend_direction.value
        result["horizon"] = self.horizon.value
        return result


class TimeSeriesForecaster:
    """Advanced time series forecasting with multiple algorithms"""

    def __init__(self):
        self.min_data_points = 7  # Minimum points needed for forecasting

    def simple_moving_average(self, data: List[float], window: int = 7) -> List[float]:
        """Simple moving average forecast"""
        if len(data) < window:
            return [np.mean(data)] * len(data)

        forecasts = []
        for i in range(len(data)):
            if i < window - 1:
                forecasts.append(np.mean(data[: i + 1]))
            else:
                forecasts.append(np.mean(data[i - window + 1 : i + 1]))

        return forecasts

    def exponential_smoothing(self, data: List[float], alpha: float = 0.3) -> List[float]:
        """Exponential smoothing forecast"""
        if not data:
            return []

        forecasts = [data[0]]
        for i in range(1, len(data)):
            forecast = alpha * data[i - 1] + (1 - alpha) * forecasts[i - 1]
            forecasts.append(forecast)

        return forecasts

    def linear_trend(self, data: List[float]) -> Tuple[float, float]:
        """Calculate linear trend slope and intercept"""
        if len(data) < 2:
            return 0.0, np.mean(data) if data else 0.0

        x = np.arange(len(data))
        slope, intercept = np.polyfit(x, data, 1)
        return slope, intercept

    def detect_seasonality(self, data: List[float], period: int = 24) -> bool:
        """Detect if data has seasonal patterns"""
        if len(data) < period * 2:
            return False

        # Simple autocorrelation check
        try:
            autocorr = np.corrcoef(data[:-period], data[period:])[0, 1]
            return not np.isnan(autocorr) and autocorr > 0.3
        except Exception:
            return False

    def forecast_future(
        self, data: List[float], periods: int, method: str = "exponential"
    ) -> Tuple[List[float], List[float], List[float]]:
        """Forecast future values with confidence intervals"""
        if len(data) < self.min_data_points:
            # Not enough data, return simple extrapolation
            avg = np.mean(data) if data else 0
            std = np.std(data) if len(data) > 1 else avg * 0.2
            forecasts = [avg] * periods
            lower_bounds = [max(0, avg - 2 * std)] * periods
            upper_bounds = [avg + 2 * std] * periods
            return forecasts, lower_bounds, upper_bounds

        if method == "linear":
            slope, intercept = self.linear_trend(data)
            base_index = len(data)
            forecasts = [slope * (base_index + i) + intercept for i in range(periods)]
        else:  # exponential smoothing
            alpha = 0.3
            last_forecast = self.exponential_smoothing(data, alpha)[-1]
            # Use last trend for future predictions
            if len(data) >= 2:
                trend = data[-1] - data[-2]
                forecasts = [max(0, last_forecast + trend * (i + 1)) for i in range(periods)]
            else:
                forecasts = [last_forecast] * periods

        # Calculate confidence intervals based on historical variance
        historical_variance = np.var(data) if len(data) > 1 else np.mean(data) * 0.1
        std_dev = np.sqrt(historical_variance)

        lower_bounds = [max(0, f - 2 * std_dev) for f in forecasts]
        upper_bounds = [f + 2 * std_dev for f in forecasts]

        return forecasts, lower_bounds, upper_bounds


class UsageAnalyzer:
    """Analyze historical usage patterns and generate insights"""

    def __init__(self):
        self.forecaster = TimeSeriesForecaster()

    def analyze_trend(self, data: List[float]) -> Tuple[TrendDirection, float]:
        """Analyze trend direction and strength"""
        if len(data) < 3:
            return TrendDirection.STABLE, 0.0

        slope, intercept = self.forecaster.linear_trend(data)

        # Calculate trend strength based on R-squared
        x = np.arange(len(data))
        y_pred = slope * x + intercept
        ss_res = np.sum((data - y_pred) ** 2)
        ss_tot = np.sum((data - np.mean(data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Determine direction
        avg_change = np.mean(np.diff(data))
        relative_change = abs(avg_change) / (np.mean(data) + 1e-6)  # Avoid division by zero

        if relative_change < 0.05:
            direction = TrendDirection.STABLE
        elif avg_change > 0:
            direction = TrendDirection.INCREASING
        else:
            direction = TrendDirection.DECREASING

        # Check for volatility
        volatility = np.std(data) / (np.mean(data) + 1e-6)
        if volatility > 0.5:
            direction = TrendDirection.VOLATILE

        return direction, min(r_squared, 1.0)

    def calculate_growth_rate(self, data: List[float], periods: int = 7) -> float:
        """Calculate compound growth rate over recent periods"""
        if len(data) < periods + 1:
            return 0.0

        start_value = np.mean(data[-periods - 1 : -periods]) if len(data) > periods else data[0]
        end_value = np.mean(data[-periods:])

        if start_value <= 0:
            return 0.0

        growth_rate = (end_value / start_value) ** (1 / periods) - 1
        return min(max(growth_rate, -0.5), 2.0)  # Cap between -50% and 200%

    def detect_usage_patterns(self, usage_data: List[UsageDataPoint]) -> Dict[str, Any]:
        """Detect patterns in usage data"""
        if not usage_data:
            return {}

        # Group by hour of day for daily patterns
        hourly_usage = {}
        daily_usage = {}

        for point in usage_data:
            hour = point.timestamp.hour
            day = point.timestamp.date()

            if hour not in hourly_usage:
                hourly_usage[hour] = []
            if day not in daily_usage:
                daily_usage[day] = 0

            hourly_usage[hour].append(point.tokens_used)
            daily_usage[day] += point.tokens_used

        # Calculate peak hours
        avg_hourly = {hour: np.mean(tokens) for hour, tokens in hourly_usage.items()}
        peak_hour = max(avg_hourly.keys(), key=lambda h: avg_hourly[h]) if avg_hourly else 12

        # Weekly patterns
        weekly_pattern = {}
        for point in usage_data:
            weekday = point.timestamp.weekday()  # 0=Monday
            if weekday not in weekly_pattern:
                weekly_pattern[weekday] = []
            weekly_pattern[weekday].append(point.tokens_used)

        avg_weekly = {day: np.mean(tokens) for day, tokens in weekly_pattern.items()}
        peak_weekday = max(avg_weekly.keys(), key=lambda d: avg_weekly[d]) if avg_weekly else 1

        return {
            "peak_hour": peak_hour,
            "peak_weekday": peak_weekday,
            "hourly_averages": avg_hourly,
            "weekly_averages": avg_weekly,
            "daily_totals": daily_usage,
        }


class CapacityPlanner:
    """Generate capacity planning recommendations"""

    def __init__(self):
        self.safety_multiplier = 1.3  # 30% safety margin
        self.growth_buffer = 0.2  # 20% growth buffer

    def calculate_recommended_limits(
        self,
        current_limits: Dict[str, int],
        predicted_usage: List[int],
        growth_rate: float,
        horizon_days: int = 30,
    ) -> Dict[str, int]:
        """Calculate recommended quota limits"""

        # Calculate peak predicted usage
        peak_daily = max(predicted_usage) if predicted_usage else 0
        avg_daily = np.mean(predicted_usage) if predicted_usage else 0

        # Project with growth
        projected_peak = peak_daily * (1 + growth_rate * horizon_days / 30)
        projected_avg = avg_daily * (1 + growth_rate * horizon_days / 30)

        # Apply safety margins
        recommended_daily = int(projected_peak * self.safety_multiplier)
        recommended_monthly = int(projected_avg * 30 * self.safety_multiplier)

        # Ensure minimums
        min_daily = max(recommended_daily, 1000)
        min_monthly = max(recommended_monthly, min_daily * 20)

        return {
            "daily_limit": min_daily,
            "monthly_limit": min_monthly,
            "current_daily": current_limits.get("daily_limit", 0),
            "current_monthly": current_limits.get("monthly_limit", 0),
            "utilization_daily": (avg_daily / min_daily) if min_daily > 0 else 0,
            "utilization_monthly": (projected_avg * 30 / min_monthly) if min_monthly > 0 else 0,
        }

    def calculate_rate_limits(
        self, request_patterns: List[int], current_rate: float, peak_factor: float = 2.0
    ) -> Dict[str, float]:
        """Calculate recommended rate limits"""

        if not request_patterns:
            return {"rate_per_sec": current_rate, "burst": int(current_rate * 5)}

        # Calculate peak requests per hour
        peak_hourly = max(request_patterns)
        avg_hourly = np.mean(request_patterns)

        # Convert to per-second with peak handling
        peak_per_sec = peak_hourly / 3600.0
        avg_per_sec = avg_hourly / 3600.0

        # Recommend rate that handles average + burst capability
        recommended_rate = avg_per_sec * 1.5  # 50% above average
        recommended_burst = int(peak_per_sec * peak_factor)

        return {
            "rate_per_sec": max(recommended_rate, 0.1),
            "burst": max(recommended_burst, 5),
            "current_rate": current_rate,
            "peak_utilization": peak_per_sec / max(current_rate, 0.1),
        }

    def assess_overrun_risk(
        self,
        predicted_usage: List[int],
        current_limits: Dict[str, int],
        confidence_upper: List[int],
    ) -> float:
        """Assess probability of quota overrun"""

        daily_limit = current_limits.get("daily_limit", float("inf"))

        if not predicted_usage or daily_limit == float("inf"):
            return 0.0

        # Count how many predictions exceed limits
        overruns = sum(1 for usage in confidence_upper if usage > daily_limit)
        risk_probability = overruns / len(confidence_upper)

        # Factor in trend
        if len(predicted_usage) >= 2:
            trend = (predicted_usage[-1] - predicted_usage[0]) / len(predicted_usage)
            if trend > 0:
                risk_probability += min(trend / daily_limit, 0.3)

        return min(risk_probability, 1.0)


class TrafficForecaster:
    """Main forecasting orchestrator"""

    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.analyzer = UsageAnalyzer()
        self.planner = CapacityPlanner()

    async def collect_usage_data(
        self, api_key: str, hours_back: int = 168  # 7 days
    ) -> List[UsageDataPoint]:
        """Collect historical usage data from Redis and metrics"""
        # This would typically query your metrics/logging system
        # For now, we'll simulate some data collection from Redis

        data_points = []
        now = datetime.now(timezone.utc)

        # In a real implementation, you'd query your metrics system here
        # For demonstration, we'll create some sample data
        for i in range(hours_back):
            timestamp = now - timedelta(hours=i)
            # This would come from your actual metrics
            data_points.append(
                UsageDataPoint(
                    timestamp=timestamp,
                    api_key=api_key,
                    tokens_used=100 + i * 2,  # Simulated growing usage
                    requests_count=10 + i // 24,
                    priority="normal",
                    route="chat_completions",
                )
            )

        return sorted(data_points, key=lambda x: x.timestamp)

    async def generate_forecast(
        self,
        api_key: str,
        horizon: ForecastHorizon = ForecastHorizon.DAILY,
        current_limits: Optional[Dict[str, Any]] = None,
    ) -> ForecastResult:
        """Generate comprehensive forecast for an API key"""

        # Collect historical data
        hours_back = {
            ForecastHorizon.HOURLY: 48,
            ForecastHorizon.DAILY: 168,  # 7 days
            ForecastHorizon.WEEKLY: 720,  # 30 days
            ForecastHorizon.MONTHLY: 2160,  # 90 days
        }[horizon]

        usage_data = await self.collect_usage_data(api_key, hours_back)

        if not usage_data:
            # Return empty forecast if no data
            now = datetime.now(timezone.utc)
            return ForecastResult(
                api_key=api_key,
                horizon=horizon,
                forecast_start=now,
                forecast_end=now + timedelta(hours=24),
                predicted_tokens=[],
                predicted_requests=[],
                predicted_costs=[],
                timestamps=[],
                tokens_lower_bound=[],
                tokens_upper_bound=[],
                trend_direction=TrendDirection.STABLE,
                trend_strength=0.0,
                seasonality_detected=False,
                recommended_daily_limit=1000,
                recommended_monthly_limit=30000,
                recommended_rate_limit=2.0,
                capacity_utilization=0.0,
                overrun_probability=0.0,
                growth_rate=0.0,
            )

        # Extract time series data
        token_series = [point.tokens_used for point in usage_data]
        request_series = [point.requests_count for point in usage_data]

        # Analyze trends
        trend_direction, trend_strength = self.analyzer.analyze_trend(token_series)
        growth_rate = self.analyzer.calculate_growth_rate(token_series)
        seasonality = self.analyzer.forecaster.detect_seasonality(token_series)

        # Generate forecasts
        forecast_periods = {
            ForecastHorizon.HOURLY: 24,
            ForecastHorizon.DAILY: 7,
            ForecastHorizon.WEEKLY: 4,
            ForecastHorizon.MONTHLY: 3,
        }[horizon]

        predicted_tokens, tokens_lower, tokens_upper = self.analyzer.forecaster.forecast_future(
            token_series, forecast_periods
        )
        predicted_requests, _, _ = self.analyzer.forecaster.forecast_future(
            request_series, forecast_periods
        )

        # Calculate timestamps
        period_delta = {
            ForecastHorizon.HOURLY: timedelta(hours=1),
            ForecastHorizon.DAILY: timedelta(days=1),
            ForecastHorizon.WEEKLY: timedelta(weeks=1),
            ForecastHorizon.MONTHLY: timedelta(days=30),
        }[horizon]

        start_time = usage_data[-1].timestamp if usage_data else datetime.now(timezone.utc)
        timestamps = [start_time + period_delta * (i + 1) for i in range(forecast_periods)]

        # Estimate costs (assuming $0.01 per 1000 tokens)
        predicted_costs = [tokens * 0.00001 for tokens in predicted_tokens]

        # Generate capacity recommendations
        current_limits = current_limits or {"daily_limit": 10000, "monthly_limit": 300000}

        limit_recommendations = self.planner.calculate_recommended_limits(
            current_limits, predicted_tokens, growth_rate
        )

        rate_recommendations = self.planner.calculate_rate_limits(
            predicted_requests, current_limits.get("rate_per_sec", 2.0)
        )

        overrun_risk = self.planner.assess_overrun_risk(
            predicted_tokens, current_limits, tokens_upper
        )

        # Calculate capacity utilization
        avg_predicted = np.mean(predicted_tokens) if predicted_tokens else 0
        capacity_utilization = avg_predicted / current_limits.get("daily_limit", 1)

        return ForecastResult(
            api_key=api_key,
            horizon=horizon,
            forecast_start=start_time,
            forecast_end=timestamps[-1] if timestamps else start_time,
            predicted_tokens=predicted_tokens,
            predicted_requests=predicted_requests,
            predicted_costs=predicted_costs,
            timestamps=timestamps,
            tokens_lower_bound=tokens_lower,
            tokens_upper_bound=tokens_upper,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            seasonality_detected=seasonality,
            recommended_daily_limit=limit_recommendations["daily_limit"],
            recommended_monthly_limit=limit_recommendations["monthly_limit"],
            recommended_rate_limit=rate_recommendations["rate_per_sec"],
            capacity_utilization=capacity_utilization,
            overrun_probability=overrun_risk,
            growth_rate=growth_rate,
        )
