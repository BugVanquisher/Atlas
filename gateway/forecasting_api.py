import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from forecasting.core import ForecastHorizon
from forecasting.metrics_collector import ForecastingIntegration

from .auth import extract_api_key
from .config import settings

# Query parameter definitions to avoid B008 bandit warnings
HORIZON_QUERY = Query(default="daily", description="Forecast horizon")
DAYS_BACK_QUERY = Query(default=30, ge=1, le=90, description="Days of history to analyze")
DATE_QUERY = Query(default=None, description="Date in YYYY-MM-DD format")
ADMIN_HORIZON_QUERY = Query(default="daily", description="Forecast horizon")
LIMIT_QUERY = Query(default=50, ge=1, le=200, description="Max users to forecast")

# Early Depends constants (those that don't reference functions defined later)
API_KEY_DEPENDS = Depends(extract_api_key)
ADMIN_KEY_DEPENDS = Depends(lambda: None)

logger = logging.getLogger(__name__)

# Create router for forecasting endpoints
forecasting_router = APIRouter(prefix="/v1/forecasting", tags=["forecasting"])


class ForecastRequest(BaseModel):
    """Request model for generating forecasts"""

    horizon: str = Field(
        default="daily", description="Forecast horizon: hourly, daily, weekly, monthly"
    )
    include_patterns: bool = Field(default=True, description="Include usage pattern analysis")
    include_recommendations: bool = Field(
        default=True, description="Include capacity recommendations"
    )


class CapacityRecommendation(BaseModel):
    """Response model for capacity recommendations"""

    current_daily_limit: int
    current_monthly_limit: int
    recommended_daily_limit: int
    recommended_monthly_limit: int
    recommended_rate_limit: float
    capacity_utilization: float
    overrun_probability: float
    growth_rate: float
    confidence_level: str


class UsagePatterns(BaseModel):
    """Response model for usage patterns"""

    peak_hour: int
    peak_weekday: int
    seasonality_detected: bool
    trend_direction: str
    trend_strength: float


class ForecastResponse(BaseModel):
    """Response model for forecast results"""

    api_key: str
    forecast_horizon: str
    generated_at: str
    forecast_period: Dict[str, str]

    predictions: Dict[str, Any]
    capacity_recommendations: Optional[CapacityRecommendation] = None
    usage_patterns: Optional[UsagePatterns] = None

    confidence_intervals: Dict[str, Any]
    risk_assessment: Dict[str, Any]


# Dependency to get forecasting integration
async def get_forecasting_integration():
    """Dependency to provide forecasting integration"""
    # This should be injected from your main app
    # For now, we'll create it here - in production, pass it from main.py
    from gateway.main import redis

    return ForecastingIntegration(redis)


# Depends constants that reference functions defined above
FORECASTING_INTEGRATION_DEPENDS = Depends(get_forecasting_integration)


@forecasting_router.get("/forecast", response_model=ForecastResponse)
async def get_forecast(
    api_key: str = API_KEY_DEPENDS,
    integration: ForecastingIntegration = FORECASTING_INTEGRATION_DEPENDS,
    horizon: str = HORIZON_QUERY,
):
    """
    Get traffic forecast and capacity recommendations for your API key.

    **Horizons:**
    - `hourly`: Next 24 hours
    - `daily`: Next 7 days
    - `weekly`: Next 4 weeks
    - `monthly`: Next 3 months
    """

    try:
        # Validate horizon
        horizon_mapping = {
            "hourly": ForecastHorizon.HOURLY,
            "daily": ForecastHorizon.DAILY,
            "weekly": ForecastHorizon.WEEKLY,
            "monthly": ForecastHorizon.MONTHLY,
        }

        if horizon not in horizon_mapping:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid horizon. Must be one of: {list(horizon_mapping.keys())}",
            )

        # Get current limits
        from gateway.main import quota

        current_limits = await quota.get_limits(api_key)

        # Generate forecast
        forecast_data = await integration.get_forecast_for_api_key(
            api_key, horizon_mapping[horizon], current_limits
        )

        # Transform to response model
        response = ForecastResponse(
            api_key=api_key,
            forecast_horizon=horizon,
            generated_at=datetime.now(timezone.utc).isoformat(),
            forecast_period={
                "start": forecast_data["forecast_start"],
                "end": forecast_data["forecast_end"],
            },
            predictions={
                "tokens": forecast_data["predicted_tokens"],
                "requests": forecast_data["predicted_requests"],
                "costs": forecast_data["predicted_costs"],
                "timestamps": forecast_data["timestamps"],
            },
            confidence_intervals={
                "tokens_lower": forecast_data["tokens_lower_bound"],
                "tokens_upper": forecast_data["tokens_upper_bound"],
            },
            risk_assessment={
                "overrun_probability": forecast_data["overrun_probability"],
                "growth_rate": forecast_data["growth_rate"],
                "trend_direction": forecast_data["trend_direction"],
                "trend_strength": forecast_data["trend_strength"],
            },
        )

        # Add capacity recommendations
        response.capacity_recommendations = CapacityRecommendation(
            current_daily_limit=current_limits["daily_limit"],
            current_monthly_limit=current_limits["monthly_limit"],
            recommended_daily_limit=forecast_data["recommended_daily_limit"],
            recommended_monthly_limit=forecast_data["recommended_monthly_limit"],
            recommended_rate_limit=forecast_data["recommended_rate_limit"],
            capacity_utilization=forecast_data["capacity_utilization"],
            overrun_probability=forecast_data["overrun_probability"],
            growth_rate=forecast_data["growth_rate"],
            confidence_level="95%",
        )

        # Add usage patterns if available
        patterns = forecast_data.get("usage_patterns", {})
        if patterns:
            response.usage_patterns = UsagePatterns(
                peak_hour=patterns.get("peak_hour", 12),
                peak_weekday=patterns.get("peak_weekday", 1),
                seasonality_detected=forecast_data["seasonality_detected"],
                trend_direction=forecast_data["trend_direction"],
                trend_strength=forecast_data["trend_strength"],
            )

        return response

    except HTTPException:
        # Re-raise HTTP exceptions (like 400 validation errors)
        raise
    except Exception as e:
        logger.error(f"Error generating forecast for {api_key}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate forecast")


@forecasting_router.get("/usage-patterns")
async def get_usage_patterns(
    days_back: int = DAYS_BACK_QUERY,
    api_key: str = API_KEY_DEPENDS,
    integration: ForecastingIntegration = FORECASTING_INTEGRATION_DEPENDS,
):
    """
    Get detailed usage patterns and analytics for your API key.
    """

    try:
        patterns = await integration.predictor.get_usage_patterns(api_key, days_back)

        if not patterns:
            return {"message": "No usage data found for the specified period", "days_analyzed": 0}

        return {
            "api_key": api_key,
            "analysis_period": {
                "days_back": days_back,
                "days_analyzed": patterns["summary"]["days_analyzed"],
            },
            "summary": patterns["summary"],
            "peak_usage": patterns["peak_usage"],
            "low_usage": patterns["low_usage"],
            "priority_distribution": patterns["priority_distribution"],
            "route_distribution": patterns["route_distribution"],
        }

    except Exception as e:
        logger.error(f"Error getting usage patterns for {api_key}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve usage patterns")


@forecasting_router.get("/capacity-recommendations")
async def get_capacity_recommendations(
    api_key: str = API_KEY_DEPENDS,
    integration: ForecastingIntegration = FORECASTING_INTEGRATION_DEPENDS,
):
    """
    Get immediate capacity recommendations based on recent usage trends.
    """

    try:
        # Get current limits
        from gateway.main import quota

        current_limits = await quota.get_limits(api_key)

        # Get recent usage data (last 7 days)
        usage_data = await integration.predictor.collect_usage_data(api_key, hours_back=168)

        if not usage_data:
            return {
                "message": "Insufficient usage data for recommendations",
                "current_limits": current_limits,
            }

        # Quick forecast for recommendations
        forecast_data = await integration.get_forecast_for_api_key(
            api_key, ForecastHorizon.DAILY, current_limits
        )

        return {
            "api_key": api_key,
            "current_limits": {
                "daily_limit": current_limits["daily_limit"],
                "monthly_limit": current_limits["monthly_limit"],
                "rate_per_sec": current_limits["rate_per_sec"],
            },
            "recommendations": {
                "daily_limit": forecast_data["recommended_daily_limit"],
                "monthly_limit": forecast_data["recommended_monthly_limit"],
                "rate_per_sec": forecast_data["recommended_rate_limit"],
            },
            "analysis": {
                "current_utilization": forecast_data["capacity_utilization"],
                "overrun_risk": forecast_data["overrun_probability"],
                "growth_rate": forecast_data["growth_rate"],
                "trend": forecast_data["trend_direction"],
            },
            "reasoning": {
                "daily_increase": forecast_data["recommended_daily_limit"]
                - current_limits["daily_limit"],
                "monthly_increase": forecast_data["recommended_monthly_limit"]
                - current_limits["monthly_limit"],
                "confidence": "Based on 7-day trend analysis",
            },
        }

    except Exception as e:
        logger.error(f"Error getting capacity recommendations for {api_key}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate recommendations")


# Admin endpoints for system-wide forecasting
@forecasting_router.get("/admin/system-metrics")
async def get_system_metrics(
    date: Optional[str] = DATE_QUERY,
    x_admin_key: str = ADMIN_KEY_DEPENDS,  # Will be implemented with proper admin auth
    integration: ForecastingIntegration = FORECASTING_INTEGRATION_DEPENDS,
):
    """
    Get system-wide usage metrics and top users.
    Requires admin authentication.
    """

    # TODO: Implement proper admin authentication
    if x_admin_key != settings.ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Admin access required")

    try:
        # Parse date if provided
        target_date = None
        if date:
            try:
                target_date = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

        metrics = await integration.get_system_wide_metrics(target_date)
        return metrics

    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system metrics")


@forecasting_router.post("/admin/maintenance")
async def run_maintenance(
    background_tasks: BackgroundTasks,
    x_admin_key: str = ADMIN_KEY_DEPENDS,
    integration: ForecastingIntegration = FORECASTING_INTEGRATION_DEPENDS,
):
    """
    Run maintenance tasks (cleanup old metrics, etc.).
    Requires admin authentication.
    """

    if x_admin_key != settings.ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Admin access required")

    # Run maintenance in background
    background_tasks.add_task(integration.run_maintenance)

    return {"message": "Maintenance tasks started in background"}


@forecasting_router.get("/admin/forecast-all")
async def forecast_all_users(
    horizon: str = HORIZON_QUERY,
    limit: int = LIMIT_QUERY,
    x_admin_key: str = ADMIN_KEY_DEPENDS,
    integration: ForecastingIntegration = FORECASTING_INTEGRATION_DEPENDS,
):
    """
    Generate forecasts for all active users.
    Requires admin authentication.
    """

    if x_admin_key != settings.ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Admin access required")

    try:
        # Get top users from recent activity
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        top_users = await integration.collector.get_top_users(yesterday, limit=limit)

        horizon_mapping = {
            "hourly": ForecastHorizon.HOURLY,
            "daily": ForecastHorizon.DAILY,
            "weekly": ForecastHorizon.WEEKLY,
            "monthly": ForecastHorizon.MONTHLY,
        }

        if horizon not in horizon_mapping:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid horizon. Must be one of: {list(horizon_mapping.keys())}",
            )

        # Generate forecasts for each user
        forecasts = []
        for user in top_users[:limit]:
            try:
                # Get current limits for user
                from gateway.main import quota

                current_limits = await quota.get_limits(user["api_key"])

                # Generate forecast
                forecast_data = await integration.get_forecast_for_api_key(
                    user["api_key"], horizon_mapping[horizon], current_limits
                )

                forecasts.append(
                    {
                        "api_key": user["api_key"],
                        "current_usage": user,
                        "forecast": {
                            "predicted_tokens": forecast_data["predicted_tokens"],
                            "overrun_probability": forecast_data["overrun_probability"],
                            "growth_rate": forecast_data["growth_rate"],
                            "recommended_daily_limit": forecast_data["recommended_daily_limit"],
                            "capacity_utilization": forecast_data["capacity_utilization"],
                        },
                    }
                )

            except Exception as user_error:
                logger.warning(f"Failed to forecast for user {user['api_key']}: {user_error}")
                continue

        return {
            "horizon": horizon,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_users_analyzed": len(forecasts),
            "forecasts": forecasts,
        }

    except Exception as e:
        logger.error(f"Error generating forecasts for all users: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate system forecasts")
