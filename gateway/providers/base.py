# gateway/providers/base.py - Base provider interface and common classes

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ProviderStatus(Enum):
    """Provider health status"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"


class ProviderError(Exception):
    """Base exception for provider-related errors"""

    def __init__(self, message: str, provider: str, error_code: Optional[str] = None):
        self.provider = provider
        self.error_code = error_code
        super().__init__(f"[{provider}] {message}")


@dataclass
class ModelCapabilities:
    """Model capabilities and constraints"""

    max_tokens: int
    supports_streaming: bool = True
    supports_functions: bool = False
    supports_vision: bool = False
    context_window: int = 4096
    cost_per_1k_tokens: float = 0.002

    # Performance characteristics
    avg_latency_ms: float = 1000.0
    tokens_per_second: float = 20.0

    # Additional metadata
    model_family: str = "unknown"
    provider_specific: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PricingInfo:
    """Provider pricing information"""

    input_cost_per_1k: float
    output_cost_per_1k: float
    currency: str = "USD"

    # Optional pricing tiers
    volume_discounts: Dict[int, float] = field(default_factory=dict)  # {min_tokens: discount_rate}

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate total cost for a request"""
        input_cost = (input_tokens / 1000) * self.input_cost_per_1k
        output_cost = (output_tokens / 1000) * self.output_cost_per_1k
        return input_cost + output_cost


@dataclass
class HealthMetrics:
    """Provider health and performance metrics"""

    status: ProviderStatus
    last_check: float
    response_time_ms: float
    success_rate: float  # 0.0 to 1.0
    error_rate: float  # 0.0 to 1.0

    # Request statistics (recent window)
    requests_per_minute: float = 0.0
    avg_tokens_per_request: float = 0.0

    # Error details
    recent_errors: List[str] = field(default_factory=list)
    consecutive_failures: int = 0


@dataclass
class ChatRequest:
    """Standardized chat completion request"""

    messages: List[Dict[str, str]]
    model: str
    max_tokens: Optional[int] = None
    temperature: float = 1.0
    stream: bool = False

    # Optional parameters
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None

    # Provider-specific parameters
    provider_params: Dict[str, Any] = field(default_factory=dict)

    # Request metadata
    user_id: Optional[str] = None
    priority: str = "normal"
    timeout: float = 30.0


@dataclass
class ChatResponse:
    """Standardized chat completion response"""

    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None

    # Provider metadata
    provider: str = ""
    provider_response_time: float = 0.0
    provider_model: str = ""

    # Cost information
    estimated_cost: Optional[float] = None

    def get_content(self) -> str:
        """Get the main response content"""
        if self.choices and len(self.choices) > 0:
            choice = self.choices[0]
            if "message" in choice:
                return choice["message"].get("content", "")
            elif "delta" in choice:
                return choice["delta"].get("content", "")
        return ""

    def get_usage_tokens(self) -> int:
        """Get total token usage"""
        if self.usage:
            return self.usage.get("total_tokens", 0)
        return 0


class ProviderInterface(ABC):
    """Abstract base class for all LLM providers"""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.health_metrics = HealthMetrics(
            status=ProviderStatus.HEALTHY,
            last_check=time.time(),
            response_time_ms=0.0,
            success_rate=1.0,
            error_rate=0.0,
        )
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_attempt = 0
        self._circuit_breaker_timeout = 60  # seconds

    @abstractmethod
    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """Execute a chat completion request"""
        pass

    @abstractmethod
    async def chat_completion_stream(self, request: ChatRequest):
        """Execute a streaming chat completion request"""
        pass

    @abstractmethod
    async def health_check(self) -> HealthMetrics:
        """Check provider health and update metrics"""
        pass

    @abstractmethod
    def get_pricing(self, model: str) -> PricingInfo:
        """Get pricing information for a model"""
        pass

    @abstractmethod
    def get_capabilities(self, model: str) -> ModelCapabilities:
        """Get model capabilities and constraints"""
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        pass

    def is_healthy(self) -> bool:
        """Check if provider is currently healthy"""
        return self.health_metrics.status in [ProviderStatus.HEALTHY, ProviderStatus.DEGRADED]

    def is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is preventing requests"""
        if self._circuit_breaker_failures < 5:
            return False

        # Check if timeout has elapsed
        if time.time() - self._circuit_breaker_last_attempt > self._circuit_breaker_timeout:
            self._circuit_breaker_failures = 0
            return False

        return True

    def record_success(self):
        """Record a successful request"""
        self._circuit_breaker_failures = max(0, self._circuit_breaker_failures - 1)

    def record_failure(self):
        """Record a failed request"""
        self._circuit_breaker_failures += 1
        self._circuit_breaker_last_attempt = time.time()

    async def execute_with_circuit_breaker(self, request: ChatRequest) -> ChatResponse:
        """Execute request with circuit breaker protection"""
        if self.is_circuit_breaker_open():
            raise ProviderError(
                f"Circuit breaker open for provider {self.name}", self.name, "CIRCUIT_BREAKER_OPEN"
            )

        try:
            start_time = time.time()
            response = await self.chat_completion(request)

            # Record success and update metrics
            self.record_success()
            response_time = (time.time() - start_time) * 1000
            self.health_metrics.response_time_ms = response_time

            return response

        except Exception as e:
            self.record_failure()
            if isinstance(e, ProviderError):
                raise
            else:
                raise ProviderError(str(e), self.name, "EXECUTION_ERROR")

    def estimate_cost(self, request: ChatRequest, response: Optional[ChatResponse] = None) -> float:
        """Estimate the cost of a request"""
        try:
            pricing = self.get_pricing(request.model)

            if response and response.usage:
                # Use actual token counts
                input_tokens = response.usage.get("prompt_tokens", 0)
                output_tokens = response.usage.get("completion_tokens", 0)
            else:
                # Estimate based on request
                input_tokens = sum(
                    len(msg.get("content", "").split()) * 1.3 for msg in request.messages
                )
                output_tokens = request.max_tokens or 100

            return pricing.calculate_cost(int(input_tokens), int(output_tokens))

        except Exception as e:
            logger.warning(f"Failed to estimate cost for {self.name}: {e}")
            return 0.0

    def __str__(self) -> str:
        return f"{self.name} ({self.health_metrics.status.value})"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"


class BaseProvider(ProviderInterface):
    """Base provider implementation with common functionality"""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.base_url = config.get("base_url", "")
        self.api_key = config.get("api_key", "")
        self.timeout = config.get("timeout", 30.0)
        self.max_retries = config.get("max_retries", 3)

        # Rate limiting
        self.rate_limit_rpm = config.get("rate_limit_rpm", 0)  # 0 = no limit
        self.rate_limit_tpm = config.get("rate_limit_tpm", 0)  # tokens per minute

        # Performance settings
        self.expected_latency_ms = config.get("expected_latency_ms", 1000)
        self.weight = config.get("weight", 1.0)  # For weighted routing

    async def health_check(self) -> HealthMetrics:
        """Default health check implementation"""
        try:
            start_time = time.time()

            # Create a minimal test request
            test_request = ChatRequest(
                messages=[{"role": "user", "content": "Hello"}],
                model=self.get_available_models()[0] if self.get_available_models() else "default",
                max_tokens=1,
            )

            # Try to execute the request
            await self.chat_completion(test_request)

            # Update health metrics
            response_time = (time.time() - start_time) * 1000
            self.health_metrics.status = ProviderStatus.HEALTHY
            self.health_metrics.response_time_ms = response_time
            self.health_metrics.last_check = time.time()

        except Exception as e:
            logger.warning(f"Health check failed for {self.name}: {e}")
            self.health_metrics.status = ProviderStatus.UNHEALTHY
            self.health_metrics.last_check = time.time()
            self.health_metrics.recent_errors.append(str(e))

            # Keep only recent errors (last 10)
            self.health_metrics.recent_errors = self.health_metrics.recent_errors[-10:]

        return self.health_metrics
