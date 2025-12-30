"""Sentinel API client for safety supervision."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx

from .config import settings


class Verdict(str, Enum):
    """Safety supervision verdict."""
    PASS = "pass"
    FAIL = "fail"
    FIX = "fix"


@dataclass
class SupervisionResult:
    """Result from Sentinel supervision API."""
    verdict: Verdict
    reasons: list[str]
    confidence: float
    metadata: dict[str, Any] | None
    tiers_invoked: list[int]
    highest_tier: int
    used_deep_ml: bool

    @property
    def is_blocked(self) -> bool:
        """Returns True if the output should be blocked."""
        return self.verdict == Verdict.FAIL

    @property
    def fixed_output(self) -> str | None:
        """Returns the redacted/fixed output if available."""
        if self.verdict == Verdict.FIX and self.metadata:
            return self.metadata.get("redacted_output")
        return None


class SentinelClient:
    """Client for Sentinel safety supervision API."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int | None = None,
        api_key: str | None = None,
    ):
        """Initialize the Sentinel client.

        Args:
            base_url: Sentinel API URL (defaults to settings.SENTINEL_URL)
            timeout: Request timeout in seconds (defaults to settings.SENTINEL_TIMEOUT)
            api_key: API key for Sentinel (defaults to internal key)
        """
        import os

        self.base_url = base_url or settings.SENTINEL_URL
        self.timeout = timeout or settings.SENTINEL_TIMEOUT
        # Use internal demo key for Atlas-Sentinel communication
        self.api_key = api_key or os.getenv("SENTINEL_INTERNAL_KEY", "aether-internal-key")
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={"x-api-key": self.api_key},
        )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def supervise(
        self,
        prompt: str,
        draft: str,
        context: dict[str, Any] | None = None,
    ) -> SupervisionResult:
        """Supervise an LLM output for safety.

        Args:
            prompt: The original user prompt
            draft: The LLM's draft response to check
            context: Optional additional context

        Returns:
            SupervisionResult with verdict and details

        Raises:
            httpx.HTTPError: If the request fails
        """
        payload = {
            "prompt": prompt,
            "draft": draft,
            "context": context or {},
        }

        response = await self.client.post(
            "/supervise",
            json=payload,
        )
        response.raise_for_status()

        data = response.json()
        return SupervisionResult(
            verdict=Verdict(data["verdict"]),
            reasons=data.get("reasons", []),
            confidence=data.get("confidence", 0.0),
            metadata=data.get("metadata"),
            tiers_invoked=data.get("tiers_invoked", [1]),
            highest_tier=data.get("highest_tier", 1),
            used_deep_ml=data.get("used_deep_ml", False),
        )

    async def health_check(self) -> bool:
        """Check if Sentinel is healthy.

        Returns:
            True if Sentinel is healthy, False otherwise
        """
        try:
            response = await self.client.get("/health")
            return response.status_code == 200
        except Exception:
            return False
