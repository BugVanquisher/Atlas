import httpx
from .config import settings


class UpstreamClient:
    def __init__(self):
        self.client = httpx.AsyncClient(base_url=settings.UPSTREAM_BASE_URL, timeout=60)

    async def close(self):
        await self.client.aclose()

    async def forward(self, method: str, path: str, headers: dict, body: bytes | None):
        upstream_headers = {k: v for k, v in headers.items()
                            if k.lower() not in {"host", "content-length"}}
        r = await self.client.request(
            method, f"/v1/{path}", headers=upstream_headers, content=body
        )
        return r