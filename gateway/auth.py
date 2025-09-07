from fastapi import Header, HTTPException

AUTH_HEADER = Header(default=None)
API_KEY_HEADER = Header(default=None)


async def extract_api_key(
    authorization: str | None = AUTH_HEADER,
    x_api_key: str | None = API_KEY_HEADER,
) -> str:
    if x_api_key:
        return x_api_key.strip()
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    raise HTTPException(status_code=401, detail="API key required")
