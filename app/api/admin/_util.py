from fastapi import HTTPException, Request
from ...core.config import settings

def require_admin(req: Request) -> None:
    token = req.headers.get("authorization") or req.headers.get("Authorization")
    expected = settings.admin_token.strip()
    if not expected:
        raise HTTPException(401, "ADMIN_TOKEN is not configured")
    if not token or not token.lower().startswith("bearer "):
        raise HTTPException(401, "Missing Bearer token")
    value = token.split(" ", 1)[1].strip()
    if value != expected:
        raise HTTPException(403, "Invalid admin token")

