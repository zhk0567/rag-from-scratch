"""API 鉴权：API Key / JWT。"""

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

import config
from jwt_auth import decode_access_token

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_bearer = HTTPBearer(auto_error=False)


def _auth_enabled() -> bool:
    mode = config.AUTH_MODE.lower()
    if mode == "none":
        return False
    if mode == "api_key" and not config.API_KEY:
        return False
    if mode == "jwt" and not config.JWT_SECRET:
        return False
    if mode == "both" and not config.API_KEY and not config.JWT_SECRET:
        return False
    return True


def require_auth(
    api_key: str | None = Security(_api_key_header),
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    if not _auth_enabled():
        return

    mode = config.AUTH_MODE.lower()
    errors: list[str] = []

    if mode in ("api_key", "both") and config.API_KEY:
        if api_key == config.API_KEY:
            return
        errors.append("invalid API key")

    if mode in ("jwt", "both") and config.JWT_SECRET:
        if credentials and credentials.scheme.lower() == "bearer":
            try:
                decode_access_token(credentials.credentials)
                return
            except Exception:
                errors.append("invalid JWT")
        else:
            errors.append("missing Bearer token")

    raise HTTPException(status_code=401, detail="; ".join(errors) or "Unauthorized")


# 兼容旧依赖名
require_api_key = require_auth
