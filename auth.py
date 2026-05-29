"""API 鉴权：可选 X-API-Key 头。"""

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

import config

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    """未配置 API_KEY 时不启用鉴权。"""
    if not config.API_KEY:
        return
    if api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
