"""JWT 签发与校验。"""

from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

import config


def create_access_token(subject: str, extra: dict[str, Any] | None = None) -> str:
    if not config.JWT_SECRET:
        raise ValueError("JWT_SECRET 未配置")
    payload = {
        "sub": subject,
        "exp": datetime.now(UTC) + timedelta(minutes=config.JWT_EXPIRE_MINUTES),
        "tenant": config.TENANT_ID or "default",
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    return jwt.decode(token, config.JWT_SECRET, algorithms=[config.JWT_ALGORITHM])
