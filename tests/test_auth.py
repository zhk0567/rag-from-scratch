"""auth 单元测试。"""

import pytest
from fastapi import HTTPException

import config
from auth import require_auth


def test_auth_disabled_when_mode_none():
    orig_mode, orig_key, orig_jwt = config.AUTH_MODE, config.API_KEY, config.JWT_SECRET
    config.AUTH_MODE = "none"
    config.API_KEY = ""
    config.JWT_SECRET = ""
    try:
        require_auth(None, None)
    finally:
        config.AUTH_MODE, config.API_KEY, config.JWT_SECRET = orig_mode, orig_key, orig_jwt


def test_invalid_api_key_raises():
    orig_mode, orig_key = config.AUTH_MODE, config.API_KEY
    config.AUTH_MODE = "api_key"
    config.API_KEY = "secret"
    try:
        with pytest.raises(HTTPException) as exc:
            require_auth("wrong", None)
        assert exc.value.status_code == 401
    finally:
        config.AUTH_MODE, config.API_KEY = orig_mode, orig_key
