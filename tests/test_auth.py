"""auth 单元测试。"""

import pytest
from fastapi import HTTPException

import config
from auth import require_api_key


def test_no_api_key_config_allows():
    original = config.API_KEY
    config.API_KEY = ""
    try:
        require_api_key(None)
    finally:
        config.API_KEY = original


def test_invalid_api_key_raises():
    original = config.API_KEY
    config.API_KEY = "secret"
    try:
        with pytest.raises(HTTPException) as exc:
            require_api_key("wrong")
        assert exc.value.status_code == 401
    finally:
        config.API_KEY = original
