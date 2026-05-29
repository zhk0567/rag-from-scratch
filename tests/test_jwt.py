"""JWT 单元测试。"""

from rag import config
from rag.jwt_auth import create_access_token, decode_access_token


def test_jwt_roundtrip():
    orig = config.JWT_SECRET
    config.JWT_SECRET = "test-secret-key"
    try:
        token = create_access_token("tester")
        payload = decode_access_token(token)
        assert payload["sub"] == "tester"
    finally:
        config.JWT_SECRET = orig
