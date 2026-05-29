"""store_factory 单元测试。"""

import config
from store_factory import create_store, store_exists


def test_create_local_store():
    orig = config.VECTOR_BACKEND
    config.VECTOR_BACKEND = "local"
    try:
        store = create_store()
        assert store.supports_hybrid() is True
        assert store_exists() is store_exists()
    finally:
        config.VECTOR_BACKEND = orig
