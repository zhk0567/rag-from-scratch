"""vision_cache 单元测试。"""

from rag import config
from rag.vision_cache import get_cached_description, set_cached_description, load_cache


def test_vision_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(config, "USE_VISION_CACHE", True)
    monkeypatch.setattr(config, "VISION_MODEL", "test-model")

    data = b"fake-image-bytes"
    assert get_cached_description(data) is None
    set_cached_description(data, "一张包含流程图的图片")
    assert get_cached_description(data) == "一张包含流程图的图片"

    cache = load_cache()
    assert cache["model"] == "test-model"
    assert len(cache["entries"]) == 1
