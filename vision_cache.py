"""视觉描述缓存：相同图片内容不重复调用视觉模型。"""

import hashlib
import json
from pathlib import Path
from typing import Any

import config
from logger import get_logger

log = get_logger()


def _cache_path() -> Path:
    return config.STORAGE_DIR / "vision_cache.json"


def _image_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_cache() -> dict[str, Any]:
    path = _cache_path()
    if not path.exists():
        return {"version": 1, "model": None, "entries": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_cache(cache: dict[str, Any]) -> None:
    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path().write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_cached_description(data: bytes) -> str | None:
    if not config.USE_VISION_CACHE:
        return None
    cache = load_cache()
    if cache.get("model") != config.VISION_MODEL:
        return None
    entry = cache.get("entries", {}).get(_image_hash(data))
    if entry:
        log.debug("视觉缓存命中 (%d 字)", len(entry.get("text", "")))
        return entry.get("text")
    return None


def set_cached_description(data: bytes, description: str) -> None:
    if not config.USE_VISION_CACHE or not description.strip():
        return
    cache = load_cache()
    cache["model"] = config.VISION_MODEL
    cache.setdefault("entries", {})[_image_hash(data)] = {
        "text": description,
        "len": len(description),
    }
    save_cache(cache)
    log.debug("视觉描述已缓存")


def clear_cache() -> None:
    path = _cache_path()
    if path.exists():
        path.unlink()
    log.info("已清空视觉描述缓存")
