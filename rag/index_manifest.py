"""索引清单：跟踪 data/ 文件变更，支持增量建索引与版本信息。"""

import hashlib
import json
from pathlib import Path
from typing import Any

from . import config
from .loader import SUPPORTED_EXTENSIONS


def _manifest_path() -> Path:
    return config.STORAGE_DIR / "manifest.json"


def file_signature(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "hash": hashlib.md5(data).hexdigest(),
        "size": len(data),
        "mtime": path.stat().st_mtime,
    }


def scan_data_dir(data_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    data_dir = data_dir or config.DATA_DIR
    files: dict[str, dict[str, Any]] = {}
    if not data_dir.exists():
        return files
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            rel = str(path.relative_to(data_dir)).replace("\\", "/")
            files[rel] = file_signature(path)
    return files


def load_manifest() -> dict[str, Any]:
    path = _manifest_path()
    if not path.exists():
        return {"files": {}, "embed_model": None, "dim": 0}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(
    files: dict[str, dict[str, Any]],
    embed_model: str | None = None,
    dim: int = 0,
) -> None:
    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "files": files,
        "embed_model": embed_model or config.EMBED_MODEL,
        "dim": dim,
    }
    _manifest_path().write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def diff_files(
    old_files: dict[str, dict[str, Any]],
    new_files: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    """返回 (新增, 修改, 删除) 的相对路径列表。"""
    old_keys = set(old_files)
    new_keys = set(new_files)
    added = sorted(new_keys - old_keys)
    deleted = sorted(old_keys - new_keys)
    modified = sorted(
        k for k in old_keys & new_keys if old_files[k].get("hash") != new_files[k].get("hash")
    )
    return added, modified, deleted


def check_index_version() -> tuple[bool, str]:
    """检查当前嵌入模型是否与 manifest 一致。"""
    manifest = load_manifest()
    stored = manifest.get("embed_model")
    if not stored:
        return True, ""
    if stored != config.EMBED_MODEL:
        return False, (
            f"索引由模型 `{stored}` 构建，当前配置为 `{config.EMBED_MODEL}`，"
            "请执行 `python ingest.py --rebuild` 重建索引。"
        )
    return True, ""
