"""可选 CLIP 双索引：图文跨模态检索，与文本向量 RRF 融合。"""

import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .logger import get_logger
from .multimodal import IMAGE_EXTENSIONS
from .retrieval import rrf_merge

log = get_logger()

_model = None


def _get_clip_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "CLIP 需要: pip install -r requirements/clip.txt"
            ) from e
        log.info("加载 CLIP 模型: %s", config.CLIP_MODEL)
        _model = SentenceTransformer(config.CLIP_MODEL)
    return _model


class ClipIndex:
    def __init__(self) -> None:
        self.ids: list[str] = []
        self.embeddings: np.ndarray = np.array([], dtype=np.float32).reshape(0, 0)
        self.metadatas: list[dict[str, Any]] = []

    @property
    def size(self) -> int:
        return len(self.ids)

    def add_image(self, path: Path, source: str) -> None:
        model = _get_clip_model()
        from PIL import Image

        text = f"[图片] {source}"
        if config.USE_MULTIMODAL:
            from multimodal import describe_image

            desc = describe_image(path)
            if desc:
                text = desc

        vec = model.encode(Image.open(path), convert_to_numpy=True).astype(np.float32)
        self.ids.append(str(uuid.uuid4()))
        self.metadatas.append(
            {
                "source": source,
                "path": str(path),
                "modality": "image",
                "chunk_index": -1,
                "text": text,
            }
        )
        if self.embeddings.size == 0:
            self.embeddings = vec.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, vec.reshape(1, -1)])

    def search_text_query(self, query: str, top_k: int) -> list[tuple[float, dict[str, Any]]]:
        if self.size == 0:
            return []
        model = _get_clip_model()
        q = model.encode([query], convert_to_numpy=True).astype(np.float32).reshape(1, -1)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            q_norm = 1e-10
        sims = (self.embeddings / norms) @ (q / q_norm).T
        sims = sims.flatten()
        k = min(top_k, self.size)
        idxs = np.argsort(sims)[::-1][:k]
        return [(float(sims[i]), dict(self.metadatas[i])) for i in idxs]

    def save(self, storage_dir: Path) -> None:
        if self.size == 0:
            return
        storage_dir.mkdir(parents=True, exist_ok=True)
        np.save(storage_dir / "clip_embeddings.npy", self.embeddings)
        data = {"ids": self.ids, "metadatas": self.metadatas, "model": config.CLIP_MODEL}
        (storage_dir / "clip_index.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, storage_dir: Path) -> bool:
        emb = storage_dir / "clip_embeddings.npy"
        idx = storage_dir / "clip_index.json"
        if not emb.exists() or not idx.exists():
            return False
        self.embeddings = np.load(emb)
        data = json.loads(idx.read_text(encoding="utf-8"))
        if data.get("model") != config.CLIP_MODEL:
            log.warning("CLIP 模型与索引不一致，请重建索引")
            return False
        self.ids = data["ids"]
        self.metadatas = data["metadatas"]
        return True

    def clear_files(self, storage_dir: Path) -> None:
        for name in ("clip_embeddings.npy", "clip_index.json"):
            p = storage_dir / name
            if p.exists():
                p.unlink()


def build_clip_index(data_dir: Path | None = None) -> int:
    """为 data/ 下所有图片建立 CLIP 向量索引。"""
    if not config.USE_CLIP:
        return 0

    data_dir = data_dir or config.DATA_DIR
    index = ClipIndex()
    count = 0
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            rel = str(path.relative_to(data_dir)).replace("\\", "/")
            try:
                index.add_image(path, rel)
                count += 1
            except Exception as e:
                log.warning("CLIP 索引跳过 %s: %s", path.name, e)

    index.save(config.STORAGE_DIR)
    log.info("CLIP 索引完成: %d 张图片", count)
    return count


def search_clip(query: str, top_k: int) -> list[tuple[float, dict[str, Any]]]:
    index = ClipIndex()
    if not index.load(config.STORAGE_DIR):
        return []
    return index.search_text_query(query, top_k)


def merge_with_text_hits(
    text_hits: list[tuple[float, dict[str, Any]]],
    clip_hits: list[tuple[float, dict[str, Any]]],
    top_k: int,
) -> list[tuple[float, dict[str, Any]]]:
    if not clip_hits:
        return text_hits
    merged = rrf_merge(text_hits, clip_hits)
    return merged[:top_k]
