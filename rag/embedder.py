"""Ollama 嵌入 API 封装（支持并发）。"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import ollama

from . import config
from .logger import get_logger

log = get_logger()


def _client():
    return ollama.Client(host=config.OLLAMA_HOST)


def _embed_one(text: str) -> list[float]:
    resp = _client().embeddings(model=config.EMBED_MODEL, prompt=text)
    return resp["embedding"]


def embed_texts(texts: list[str], batch_size: int | None = None) -> np.ndarray:
    if not texts:
        return np.array([], dtype=np.float32).reshape(0, 0)

    workers = min(config.EMBED_WORKERS, len(texts))
    all_vectors: list[list[float] | None] = [None] * len(texts)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_embed_one, t): i for i, t in enumerate(texts)}
            for fut in as_completed(futures):
                idx = futures[fut]
                all_vectors[idx] = fut.result()
    except Exception as e:
        raise RuntimeError(
            f"嵌入失败。请确认 Ollama 已启动（{config.OLLAMA_HOST}），"
            f"且已执行: ollama pull {config.EMBED_MODEL}\n原始错误: {e}"
        ) from e

    log.debug("嵌入 %d 条文本 (workers=%d)", len(texts), workers)
    return np.array(all_vectors, dtype=np.float32)
