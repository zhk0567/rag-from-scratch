"""Ollama 嵌入 API 封装。"""

import numpy as np
import ollama

import config


def _client():
    return ollama.Client(host=config.OLLAMA_HOST)


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """
    批量获取文本嵌入向量，返回 shape (N, dim) 的 float32 数组。
    """
    if not texts:
        return np.array([], dtype=np.float32).reshape(0, 0)

    client = _client()
    all_vectors: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            for text in batch:
                resp = client.embeddings(model=config.EMBED_MODEL, prompt=text)
                all_vectors.append(resp["embedding"])
        except Exception as e:
            raise RuntimeError(
                f"嵌入失败。请确认 Ollama 已启动（{config.OLLAMA_HOST}），"
                f"且已执行: ollama pull {config.EMBED_MODEL}\n原始错误: {e}"
            ) from e

    return np.array(all_vectors, dtype=np.float32)
