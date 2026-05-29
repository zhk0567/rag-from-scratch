"""向量存储工厂：local / qdrant / milvus。"""

from typing import Any, Protocol

import numpy as np

from . import config
from .chunker import Chunk


class VectorStoreBackend(Protocol):
    @property
    def size(self) -> int: ...

    @property
    def embedding_dim(self) -> int: ...

    def clear(self) -> None: ...
    def remove_by_sources(self, sources: set[str]) -> int: ...
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...
    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[float, dict[str, Any]]]: ...
    def save(self, storage_dir: Any) -> None: ...
    def load(self, storage_dir: Any) -> bool: ...
    def supports_hybrid(self) -> bool: ...

    @staticmethod
    def exists(storage_dir: Any) -> bool: ...


def create_store() -> VectorStoreBackend:
    backend = config.VECTOR_BACKEND.lower()
    if backend == "qdrant":
        from .backends.qdrant_store import QdrantVectorStore

        return QdrantVectorStore()
    if backend == "milvus":
        from .backends.milvus_store import MilvusVectorStore

        return MilvusVectorStore()
    from .vector_store import VectorStore

    return VectorStore()


def store_exists() -> bool:
    backend = config.VECTOR_BACKEND.lower()
    if backend == "qdrant":
        from .backends.qdrant_store import QdrantVectorStore

        return QdrantVectorStore.exists(config.STORAGE_DIR)
    if backend == "milvus":
        from .backends.milvus_store import MilvusVectorStore

        return MilvusVectorStore.exists(config.STORAGE_DIR)
    from .vector_store import VectorStore

    return VectorStore.exists(config.STORAGE_DIR)
