"""Qdrant 向量库后端。"""

import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .. import config
from ..chunker import Chunk
from ..logger import get_logger

log = get_logger()


def _collection_name() -> str:
    base = config.QDRANT_COLLECTION
    if config.TENANT_ID:
        return f"{base}_{config.TENANT_ID}"
    return base


class QdrantVectorStore:
    def __init__(self) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
        except ImportError as e:
            raise ImportError("Qdrant 需要: pip install -r requirements/qdrant.txt") from e

        self._models = models
        self.client = QdrantClient(url=config.QDRANT_URL)
        self.collection = _collection_name()
        self._dim = 0

    @property
    def size(self) -> int:
        try:
            info = self.client.get_collection(self.collection)
            return int(info.points_count or 0)
        except Exception:
            return 0

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def supports_hybrid(self) -> bool:
        return False

    def _ensure_collection(self, dim: int) -> None:
        if dim <= 0:
            return
        self._dim = dim
        names = [c.name for c in self.client.get_collections().collections]
        if self.collection in names:
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=self._models.VectorParams(size=dim, distance=self._models.Distance.COSINE),
        )
        log.info("已创建 Qdrant 集合: %s (dim=%d)", self.collection, dim)

    def clear(self) -> None:
        try:
            self.client.delete_collection(self.collection)
        except Exception:
            pass
        self._dim = 0

    def remove_by_sources(self, sources: set[str]) -> int:
        if not sources:
            return 0
        before = self.size
        self.client.delete(
            collection_name=self.collection,
            points_selector=self._models.FilterSelector(
                filter=self._models.Filter(
                    must=[
                        self._models.FieldCondition(
                            key="source",
                            match=self._models.MatchAny(any=list(sources)),
                        )
                    ]
                )
            ),
        )
        return max(before - self.size, 0)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks 与 vectors 数量不一致")
        dim = int(vectors.shape[1])
        self._ensure_collection(dim)
        points = []
        for chunk, vec in zip(chunks, vectors):
            points.append(
                self._models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec.tolist(),
                    payload={
                        "text": chunk["text"],
                        "source": chunk["source"],
                        "doc_id": chunk["doc_id"],
                        "chunk_index": chunk["chunk_index"],
                    },
                )
            )
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[tuple[float, dict[str, Any]]]:
        if self.size == 0:
            return []
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector.astype(np.float32).tolist(),
            limit=min(top_k, self.size),
        )
        return [(float(h.score), dict(h.payload)) for h in hits]

    def save(self, storage_dir: Path) -> None:
        log.debug("Qdrant 数据已实时持久化，跳过本地 save")

    def load(self, storage_dir: Path) -> bool:
        try:
            info = self.client.get_collection(self.collection)
            self._dim = info.config.params.vectors.size
            return self.size > 0
        except Exception:
            return False

    @staticmethod
    def exists(storage_dir: Path) -> bool:
        try:
            from qdrant_client import QdrantClient

            client = QdrantClient(url=config.QDRANT_URL)
            names = [c.name for c in client.get_collections().collections]
            if _collection_name() not in names:
                return False
            return int(client.get_collection(_collection_name()).points_count or 0) > 0
        except Exception:
            return False
