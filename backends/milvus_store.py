"""Milvus 向量库后端。"""

import uuid
from pathlib import Path
from typing import Any

import numpy as np

import config
from chunker import Chunk
from logger import get_logger

log = get_logger()


def _collection_name() -> str:
    base = config.MILVUS_COLLECTION.replace("-", "_")
    if config.TENANT_ID:
        return f"{base}_{config.TENANT_ID}"
    return base


class MilvusVectorStore:
    def __init__(self) -> None:
        try:
            from pymilvus import (
                Collection,
                CollectionSchema,
                DataType,
                FieldSchema,
                connections,
                utility,
            )
        except ImportError as e:
            raise ImportError("Milvus 需要: pip install -r requirements-milvus.txt") from e

        self._utility = utility
        self._Collection = Collection
        self._FieldSchema = FieldSchema
        self._CollectionSchema = CollectionSchema
        self._DataType = DataType

        connections.connect(alias="default", uri=config.MILVUS_URI)
        self.collection_name = _collection_name()
        self._collection: Collection | None = None
        self._dim = 0

    @property
    def size(self) -> int:
        if not self._collection:
            return 0
        return int(self._collection.num_entities)

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def supports_hybrid(self) -> bool:
        return False

    def _get_collection(self, dim: int) -> "Collection":
        if self._utility.has_collection(self.collection_name):
            col = self._Collection(self.collection_name)
            col.load()
            self._collection = col
            return col

        fields = [
            self._FieldSchema(name="id", dtype=self._DataType.VARCHAR, is_primary=True, max_length=64),
            self._FieldSchema(name="vector", dtype=self._DataType.FLOAT_VECTOR, dim=dim),
            self._FieldSchema(name="text", dtype=self._DataType.VARCHAR, max_length=65535),
            self._FieldSchema(name="source", dtype=self._DataType.VARCHAR, max_length=512),
            self._FieldSchema(name="doc_id", dtype=self._DataType.VARCHAR, max_length=512),
            self._FieldSchema(name="chunk_index", dtype=self._DataType.INT64),
        ]
        schema = self._CollectionSchema(fields=fields)
        col = self._Collection(self.collection_name, schema)
        col.create_index(
            field_name="vector",
            index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE"},
        )
        col.load()
        self._collection = col
        self._dim = dim
        log.info("已创建 Milvus 集合: %s (dim=%d)", self.collection_name, dim)
        return col

    def clear(self) -> None:
        if self._utility.has_collection(self.collection_name):
            self._utility.drop_collection(self.collection_name)
        self._collection = None
        self._dim = 0

    def remove_by_sources(self, sources: set[str]) -> int:
        if not sources or not self._collection:
            return 0
        expr = "source in [" + ",".join(f'"{s}"' for s in sources) + "]"
        self._collection.delete(expr)
        return 0

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks 与 vectors 数量不一致")
        dim = int(vectors.shape[1])
        col = self._get_collection(dim)
        ids, vecs, texts, srcs, doc_ids, idxs = [], [], [], [], [], []
        for chunk, vec in zip(chunks, vectors):
            ids.append(str(uuid.uuid4()))
            vecs.append(vec.tolist())
            texts.append(chunk["text"][:65000])
            srcs.append(chunk["source"])
            doc_ids.append(chunk["doc_id"])
            idxs.append(chunk["chunk_index"])
        col.insert([ids, vecs, texts, srcs, doc_ids, idxs])
        col.flush()

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[tuple[float, dict[str, Any]]]:
        if not self._collection or self.size == 0:
            return []
        results = self._collection.search(
            data=[query_vector.astype(np.float32).tolist()],
            anns_field="vector",
            param={"metric_type": "COSINE"},
            limit=min(top_k, self.size),
            output_fields=["text", "source", "doc_id", "chunk_index"],
        )
        hits: list[tuple[float, dict[str, Any]]] = []
        for hit in results[0]:
            entity = hit.entity
            meta = {
                "text": entity.get("text"),
                "source": entity.get("source"),
                "doc_id": entity.get("doc_id"),
                "chunk_index": entity.get("chunk_index"),
            }
            hits.append((float(hit.distance), meta))
        return hits

    def save(self, storage_dir: Path) -> None:
        log.debug("Milvus 数据已实时持久化，跳过本地 save")

    def load(self, storage_dir: Path) -> bool:
        if not self._utility.has_collection(self.collection_name):
            return False
        col = self._Collection(self.collection_name)
        col.load()
        self._collection = col
        return self.size > 0

    @staticmethod
    def exists(storage_dir: Path) -> bool:
        try:
            from pymilvus import connections, utility

            connections.connect(alias="default", uri=config.MILVUS_URI)
            if not utility.has_collection(_collection_name()):
                return False
            from pymilvus import Collection

            return int(Collection(_collection_name()).num_entities) > 0
        except Exception:
            return False
