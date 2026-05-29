"""本地向量存储：余弦相似度检索 + NumPy/JSON 持久化。"""

import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .chunker import Chunk


class VectorStore:
    def __init__(self) -> None:
        self.ids: list[str] = []
        self.embeddings: np.ndarray = np.array([], dtype=np.float32).reshape(0, 0)
        self.metadatas: list[dict[str, Any]] = []

    @property
    def size(self) -> int:
        return len(self.ids)

    @property
    def embedding_dim(self) -> int:
        if self.embeddings.size == 0:
            return 0
        return int(self.embeddings.shape[1])

    def supports_hybrid(self) -> bool:
        return True

    def clear(self) -> None:
        self.ids = []
        self.embeddings = np.array([], dtype=np.float32).reshape(0, 0)
        self.metadatas = []

    def remove_by_sources(self, sources: set[str]) -> int:
        """删除指定来源文件的所有块，返回删除数量。"""
        if not sources or self.size == 0:
            return 0

        keep_indices = [
            i for i, m in enumerate(self.metadatas) if m.get("source") not in sources
        ]
        removed = self.size - len(keep_indices)
        if removed == 0:
            return 0

        if keep_indices:
            self.ids = [self.ids[i] for i in keep_indices]
            self.metadatas = [self.metadatas[i] for i in keep_indices]
            self.embeddings = self.embeddings[keep_indices]
        else:
            self.clear()
        return removed

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks 与 vectors 数量不一致")

        for chunk, vec in zip(chunks, vectors):
            self.ids.append(str(uuid.uuid4()))
            self.metadatas.append(
                {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "doc_id": chunk["doc_id"],
                    "chunk_index": chunk["chunk_index"],
                }
            )

        if self.embeddings.size == 0:
            self.embeddings = vectors.copy()
        else:
            self.embeddings = np.vstack([self.embeddings, vectors])

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[tuple[float, dict[str, Any]]]:
        if self.size == 0:
            return []

        q = query_vector.astype(np.float32).reshape(1, -1)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            q_norm = 1e-10

        similarities = (self.embeddings / norms) @ (q / q_norm).T
        similarities = similarities.flatten()

        k = min(top_k, self.size)
        top_indices = np.argsort(similarities)[::-1][:k]

        results: list[tuple[float, dict[str, Any]]] = []
        for idx in top_indices:
            score = float(similarities[idx])
            meta = dict(self.metadatas[idx])
            results.append((score, meta))
        return results

    def save(self, storage_dir: Path) -> None:
        storage_dir.mkdir(parents=True, exist_ok=True)
        np.save(storage_dir / "embeddings.npy", self.embeddings)
        index_data = {
            "ids": self.ids,
            "metadatas": self.metadatas,
            "dim": int(self.embeddings.shape[1]) if self.embeddings.size else 0,
            "count": self.size,
            "embed_model": None,
        }
        (storage_dir / "index.json").write_text(
            json.dumps(index_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, storage_dir: Path) -> bool:
        emb_path = storage_dir / "embeddings.npy"
        idx_path = storage_dir / "index.json"
        if not emb_path.exists() or not idx_path.exists():
            return False

        self.embeddings = np.load(emb_path)
        index_data = json.loads(idx_path.read_text(encoding="utf-8"))
        self.ids = index_data["ids"]
        self.metadatas = index_data["metadatas"]
        return True

    @staticmethod
    def exists(storage_dir: Path) -> bool:
        return (storage_dir / "embeddings.npy").exists() and (storage_dir / "index.json").exists()
