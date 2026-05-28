"""vector_store 单元测试。"""

import numpy as np

from chunker import Chunk
from vector_store import VectorStore


def _chunk(text: str, source: str = "a.txt", idx: int = 0) -> Chunk:
    return Chunk(text=text, source=source, doc_id=source, chunk_index=idx)


def test_search_returns_highest_similarity_first():
    store = VectorStore()
    chunks = [_chunk("apple fruit"), _chunk("car vehicle")]
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    store.add(chunks, vecs)
    hits = store.search(np.array([0.9, 0.1], dtype=np.float32), top_k=1)
    assert len(hits) == 1
    assert "apple" in hits[0][1]["text"]


def test_remove_by_sources():
    store = VectorStore()
    chunks = [_chunk("a", "x.txt", 0), _chunk("b", "y.txt", 0)]
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    store.add(chunks, vecs)
    n = store.remove_by_sources({"x.txt"})
    assert n == 1
    assert store.size == 1
    assert store.metadatas[0]["source"] == "y.txt"
