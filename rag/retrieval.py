"""混合检索（向量 + BM25）与轻量重排序。"""

import math
import re
from collections import Counter
from typing import Any

import numpy as np

from .store_factory import VectorStoreBackend


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())


class SimpleBM25:
    """轻量 BM25，无额外依赖。"""

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs = [_tokenize(d) for d in corpus]
        self.n_docs = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(self.n_docs, 1)
        self.doc_freqs: Counter[str] = Counter()
        for doc in self.docs:
            for term in set(doc):
                self.doc_freqs[term] += 1

    def score(self, query: str, doc_idx: int) -> float:
        q_terms = _tokenize(query)
        doc = self.docs[doc_idx]
        dl = len(doc)
        tf_counter = Counter(doc)
        score = 0.0
        for term in q_terms:
            if term not in tf_counter:
                continue
            df = self.doc_freqs.get(term, 0)
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
            tf = tf_counter[term]
            denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
            score += idf * (tf * (self.k1 + 1)) / denom
        return score

    def score_all(self, query: str) -> list[float]:
        return [self.score(query, i) for i in range(self.n_docs)]


def rrf_merge(
    vector_hits: list[tuple[float, dict[str, Any]]],
    bm25_hits: list[tuple[float, dict[str, Any]]],
    k: int = 60,
) -> list[tuple[float, dict[str, Any]]]:
    """Reciprocal Rank Fusion 融合两路检索结果。"""
    scores: dict[str, float] = {}
    meta_map: dict[str, dict[str, Any]] = {}

    for rank, (_, meta) in enumerate(vector_hits):
        key = f"{meta['source']}::{meta.get('chunk_index', 0)}"
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        meta_map[key] = meta

    for rank, (_, meta) in enumerate(bm25_hits):
        key = f"{meta['source']}::{meta.get('chunk_index', 0)}"
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        meta_map[key] = meta

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, meta_map[key]) for key, score in ranked]


def hybrid_search(
    store: VectorStoreBackend,
    query: str,
    query_vector: np.ndarray,
    top_k: int,
    retrieve_n: int | None = None,
) -> list[tuple[float, dict[str, Any]]]:
    n = retrieve_n or max(top_k * 3, top_k)
    vector_hits = store.search(query_vector, top_k=n)

    if store.size == 0 or not store.supports_hybrid():
        return vector_hits[:top_k]

    corpus = [m["text"] for m in store.metadatas]  # type: ignore[attr-defined]
    bm25 = SimpleBM25(corpus)
    bm25_scores = bm25.score_all(query)
    bm25_indices = np.argsort(bm25_scores)[::-1][:n]
    bm25_hits = [
        (float(bm25_scores[i]), dict(store.metadatas[i])) for i in bm25_indices if bm25_scores[i] > 0
    ]

    merged = rrf_merge(vector_hits, bm25_hits)
    return merged[:top_k]


def rerank_by_keywords(
    query: str,
    hits: list[tuple[float, dict[str, Any]]],
) -> list[tuple[float, dict[str, Any]]]:
    """轻量重排：在 RRF/向量分数上叠加查询词命中比例。"""
    q_terms = set(_tokenize(query))
    if not q_terms:
        return hits

    rescored: list[tuple[float, dict[str, Any]]] = []
    for score, meta in hits:
        text_terms = set(_tokenize(meta.get("text", "")))
        overlap = len(q_terms & text_terms) / len(q_terms)
        rescored.append((score + 0.1 * overlap, meta))

    rescored.sort(key=lambda x: x[0], reverse=True)
    return rescored


def merge_similar_chunks(
    hits: list[tuple[float, dict[str, Any]]],
    threshold: float = 0.95,
) -> list[tuple[float, dict[str, Any]]]:
    """合并同一文档相邻且文本高度重叠的块。"""
    if len(hits) <= 1:
        return hits

    merged: list[tuple[float, dict[str, Any]]] = []
    for score, meta in hits:
        if not merged:
            merged.append((score, dict(meta)))
            continue
        prev_score, prev = merged[-1]
        same_source = prev.get("source") == meta.get("source")
        prev_text = prev.get("text", "")
        cur_text = meta.get("text", "")
        if same_source and prev_text and cur_text:
            shorter, longer = (
                (prev_text, cur_text) if len(prev_text) < len(cur_text) else (cur_text, prev_text)
            )
            if shorter in longer or _jaccard(prev_text, cur_text) >= threshold:
                if len(cur_text) > len(prev_text):
                    prev["text"] = cur_text
                merged[-1] = (max(prev_score, score), prev)
                continue
        merged.append((score, dict(meta)))
    return merged


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(_tokenize(a)), set(_tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def filter_by_threshold(
    hits: list[tuple[float, dict[str, Any]]],
    threshold: float,
) -> list[tuple[float, dict[str, Any]]]:
    return [(s, m) for s, m in hits if s >= threshold]
