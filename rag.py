"""RAG 流水线：建索引与问答。"""

import shutil
import time
from collections.abc import Iterator
from typing import Any

import ollama

import config
from chunker import split_documents
from embedder import embed_texts
from index_manifest import (
    check_index_version,
    diff_files,
    load_manifest,
    save_manifest,
    scan_data_dir,
)
from loader import load_documents
from logger import get_logger
from prompts import build_rag_prompt
from retrieval import (
    filter_by_threshold,
    hybrid_search,
    merge_similar_chunks,
    rerank_by_keywords,
)
from vector_store import VectorStore

log = get_logger()
_store: VectorStore | None = None


def _get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
        _store.load(config.STORAGE_DIR)
    return _store


def reset_store() -> None:
    global _store
    _store = VectorStore()


def _hits_to_sources(hits: list[tuple[float, dict[str, Any]]]) -> list[dict[str, Any]]:
    sources = []
    for score, meta in hits:
        excerpt = meta.get("text", "")
        if len(excerpt) > 300:
            excerpt = excerpt[:300] + "..."
        sources.append(
            {
                "source": meta.get("source", ""),
                "score": score,
                "excerpt": excerpt,
                "chunk_index": meta.get("chunk_index", 0),
            }
        )
    return sources


def _retrieve(question: str, store: VectorStore, top_k: int) -> list[tuple[float, dict[str, Any]]]:
    q_vec = embed_texts([question])[0]
    if config.HYBRID_SEARCH and store.size > 0:
        hits = hybrid_search(
            store, question, q_vec, top_k=config.RETRIEVE_N, retrieve_n=config.RETRIEVE_N
        )
    else:
        hits = store.search(q_vec, top_k=config.RETRIEVE_N)

    hits = rerank_by_keywords(question, hits)
    hits = merge_similar_chunks(hits)
    hits = filter_by_threshold(hits, config.SIMILARITY_THRESHOLD)
    return hits[:top_k]


def _expand_query(question: str, history: list[dict[str, str]] | None) -> str:
    parts = [question]
    if history:
        recent = [m["content"] for m in history if m.get("role") == "user"][-config.CHAT_HISTORY_TURNS :]
        if recent:
            parts.insert(0, " ".join(recent))
    return " ".join(parts)


def ingest(rebuild: bool = False, incremental: bool = True) -> dict[str, Any]:
    """加载文档、分块、嵌入并持久化；支持增量更新。"""
    global _store
    t0 = time.perf_counter()
    current_files = scan_data_dir()

    if rebuild:
        if config.STORAGE_DIR.exists():
            shutil.rmtree(config.STORAGE_DIR)
        config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        reset_store()
        return _ingest_all(current_files, "rebuild", t0)

    if incremental and VectorStore.exists(config.STORAGE_DIR):
        manifest = load_manifest()
        added, modified, deleted = diff_files(manifest.get("files", {}), current_files)
        if not added and not modified and not deleted:
            log.info("无文件变更，跳过建索引")
            store = _get_store()
            return {
                "doc_count": len(current_files),
                "chunk_count": store.size,
                "dim": int(store.embeddings.shape[1]) if store.size else 0,
                "mode": "skip",
            }

        store = VectorStore()
        store.load(config.STORAGE_DIR)
        to_remove = set(deleted) | set(modified)
        removed = store.remove_by_sources(to_remove)
        log.info("增量索引: +%d ~%d -%d (移除 %d 块)", len(added), len(modified), len(deleted), removed)

        to_index = set(added) | set(modified)
        if to_index:
            docs = load_documents(config.DATA_DIR, sources=to_index)
            chunks = split_documents(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            if chunks:
                vectors = embed_texts([c["text"] for c in chunks], batch_size=config.EMBED_BATCH_SIZE)
                store.add(chunks, vectors)

        store.save(config.STORAGE_DIR)
        dim = int(store.embeddings.shape[1]) if store.size else 0
        save_manifest(current_files, config.EMBED_MODEL, dim)
        _store = store

        elapsed = time.perf_counter() - t0
        log.info("增量建索引完成: %d 块, %.2fs", store.size, elapsed)
        return {
            "doc_count": len(current_files),
            "chunk_count": store.size,
            "dim": dim,
            "mode": "incremental",
            "elapsed": elapsed,
        }

    reset_store()
    return _ingest_all(current_files, "full", t0)


def _ingest_all(current_files: dict, mode: str, t0: float) -> dict[str, Any]:
    global _store
    docs = load_documents(config.DATA_DIR)
    if not docs:
        save_manifest(current_files, config.EMBED_MODEL, 0)
        return {"doc_count": 0, "chunk_count": 0, "dim": 0, "mode": mode}

    chunks = split_documents(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    if not chunks:
        save_manifest(current_files, config.EMBED_MODEL, 0)
        return {"doc_count": len(docs), "chunk_count": 0, "dim": 0, "mode": mode}

    vectors = embed_texts([c["text"] for c in chunks], batch_size=config.EMBED_BATCH_SIZE)
    store = VectorStore()
    store.add(chunks, vectors)
    store.save(config.STORAGE_DIR)
    dim = int(vectors.shape[1]) if vectors.size else 0
    save_manifest(current_files, config.EMBED_MODEL, dim)
    _store = store

    elapsed = time.perf_counter() - t0
    log.info("建索引完成 [%s] %d 文档, %d 块, %.2fs", mode, len(docs), store.size, elapsed)
    return {
        "doc_count": len(docs),
        "chunk_count": store.size,
        "dim": dim,
        "mode": mode,
        "elapsed": elapsed,
    }


def query(
    question: str,
    top_k: int | None = None,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    k = top_k if top_k is not None else config.TOP_K
    store = _get_store()

    if store.size == 0:
        return {
            "answer": "知识库尚未建立索引。请先运行 `python ingest.py` 或在界面中点击「重建索引」。",
            "sources": [],
        }

    ok, version_msg = check_index_version()
    if not ok:
        return {"answer": version_msg, "sources": []}

    t0 = time.perf_counter()
    search_q = _expand_query(question, history)
    hits = _retrieve(search_q, store, k)
    log.info("检索完成: %d 条命中, %.2fs", len(hits), time.perf_counter() - t0)

    prompt = build_rag_prompt(question, hits)
    sources = _hits_to_sources(hits)

    try:
        client = ollama.Client(host=config.OLLAMA_HOST)
        response = client.chat(
            model=config.CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response["message"]["content"]
    except Exception as e:
        raise RuntimeError(
            f"生成失败。请确认 Ollama 已启动，且已执行: ollama pull {config.CHAT_MODEL}\n原始错误: {e}"
        ) from e

    log.info("生成完成, 总耗时 %.2fs", time.perf_counter() - t0)
    return {"answer": answer, "sources": sources}


def _stream_answer(prompt: str) -> Iterator[str]:
    client = ollama.Client(host=config.OLLAMA_HOST)
    stream = client.chat(
        model=config.CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        if content:
            yield content


def query_with_stream(
    question: str,
    top_k: int | None = None,
    history: list[dict[str, str]] | None = None,
) -> tuple[Iterator[str], list[dict[str, Any]]]:
    """返回 (token 迭代器, sources)。"""
    k = top_k if top_k is not None else config.TOP_K
    store = _get_store()

    if store.size == 0:
        def _msg() -> Iterator[str]:
            yield "知识库尚未建立索引。"

        return _msg(), []

    ok, version_msg = check_index_version()
    if not ok:

        def _verr() -> Iterator[str]:
            yield version_msg

        return _verr(), []

    search_q = _expand_query(question, history)
    hits = _retrieve(search_q, store, k)
    prompt = build_rag_prompt(question, hits)
    return _stream_answer(prompt), _hits_to_sources(hits)


def has_index() -> bool:
    return VectorStore.exists(config.STORAGE_DIR)
