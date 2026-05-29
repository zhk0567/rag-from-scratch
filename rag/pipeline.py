"""RAG 流水线：建索引与问答。"""

import shutil
import time
from collections.abc import Iterator
from typing import Any

import ollama

from . import config
from .chunker import split_documents
from .embedder import embed_texts
from .graph_rag import boost_hits, build_graph_from_chunks, load_graph, save_graph
from .index_manifest import (
    check_index_version,
    diff_files,
    load_manifest,
    save_manifest,
    scan_data_dir,
)
from .loader import load_documents
from .logger import get_logger
from .prompts import build_rag_prompt
from .query_rewrite import build_search_query, rewrite_query
from .retrieval import (
    filter_by_threshold,
    hybrid_search,
    merge_similar_chunks,
    rerank_by_keywords,
)
from .store_factory import VectorStoreBackend, create_store, store_exists

log = get_logger()
_store: VectorStoreBackend | None = None


def _get_store() -> VectorStoreBackend:
    global _store
    if _store is None:
        _store = create_store()
        _store.load(config.STORAGE_DIR)
    return _store


def reset_store() -> None:
    global _store
    _store = create_store()
    _store.clear()


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


def _raw_retrieve(
    search_q: str,
    store: VectorStoreBackend,
    top_k: int,
) -> list[tuple[float, dict[str, Any]]]:
    q_vec = embed_texts([search_q])[0]
    if config.HYBRID_SEARCH and store.size > 0:
        hits = hybrid_search(
            store, search_q, q_vec, top_k=config.RETRIEVE_N, retrieve_n=config.RETRIEVE_N
        )
    else:
        hits = store.search(q_vec, top_k=config.RETRIEVE_N)

    hits = rerank_by_keywords(search_q, hits)
    hits = merge_similar_chunks(hits)

    if config.USE_GRAPH_RAG:
        graph = load_graph()
        hits = boost_hits(hits, search_q, graph)

    hits = filter_by_threshold(hits, config.SIMILARITY_THRESHOLD)

    if config.USE_CLIP:
        try:
            from .clip_index import merge_with_text_hits, search_clip

            clip_hits = search_clip(search_q, config.RETRIEVE_N)
            if clip_hits:
                hits = merge_with_text_hits(hits, clip_hits, config.RETRIEVE_N)
        except Exception as e:
            log.warning("CLIP 检索跳过: %s", e)

    return hits[:top_k]


def _retrieve(
    question: str,
    store: VectorStoreBackend,
    top_k: int,
    history: list[dict[str, str]] | None = None,
) -> list[tuple[float, dict[str, Any]]]:
    search_q = build_search_query(question, history)
    hits = _raw_retrieve(search_q, store, top_k)

    if config.USE_AGENTIC_RAG:
        max_score = hits[0][0] if hits else 0.0
        if not hits or max_score < config.AGENTIC_MIN_SCORE:
            log.info("Agentic 重检索 (max_score=%.3f)", max_score)
            alt_q = rewrite_query(question)
            if alt_q != search_q:
                alt_hits = _raw_retrieve(alt_q, store, top_k)
                if alt_hits and (not hits or alt_hits[0][0] > max_score):
                    hits = alt_hits

    return hits


def _index_chunks(chunks: list, store: VectorStoreBackend | None = None) -> VectorStoreBackend:
    t_embed = time.perf_counter()
    vectors = embed_texts([c["text"] for c in chunks], batch_size=config.EMBED_BATCH_SIZE)
    embed_sec = time.perf_counter() - t_embed

    if store is None:
        store = create_store()
    store.add(chunks, vectors)

    graph = build_graph_from_chunks(chunks)
    save_graph(graph)

    log.info("嵌入 %d 块耗时 %.2fs", len(chunks), embed_sec)
    return store


def _maybe_build_clip() -> None:
    if not config.USE_CLIP:
        return
    try:
        from .clip_index import build_clip_index

        build_clip_index()
    except Exception as e:
        log.warning("CLIP 索引跳过: %s", e)


def ingest(rebuild: bool = False, incremental: bool = True) -> dict[str, Any]:
    global _store
    t0 = time.perf_counter()
    current_files = scan_data_dir()

    if rebuild:
        if config.STORAGE_DIR.exists():
            shutil.rmtree(config.STORAGE_DIR)
        config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        reset_store()
        if config.USE_VISION_CACHE:
            from .vision_cache import clear_cache

            clear_cache()
        return _ingest_all(current_files, "rebuild", t0)

    if incremental and store_exists():
        manifest = load_manifest()
        added, modified, deleted = diff_files(manifest.get("files", {}), current_files)
        if not added and not modified and not deleted:
            log.info("无文件变更，跳过建索引")
            store = _get_store()
            return {
                "doc_count": len(current_files),
                "chunk_count": store.size,
                "dim": store.embedding_dim,
                "mode": "skip",
            }

        store = create_store()
        store.load(config.STORAGE_DIR)
        to_remove = set(deleted) | set(modified)
        removed = store.remove_by_sources(to_remove)
        log.info("增量索引: +%d ~%d -%d (移除 %d 块)", len(added), len(modified), len(deleted), removed)

        to_index = set(added) | set(modified)
        if to_index:
            docs = load_documents(config.DATA_DIR, sources=to_index)
            chunks = split_documents(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            if chunks:
                store = _index_chunks(chunks, store)

        store.save(config.STORAGE_DIR)
        dim = store.embedding_dim
        save_manifest(current_files, config.EMBED_MODEL, dim)
        _store = store

        elapsed = time.perf_counter() - t0
        _maybe_build_clip()
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

    store = _index_chunks(chunks)
    store.save(config.STORAGE_DIR)
    dim = store.embedding_dim
    save_manifest(current_files, config.EMBED_MODEL, dim)
    _store = store

    elapsed = time.perf_counter() - t0
    log.info("建索引完成 [%s] %d 文档, %d 块, %.2fs", mode, len(docs), store.size, elapsed)
    _maybe_build_clip()
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
    hits = _retrieve(question, store, k, history)
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

    hits = _retrieve(question, store, k, history)
    prompt = build_rag_prompt(question, hits)
    return _stream_answer(prompt), _hits_to_sources(hits)


def has_index() -> bool:
    return store_exists()
