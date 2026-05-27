"""RAG 流水线：建索引与问答。"""

import shutil
from typing import Any

import ollama

import config
from chunker import split_documents
from embedder import embed_texts
from loader import load_documents
from vector_store import VectorStore

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


def ingest(rebuild: bool = False) -> dict[str, Any]:
    """加载文档、分块、嵌入并持久化到 storage/。"""
    if rebuild and config.STORAGE_DIR.exists():
        shutil.rmtree(config.STORAGE_DIR)
        config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    docs = load_documents(config.DATA_DIR)
    if not docs:
        reset_store()
        return {"doc_count": 0, "chunk_count": 0, "dim": 0}

    chunks = split_documents(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    if not chunks:
        reset_store()
        return {"doc_count": len(docs), "chunk_count": 0, "dim": 0}

    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)

    store = VectorStore()
    store.add(chunks, vectors)
    store.save(config.STORAGE_DIR)

    global _store
    _store = store

    dim = int(vectors.shape[1]) if vectors.size else 0
    return {"doc_count": len(docs), "chunk_count": len(chunks), "dim": dim}


def _build_prompt(question: str, hits: list[tuple[float, dict[str, Any]]]) -> str:
    ref_lines: list[str] = []
    for i, (score, meta) in enumerate(hits, start=1):
        source = meta.get("source", "unknown")
        text = meta.get("text", "")
        ref_lines.append(f"[{i}] (来源: {source}, 相似度: {score:.3f})\n{text}")

    references = "\n\n".join(ref_lines) if ref_lines else "（无相关资料）"

    return f"""你是基于检索内容的问答助手。仅根据下列「参考资料」回答；资料不足则明确说不知道，不要编造。

【参考资料】
{references}

【问题】
{question}"""


def query(question: str, top_k: int | None = None) -> dict[str, Any]:
    """
    检索 + 生成，返回 answer 与 sources 列表。
    """
    k = top_k if top_k is not None else config.TOP_K
    store = _get_store()

    if store.size == 0:
        return {
            "answer": "知识库尚未建立索引。请先运行 `python ingest.py` 或在界面中点击「重建索引」。",
            "sources": [],
        }

    q_vec = embed_texts([question])[0]
    hits = store.search(q_vec, top_k=k)

    prompt = _build_prompt(question, hits)

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

    return {"answer": answer, "sources": sources}


def has_index() -> bool:
    return VectorStore.exists(config.STORAGE_DIR)
