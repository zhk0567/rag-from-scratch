"""轻量 Graph 增强：从分块中提取标题/章节关键词，检索时加权。"""

import json
import re
from pathlib import Path
from typing import Any

from . import config

_HEADER_RE = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)


def build_graph_from_chunks(chunks: list[dict[str, Any]]) -> dict[str, list[str]]:
    """source -> 章节标题/关键词列表。"""
    graph: dict[str, list[str]] = {}
    for c in chunks:
        source = c.get("source", "")
        text = c.get("text", "")
        titles = _HEADER_RE.findall(text)
        if not titles and source.endswith(".md"):
            first_line = text.split("\n", 1)[0].strip()
            if first_line:
                titles = [first_line[:80]]
        if titles:
            graph.setdefault(source, []).extend(titles)
    for src in graph:
        graph[src] = list(dict.fromkeys(graph[src]))
    return graph


def save_graph(graph: dict[str, list[str]], storage_dir: Path | None = None) -> None:
    storage_dir = storage_dir or config.STORAGE_DIR
    storage_dir.mkdir(parents=True, exist_ok=True)
    (storage_dir / "graph.json").write_text(
        json.dumps(graph, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_graph(storage_dir: Path | None = None) -> dict[str, list[str]]:
    storage_dir = storage_dir or config.STORAGE_DIR
    path = storage_dir / "graph.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def boost_hits(
    hits: list[tuple[float, dict[str, Any]]],
    question: str,
    graph: dict[str, list[str]],
    boost: float = 0.08,
) -> list[tuple[float, dict[str, Any]]]:
    """若问题命中某文档的章节标题，提升该块分数。"""
    if not graph or not hits:
        return hits

    q_lower = question.lower()
    boosted: list[tuple[float, dict[str, Any]]] = []
    for score, meta in hits:
        source = meta.get("source", "")
        extra = 0.0
        for title in graph.get(source, []):
            if title.lower() in q_lower or any(
                w in q_lower for w in title.lower().split() if len(w) > 1
            ):
                extra = boost
                break
        boosted.append((score + extra, meta))
    boosted.sort(key=lambda x: x[0], reverse=True)
    return boosted
