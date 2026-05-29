"""RAG 评估：Hit@K 与关键词命中率。"""

import json
from pathlib import Path
from typing import Any

from . import config
from .logger import get_logger
from .pipeline import has_index, query

log = get_logger()

DEFAULT_EVAL = config._PROJECT_ROOT / "data" / "eval_qa.json"


def load_eval_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("cases", [])


def hit_at_k(sources: list[dict], expected_source: str, k: int) -> bool:
    top = sources[:k]
    return any(s.get("source") == expected_source for s in top)


def keyword_hit(sources: list[dict], keywords: list[str], k: int) -> bool:
    if not keywords:
        return False
    for s in sources[:k]:
        text = (s.get("excerpt") or "").lower()
        if any(kw.lower() in text for kw in keywords):
            return True
    return False


def run_eval(cases: list[dict[str, Any]], top_k: int | None = None) -> dict[str, Any]:
    k = top_k or config.TOP_K
    if not has_index():
        raise RuntimeError("索引未建立，请先运行 python ingest.py")

    source_hits = 0
    kw_hits = 0
    details: list[dict[str, Any]] = []

    for case in cases:
        q = case["question"]
        resp = query(q, top_k=k)
        sources = resp.get("sources", [])
        expected = case.get("expected_source", "")
        keywords = case.get("keywords", [])

        src_ok = hit_at_k(sources, expected, k) if expected else False
        kw_ok = keyword_hit(sources, keywords, k) if keywords else src_ok

        if src_ok:
            source_hits += 1
        if kw_ok:
            kw_hits += 1

        details.append(
            {
                "question": q,
                "source_hit": src_ok,
                "keyword_hit": kw_ok,
                "top_source": sources[0]["source"] if sources else None,
                "top_score": sources[0]["score"] if sources else 0,
            }
        )
        log.info("Q: %s | source_hit=%s kw_hit=%s", q[:50], src_ok, kw_ok)

    n = len(cases) or 1
    return {
        "total": len(cases),
        "hit_at_k_source": source_hits / n,
        "hit_at_k_keyword": kw_hits / n,
        "top_k": k,
        "details": details,
    }
