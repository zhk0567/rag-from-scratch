"""RAG 评估：Hit@K 与关键词命中率。"""

import argparse
import json
from pathlib import Path
from typing import Any

import config
import rag
from logger import get_logger

log = get_logger()

DEFAULT_EVAL = Path(__file__).parent / "data" / "eval_qa.json"


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
    if not rag.has_index():
        raise RuntimeError("索引未建立，请先运行 python ingest.py")

    results: list[dict[str, Any]] = []
    source_hits = 0
    kw_hits = 0

    for case in cases:
        q = case["question"]
        resp = rag.query(q, top_k=k)
        sources = resp.get("sources", [])
        expected = case.get("expected_source", "")
        keywords = case.get("keywords", [])

        src_ok = hit_at_k(sources, expected, k) if expected else False
        kw_ok = keyword_hit(sources, keywords, k) if keywords else src_ok

        if src_ok:
            source_hits += 1
        if kw_ok:
            kw_hits += 1

        results.append(
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
        "details": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality")
    parser.add_argument("--file", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    cases = load_eval_cases(args.file)
    summary = run_eval(cases, top_k=args.top_k)

    print("--- Evaluation ---")
    print(f"Cases: {summary['total']}")
    print(f"Hit@K (expected source): {summary['hit_at_k_source']:.1%}")
    print(f"Hit@K (keywords):        {summary['hit_at_k_keyword']:.1%}")

    if args.json_out:
        args.json_out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
