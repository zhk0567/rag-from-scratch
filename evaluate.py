"""CLI：RAG 检索质量评估。"""

import argparse
import json
from pathlib import Path

from rag.evaluate import DEFAULT_EVAL, load_eval_cases, run_eval


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
