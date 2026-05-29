"""CLI：构建向量索引。"""

import argparse
import time

from rag import config
from rag.logger import get_logger
from rag.ollama_health import get_health_status
from rag.pipeline import ingest

log = get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="从 data/ 目录构建 RAG 向量索引")
    parser.add_argument("--rebuild", action="store_true", help="清空 storage/ 后重新建索引")
    parser.add_argument("--no-incremental", action="store_true", help="禁用增量，始终全量处理")
    args = parser.parse_args()

    health = get_health_status()
    if not health["ok"]:
        for msg in health["messages"]:
            log.error(msg)
        raise SystemExit(1)

    start = time.perf_counter()
    stats = ingest(rebuild=args.rebuild, incremental=not args.no_incremental)
    elapsed = time.perf_counter() - start

    print("--- 建索引完成 ---")
    print(f"模式:   {stats.get('mode', 'unknown')}")
    print(f"文档数: {stats['doc_count']}")
    print(f"块数:   {stats['chunk_count']}")
    print(f"向量维度: {stats['dim']}")
    print(f"耗时:   {elapsed:.2f}s")
    print(f"索引目录: {config.STORAGE_DIR}")


if __name__ == "__main__":
    main()
