"""CLI：构建向量索引。"""

import argparse
import shutil
import time

import config
from rag import ingest


def main() -> None:
    parser = argparse.ArgumentParser(description="从 data/ 目录构建 RAG 向量索引")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="清空 storage/ 后重新建索引",
    )
    args = parser.parse_args()

    if args.rebuild and config.STORAGE_DIR.exists():
        shutil.rmtree(config.STORAGE_DIR)
        config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    stats = ingest(rebuild=args.rebuild)
    elapsed = time.perf_counter() - start

    print("--- 建索引完成 ---")
    print(f"文档数: {stats['doc_count']}")
    print(f"块数:   {stats['chunk_count']}")
    print(f"向量维度: {stats['dim']}")
    print(f"耗时:   {elapsed:.2f}s")
    print(f"索引目录: {config.STORAGE_DIR}")


if __name__ == "__main__":
    main()
