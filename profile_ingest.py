"""建索引性能探测：比较不同 EMBED_BATCH_SIZE。"""

import argparse
import time

import config
from embedder import embed_texts
from loader import load_documents
from chunker import split_documents
from logger import get_logger

log = get_logger()


def profile_batch_sizes(sizes: list[int]) -> None:
    docs = load_documents(config.DATA_DIR)
    chunks = split_documents(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    if not chunks:
        print("无文档块可测试")
        return

    texts = [c["text"] for c in chunks]
    print(f"文档 {len(docs)} 篇, 块 {len(texts)} 个\n")

    for size in sizes:
        t0 = time.perf_counter()
        embed_texts(texts[: min(len(texts), 20)], batch_size=size)
        elapsed = time.perf_counter() - t0
        n = min(len(texts), 20)
        print(f"batch_size={size:3d}  sample={n} chunks  time={elapsed:.2f}s  per_chunk={elapsed/n:.3f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile embedding batch sizes")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16, 32],
        help="Batch sizes to test",
    )
    args = parser.parse_args()
    profile_batch_sizes(args.sizes)


if __name__ == "__main__":
    main()
