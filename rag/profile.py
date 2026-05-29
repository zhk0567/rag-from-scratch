"""建索引性能探测：比较不同 EMBED_BATCH_SIZE。"""

import time
from pathlib import Path

from . import config
from .chunker import split_documents
from .embedder import embed_texts
from .loader import load_documents
from .logger import get_logger

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
