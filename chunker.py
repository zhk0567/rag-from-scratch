"""文档分块，优先在段落边界切分。"""

from typing import TypedDict

from loader import Document


class Chunk(TypedDict):
    text: str
    source: str
    doc_id: str
    chunk_index: int


def _split_paragraphs(text: str) -> list[str]:
    parts = text.split("\n\n")
    return [p.strip() for p in parts if p.strip()]


def _merge_to_chunks(paragraphs: list[str], chunk_size: int, overlap: int) -> list[str]:
    """将段落合并为不超过 chunk_size 的块，块间保留 overlap 字符重叠。"""
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}".strip() if current else para
        else:
            if current:
                chunks.append(current)
            if len(para) <= chunk_size:
                if chunks and overlap > 0:
                    tail = chunks[-1][-overlap:]
                    current = tail + para if tail else para
                else:
                    current = para
            else:
                start = 0
                while start < len(para):
                    end = start + chunk_size
                    piece = para[start:end]
                    chunks.append(piece)
                    start = end - overlap if overlap < chunk_size else end
                current = ""

    if current:
        chunks.append(current)

    return chunks


def split_documents(
    docs: list[Document],
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[Chunk]:
    """将文档列表切分为带元数据的块。"""
    all_chunks: list[Chunk] = []

    for doc in docs:
        paragraphs = _split_paragraphs(doc["text"])
        texts = _merge_to_chunks(paragraphs, chunk_size, overlap)

        if not texts and doc["text"]:
            texts = _merge_to_chunks([doc["text"]], chunk_size, overlap)

        for idx, text in enumerate(texts):
            all_chunks.append(
                Chunk(
                    text=text,
                    source=doc["source"],
                    doc_id=doc["doc_id"],
                    chunk_index=idx,
                )
            )

    return all_chunks
