"""文档分块：段落边界 + Markdown 标题结构。"""

import re
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


def _split_markdown_sections(text: str) -> list[str]:
    """按 Markdown 标题（# ~ ###）切分为章节。"""
    pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        return _split_paragraphs(text)

    sections: list[str] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)

    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.insert(0, preamble)
    return sections if sections else _split_paragraphs(text)


def _merge_to_chunks(paragraphs: list[str], chunk_size: int, overlap: int) -> list[str]:
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
                    chunks.append(para[start:end])
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
    all_chunks: list[Chunk] = []

    for doc in docs:
        source = doc["source"].lower()
        if source.endswith(".md"):
            sections = _split_markdown_sections(doc["text"])
        else:
            sections = _split_paragraphs(doc["text"])

        texts = _merge_to_chunks(sections, chunk_size, overlap)
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
