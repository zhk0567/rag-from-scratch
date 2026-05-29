"""chunker 单元测试。"""

from rag.chunker import split_documents
from rag.loader import Document


def test_split_markdown_by_headers():
    doc = Document(
        text="# A\n\npara a\n\n## B\n\npara b",
        source="t.md",
        doc_id="t.md",
    )
    chunks = split_documents([doc], chunk_size=200, overlap=20)
    assert len(chunks) >= 1
    assert all(c["source"] == "t.md" for c in chunks)


def test_chunk_indices_sequential():
    doc = Document(text="word " * 300, source="t.txt", doc_id="t.txt")
    chunks = split_documents([doc], chunk_size=100, overlap=10)
    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))
