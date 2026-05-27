"""从 data 目录加载文档。"""

from pathlib import Path
from typing import TypedDict

from pypdf import PdfReader


class Document(TypedDict):
    text: str
    source: str
    doc_id: str


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            parts.append(f"[第 {i} 页]\n{text}")
    return "\n\n".join(parts)


def load_documents(data_dir: Path) -> list[Document]:
    """遍历 data_dir，加载所有支持的文档。"""
    documents: list[Document] = []
    if not data_dir.exists():
        return documents

    files = sorted(
        p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    for path in files:
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                text = _read_pdf(path)
            else:
                text = _read_text_file(path)
        except Exception as e:
            print(f"跳过 {path.name}: {e}")
            continue

        text = text.strip()
        if not text:
            print(f"跳过空文件: {path.name}")
            continue

        rel_source = str(path.relative_to(data_dir))
        documents.append(
            Document(
                text=text,
                source=rel_source,
                doc_id=rel_source,
            )
        )

    return documents
