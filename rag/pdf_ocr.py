"""扫描版 PDF OCR 回退（可选依赖 pytesseract、pdf2image）。"""

from pathlib import Path

from .logger import get_logger

log = get_logger()


def ocr_pdf(path: Path, lang: str = "chi_sim+eng") -> str:
    """
    对 PDF 逐页 OCR。需本机安装 Tesseract 与 Poppler。
    """
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError as e:
        raise ImportError(
            "OCR 需要: pip install pytesseract pdf2image，并安装 Tesseract OCR 与 Poppler"
        ) from e

    images = convert_from_path(str(path))
    parts: list[str] = []
    for i, img in enumerate(images, start=1):
        text = pytesseract.image_to_string(img, lang=lang)
        if text.strip():
            parts.append(f"[第 {i} 页 OCR]\n{text.strip()}")
    return "\n\n".join(parts)


def pdf_text_with_ocr_fallback(path: Path, extracted: str, min_len: int) -> str:
    """文本过少时尝试 OCR。"""
    if len(extracted.strip()) >= min_len:
        return extracted

    log.info("PDF 文本过短，尝试 OCR: %s", path.name)
    try:
        ocr_text = ocr_pdf(path)
        if len(ocr_text.strip()) >= min_len:
            return ocr_text
    except Exception as e:
        log.warning("OCR 跳过 %s: %s", path.name, e)
    return extracted
