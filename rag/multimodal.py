"""多模态：用 Ollama 视觉模型描述图片，转为可检索文本。"""

import tempfile
from pathlib import Path

import ollama

from . import config
from .logger import get_logger
from .vision_cache import get_cached_description, set_cached_description

log = get_logger()

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def _client():
    return ollama.Client(host=config.OLLAMA_HOST)


def describe_image_bytes(data: bytes, label: str = "image") -> str:
    """对图片字节生成描述，带视觉缓存。"""
    cached = get_cached_description(data)
    if cached is not None:
        log.info("视觉缓存命中: %s", label)
        return cached

    prompt = (
        "请用中文详细描述这张图片中的文字、图表、流程和关键信息。"
        "输出适合作为文档检索用的纯文本，不要 Markdown 标题。"
    )
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(data)
            tmp_path = Path(f.name)
        resp = _client().chat(
            model=config.VISION_MODEL,
            messages=[{"role": "user", "content": prompt, "images": [str(tmp_path.resolve())]}],
        )
        tmp_path.unlink(missing_ok=True)
        text = (resp.get("message") or {}).get("content", "").strip()
        if text:
            set_cached_description(data, text)
            log.info("图片描述完成: %s (%d 字)", label, len(text))
        return text
    except Exception as e:
        log.warning("图片描述失败 %s: %s", label, e)
        return ""


def describe_image(path: Path) -> str:
    return describe_image_bytes(path.read_bytes(), label=path.name)


def extract_pdf_image_descriptions(pdf_path: Path) -> str:
    """提取 PDF 内嵌图片并逐一描述。"""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    count = 0
    max_images = config.MULTIMODAL_MAX_IMAGES

    for page_num, page in enumerate(reader.pages, start=1):
        images = getattr(page, "images", None)
        if not images:
            continue
        for img_idx, image in enumerate(images):
            if count >= max_images:
                log.info("已达图片上限 %d，停止提取", max_images)
                break
            try:
                data = image.data
                desc = describe_image_bytes(
                    data, label=f"{pdf_path.name} p{page_num} #{img_idx}"
                )
                if desc:
                    parts.append(f"[第 {page_num} 页 · 图片 {img_idx + 1}]\n{desc}")
                    count += 1
            except Exception as e:
                log.warning("PDF 图片提取失败 p%d #%d: %s", page_num, img_idx, e)

    return "\n\n".join(parts)


def merge_text_and_vision(base_text: str, vision_text: str) -> str:
    base_text = (base_text or "").strip()
    vision_text = (vision_text or "").strip()
    if not vision_text:
        return base_text
    block = f"## 图片与插图描述\n\n{vision_text}"
    return f"{base_text}\n\n{block}".strip() if base_text else block


def describe_image_file(path: Path) -> str:
    desc = describe_image(path)
    if desc:
        return f"# 图片文档: {path.name}\n\n{desc}"
    return ""
