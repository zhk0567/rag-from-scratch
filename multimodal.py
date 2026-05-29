"""多模态：用 Ollama 视觉模型描述图片，转为可检索文本。"""

import tempfile
from pathlib import Path

import ollama

import config
from logger import get_logger

log = get_logger()

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def _client():
    return ollama.Client(host=config.OLLAMA_HOST)


def describe_image(path: Path) -> str:
    """对单张图片生成中文描述，供嵌入索引。"""
    prompt = (
        "请用中文详细描述这张图片中的文字、图表、流程和关键信息。"
        "输出适合作为文档检索用的纯文本，不要 Markdown 标题。"
    )
    try:
        resp = _client().chat(
            model=config.VISION_MODEL,
            messages=[{"role": "user", "content": prompt, "images": [str(path.resolve())]}],
        )
        text = (resp.get("message") or {}).get("content", "").strip()
        if text:
            log.info("图片描述完成: %s (%d 字)", path.name, len(text))
        return text
    except Exception as e:
        log.warning("图片描述失败 %s: %s", path.name, e)
        return ""


def extract_pdf_image_descriptions(pdf_path: Path) -> str:
    """提取 PDF 内嵌图片并逐一描述。"""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    count = 0
    max_images = config.MULTIMODAL_MAX_IMAGES

    with tempfile.TemporaryDirectory(prefix="rag_mm_") as tmpdir:
        tmp = Path(tmpdir)
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
                    name = getattr(image, "name", f"p{page_num}_{img_idx}.png")
                    ext = Path(name).suffix or ".png"
                    img_path = tmp / f"page{page_num}_{img_idx}{ext}"
                    img_path.write_bytes(data)
                    desc = describe_image(img_path)
                    if desc:
                        parts.append(f"[第 {page_num} 页 · 图片 {img_idx + 1}]\n{desc}")
                        count += 1
                except Exception as e:
                    log.warning("PDF 图片提取失败 p%d #%d: %s", page_num, img_idx, e)

    return "\n\n".join(parts)


def merge_text_and_vision(base_text: str, vision_text: str) -> str:
    """合并正文与图片描述。"""
    base_text = (base_text or "").strip()
    vision_text = (vision_text or "").strip()
    if not vision_text:
        return base_text
    block = f"## 图片与插图描述\n\n{vision_text}"
    return f"{base_text}\n\n{block}".strip() if base_text else block


def describe_image_file(path: Path) -> str:
    """独立图片文件作为文档正文。"""
    desc = describe_image(path)
    if desc:
        return f"# 图片文档: {path.name}\n\n{desc}"
    return ""
