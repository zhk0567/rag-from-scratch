"""multimodal 单元测试（不调用 Ollama）。"""

from rag.multimodal import IMAGE_EXTENSIONS, merge_text_and_vision


def test_merge_text_and_vision():
    base = "正文内容"
    vision = "[第 1 页 · 图片 1]\n图表显示销售额上升"
    merged = merge_text_and_vision(base, vision)
    assert "正文内容" in merged
    assert "图片与插图描述" in merged
    assert "销售额" in merged


def test_merge_vision_only():
    merged = merge_text_and_vision("", "仅图片描述")
    assert "仅图片描述" in merged


def test_image_extensions():
    assert ".png" in IMAGE_EXTENSIONS
    assert ".pdf" not in IMAGE_EXTENSIONS
