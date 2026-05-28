"""提示词模板（便于测试与复用）。"""

from typing import Any


def build_rag_prompt(question: str, hits: list[tuple[float, dict[str, Any]]]) -> str:
    ref_lines: list[str] = []
    for i, (score, meta) in enumerate(hits, start=1):
        source = meta.get("source", "unknown")
        text = meta.get("text", "")
        ref_lines.append(f"[{i}] (来源: {source}, 相似度: {score:.3f})\n{text}")

    references = "\n\n".join(ref_lines) if ref_lines else "（无相关资料）"

    return f"""你是基于检索内容的问答助手。仅根据下列「参考资料」回答；资料不足则明确说不知道，不要编造。

【参考资料】
{references}

【问题】
{question}"""
