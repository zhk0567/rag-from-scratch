"""prompts 单元测试。"""

from rag.prompts import build_rag_prompt


def test_build_rag_prompt_includes_question_and_source():
    hits = [(0.85, {"source": "sample.md", "text": "RAG 包含检索步骤"})]
    prompt = build_rag_prompt("有哪些步骤？", hits)
    assert "有哪些步骤？" in prompt
    assert "sample.md" in prompt
    assert "RAG 包含检索步骤" in prompt


def test_build_rag_prompt_empty_hits():
    prompt = build_rag_prompt("test?", [])
    assert "无相关资料" in prompt
