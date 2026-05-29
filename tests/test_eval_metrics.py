"""评估指标单元测试。"""

from rag.evaluate import hit_at_k, keyword_hit


def test_hit_at_k():
    sources = [{"source": "a.md"}, {"source": "b.md"}]
    assert hit_at_k(sources, "a.md", 1) is True
    assert hit_at_k(sources, "c.md", 2) is False


def test_keyword_hit():
    sources = [{"excerpt": "NumPy vector store"}]
    assert keyword_hit(sources, ["NumPy"], 1) is True
    assert keyword_hit(sources, ["missing"], 1) is False
