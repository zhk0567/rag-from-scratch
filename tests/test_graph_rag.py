"""graph_rag 单元测试。"""

from graph_rag import boost_hits, build_graph_from_chunks


def test_build_graph_from_markdown_chunks():
    chunks = [
        {"source": "a.md", "text": "# Intro\n\nhello"},
        {"source": "a.md", "text": "## Details\n\nworld"},
    ]
    graph = build_graph_from_chunks(chunks)
    assert "a.md" in graph
    assert "Intro" in graph["a.md"]


def test_boost_hits_when_question_matches_title():
    hits = [(0.5, {"source": "a.md", "text": "body", "chunk_index": 0})]
    graph = {"a.md": ["Intro"]}
    boosted = boost_hits(hits, "tell me about Intro", graph, boost=0.1)
    assert boosted[0][0] > 0.5
