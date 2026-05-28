"""retrieval 单元测试。"""

from retrieval import SimpleBM25, filter_by_threshold, merge_similar_chunks


def test_bm25_ranks_relevant_doc():
    corpus = ["python programming language", "banana fruit yellow"]
    bm25 = SimpleBM25(corpus)
    scores = bm25.score_all("python code")
    assert scores[0] > scores[1]


def test_filter_by_threshold():
    hits = [(0.9, {"text": "a"}), (0.2, {"text": "b"})]
    filtered = filter_by_threshold(hits, 0.5)
    assert len(filtered) == 1


def test_merge_similar_chunks():
    hits = [
        (0.8, {"source": "a.md", "text": "hello world", "chunk_index": 0}),
        (0.75, {"source": "a.md", "text": "hello world extended", "chunk_index": 1}),
    ]
    merged = merge_similar_chunks(hits, threshold=0.5)
    assert len(merged) == 1
