"""metrics 单元测试。"""

from metrics import record_ingest, record_query, reset, snapshot


def test_metrics_snapshot():
    reset()
    record_query(100.0)
    record_query(200.0)
    record_ingest(500.0)
    s = snapshot()
    assert s["query_count"] == 2
    assert s["ingest_count"] == 1
    assert s["query_avg_ms"] == 150.0
