"""Prometheus 指标导出。"""

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

QUERY_TOTAL = Counter("rag_queries_total", "Total RAG query requests", ["status"])
INGEST_TOTAL = Counter("rag_ingest_total", "Total ingest operations", ["status"])
QUERY_LATENCY = Histogram(
    "rag_query_latency_seconds",
    "RAG query latency in seconds",
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)
INGEST_LATENCY = Histogram(
    "rag_ingest_latency_seconds",
    "RAG ingest latency in seconds",
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
)


def observe_query(duration_sec: float, ok: bool = True) -> None:
    status = "ok" if ok else "error"
    QUERY_TOTAL.labels(status=status).inc()
    if ok:
        QUERY_LATENCY.observe(duration_sec)


def observe_ingest(duration_sec: float, ok: bool = True) -> None:
    status = "ok" if ok else "error"
    INGEST_TOTAL.labels(status=status).inc()
    if ok:
        INGEST_LATENCY.observe(duration_sec)


def render_prometheus() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
