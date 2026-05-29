"""简易运行指标（内存计数），供 /metrics 暴露。"""

import threading
import time
from typing import Any

_lock = threading.Lock()
_stats: dict[str, Any] = {
    "ingest_count": 0,
    "query_count": 0,
    "error_count": 0,
    "ingest_total_ms": 0.0,
    "query_total_ms": 0.0,
    "started_at": time.time(),
}


def record_ingest(duration_ms: float, ok: bool = True) -> None:
    with _lock:
        if ok:
            _stats["ingest_count"] += 1
            _stats["ingest_total_ms"] += duration_ms
        else:
            _stats["error_count"] += 1


def record_query(duration_ms: float, ok: bool = True) -> None:
    with _lock:
        if ok:
            _stats["query_count"] += 1
            _stats["query_total_ms"] += duration_ms
        else:
            _stats["error_count"] += 1


def snapshot() -> dict[str, Any]:
    with _lock:
        data = dict(_stats)
    uptime = time.time() - data["started_at"]
    data["uptime_sec"] = round(uptime, 1)
    if data["query_count"]:
        data["query_avg_ms"] = round(data["query_total_ms"] / data["query_count"], 2)
    else:
        data["query_avg_ms"] = 0
    if data["ingest_count"]:
        data["ingest_avg_ms"] = round(data["ingest_total_ms"] / data["ingest_count"], 2)
    else:
        data["ingest_avg_ms"] = 0
    return data


def reset() -> None:
    with _lock:
        _stats.update(
            {
                "ingest_count": 0,
                "query_count": 0,
                "error_count": 0,
                "ingest_total_ms": 0.0,
                "query_total_ms": 0.0,
                "started_at": time.time(),
            }
        )
