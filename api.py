"""FastAPI REST 服务（鉴权 + 指标 + 流式问答）。"""

import time
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from rag import config
from rag.auth import require_auth
from rag.evaluate import DEFAULT_EVAL, load_eval_cases, run_eval
from rag.jwt_auth import create_access_token
from rag.metrics import record_ingest, record_query, snapshot
from rag.ollama_health import get_health_status
from rag.pipeline import has_index, ingest, query, query_with_stream
from rag.prometheus_metrics import observe_ingest, observe_query, render_prometheus

app = FastAPI(title="RAG API", version="1.2.0")


class IngestRequest(BaseModel):
    rebuild: bool = False
    incremental: bool = True


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = None


class EvalRequest(BaseModel):
    top_k: int | None = None


class TokenRequest(BaseModel):
    username: str | None = None
    password: str | None = None
    api_key: str | None = None


def _record_ingest_metrics(duration_ms: float, ok: bool) -> None:
    record_ingest(duration_ms, ok=ok)
    if config.PROMETHEUS_ENABLED:
        observe_ingest(duration_ms / 1000.0, ok=ok)


def _record_query_metrics(duration_ms: float, ok: bool) -> None:
    record_query(duration_ms, ok=ok)
    if config.PROMETHEUS_ENABLED:
        observe_query(duration_ms / 1000.0, ok=ok)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "rag-from-scratch",
        "docs": "/docs",
        "tenant": config.TENANT_ID or "default",
        "vector_backend": config.VECTOR_BACKEND,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return get_health_status()


@app.post("/auth/token")
def issue_token(req: TokenRequest) -> dict[str, str]:
    if not config.JWT_SECRET:
        raise HTTPException(400, detail="JWT_SECRET not configured")

    ok = False
    subject = ""
    if req.api_key and config.API_KEY and req.api_key == config.API_KEY:
        ok = True
        subject = "api_key_user"
    elif (
        req.username == config.ADMIN_USERNAME
        and req.password == config.ADMIN_PASSWORD
        and config.ADMIN_PASSWORD
    ):
        ok = True
        subject = req.username or "admin"

    if not ok:
        raise HTTPException(401, detail="Invalid credentials")

    return {"access_token": create_access_token(subject), "token_type": "bearer"}


@app.get("/metrics")
def metrics(_: None = Depends(require_auth)) -> dict[str, Any]:
    body = snapshot()
    body["tenant_id"] = config.TENANT_ID or "default"
    body["vector_backend"] = config.VECTOR_BACKEND
    return body


@app.get("/metrics/prometheus")
def metrics_prometheus() -> Response:
    if not config.PROMETHEUS_ENABLED:
        raise HTTPException(404, detail="Prometheus export disabled")
    payload, content_type = render_prometheus()
    return Response(content=payload, media_type=content_type)


@app.post("/ingest", dependencies=[Depends(require_auth)])
def api_ingest(req: IngestRequest) -> dict[str, Any]:
    h = get_health_status()
    if not h["ok"]:
        raise HTTPException(503, detail=h["messages"])
    t0 = time.perf_counter()
    try:
        result = ingest(rebuild=req.rebuild, incremental=req.incremental)
        _record_ingest_metrics((time.perf_counter() - t0) * 1000, ok=True)
        return result
    except Exception:
        _record_ingest_metrics((time.perf_counter() - t0) * 1000, ok=False)
        raise


@app.post("/query", dependencies=[Depends(require_auth)])
def api_query(req: QueryRequest) -> dict[str, Any]:
    if not has_index():
        raise HTTPException(400, detail="Index not built. POST /ingest first.")
    h = get_health_status()
    if not h["ok"]:
        raise HTTPException(503, detail=h["messages"])
    t0 = time.perf_counter()
    try:
        result = query(req.question, top_k=req.top_k)
        _record_query_metrics((time.perf_counter() - t0) * 1000, ok=True)
        return result
    except Exception:
        _record_query_metrics((time.perf_counter() - t0) * 1000, ok=False)
        raise


@app.post("/query/stream", dependencies=[Depends(require_auth)])
def api_query_stream(req: QueryRequest) -> StreamingResponse:
    if not has_index():
        raise HTTPException(400, detail="Index not built. POST /ingest first.")
    h = get_health_status()
    if not h["ok"]:
        raise HTTPException(503, detail=h["messages"])

    t0 = time.perf_counter()

    def event_stream():
        try:
            stream, _sources = query_with_stream(req.question, top_k=req.top_k)
            for token in stream:
                yield token
            _record_query_metrics((time.perf_counter() - t0) * 1000, ok=True)
        except Exception as e:
            _record_query_metrics((time.perf_counter() - t0) * 1000, ok=False)
            yield f"\n[Error] {e}"

    return StreamingResponse(event_stream(), media_type="text/plain; charset=utf-8")


@app.post("/evaluate", dependencies=[Depends(require_auth)])
def api_evaluate(req: EvalRequest) -> dict[str, Any]:
    if not has_index():
        raise HTTPException(400, detail="Index not built. POST /ingest first.")
    cases = load_eval_cases(DEFAULT_EVAL)
    return run_eval(cases, top_k=req.top_k)
