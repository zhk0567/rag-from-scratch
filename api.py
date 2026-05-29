"""FastAPI REST 服务（鉴权 + 指标 + 流式问答）。"""

import time
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import config
from auth import require_api_key
from evaluate import load_eval_cases, run_eval
from metrics import record_ingest, record_query, snapshot
from ollama_health import get_health_status
from rag import has_index, ingest, query, query_with_stream

app = FastAPI(title="RAG API", version="1.1.0")


class IngestRequest(BaseModel):
    rebuild: bool = False
    incremental: bool = True


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = None


class EvalRequest(BaseModel):
    top_k: int | None = None


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "rag-from-scratch", "docs": "/docs", "tenant": config.TENANT_ID or "default"}


@app.get("/health")
def health() -> dict[str, Any]:
    return get_health_status()


@app.get("/metrics")
def metrics(_: None = Depends(require_api_key)) -> dict[str, Any]:
    body = snapshot()
    body["tenant_id"] = config.TENANT_ID or "default"
    return body


@app.post("/ingest", dependencies=[Depends(require_api_key)])
def api_ingest(req: IngestRequest) -> dict[str, Any]:
    h = get_health_status()
    if not h["ok"]:
        raise HTTPException(503, detail=h["messages"])
    t0 = time.perf_counter()
    try:
        result = ingest(rebuild=req.rebuild, incremental=req.incremental)
        record_ingest((time.perf_counter() - t0) * 1000, ok=True)
        return result
    except Exception:
        record_ingest((time.perf_counter() - t0) * 1000, ok=False)
        raise


@app.post("/query", dependencies=[Depends(require_api_key)])
def api_query(req: QueryRequest) -> dict[str, Any]:
    if not has_index():
        raise HTTPException(400, detail="Index not built. POST /ingest first.")
    h = get_health_status()
    if not h["ok"]:
        raise HTTPException(503, detail=h["messages"])
    t0 = time.perf_counter()
    try:
        result = query(req.question, top_k=req.top_k)
        record_query((time.perf_counter() - t0) * 1000, ok=True)
        return result
    except Exception:
        record_query((time.perf_counter() - t0) * 1000, ok=False)
        raise


@app.post("/query/stream", dependencies=[Depends(require_api_key)])
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
            record_query((time.perf_counter() - t0) * 1000, ok=True)
        except Exception as e:
            record_query((time.perf_counter() - t0) * 1000, ok=False)
            yield f"\n[Error] {e}"

    return StreamingResponse(event_stream(), media_type="text/plain; charset=utf-8")


@app.post("/evaluate", dependencies=[Depends(require_api_key)])
def api_evaluate(req: EvalRequest) -> dict[str, Any]:
    if not has_index():
        raise HTTPException(400, detail="Index not built. POST /ingest first.")
    from pathlib import Path

    cases = load_eval_cases(Path(__file__).parent / "data" / "eval_qa.json")
    return run_eval(cases, top_k=req.top_k)
