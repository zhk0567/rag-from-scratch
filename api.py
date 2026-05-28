"""FastAPI REST 服务。"""

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config
from ollama_health import get_health_status
from rag import has_index, ingest, query

app = FastAPI(title="RAG API", version="1.0.0")


class IngestRequest(BaseModel):
    rebuild: bool = False
    incremental: bool = True


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = None


@app.get("/health")
def health() -> dict[str, Any]:
    return get_health_status()


@app.post("/ingest")
def api_ingest(req: IngestRequest) -> dict[str, Any]:
    h = get_health_status()
    if not h["ok"]:
        raise HTTPException(503, detail=h["messages"])
    return ingest(rebuild=req.rebuild, incremental=req.incremental)


@app.post("/query")
def api_query(req: QueryRequest) -> dict[str, Any]:
    if not has_index():
        raise HTTPException(400, detail="Index not built. POST /ingest first.")
    h = get_health_status()
    if not h["ok"]:
        raise HTTPException(503, detail=h["messages"])
    return query(req.question, top_k=req.top_k)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "rag-from-scratch", "docs": "/docs"}
