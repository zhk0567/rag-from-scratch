# Changelog

## [1.0.0] - 2025-05-29

### Added

- RAG pipeline: load, chunk, embed, hybrid retrieve, Ollama generate
- Streamlit UI with i18n (zh/en), citations, incremental index
- FastAPI: health, ingest, query, stream, evaluate, metrics, JWT
- Multimodal: vision describe, video frames, optional CLIP dual-index
- Vector backends: local (NumPy), Qdrant, Milvus
- Agentic re-retrieval, graph section boost, vision description cache
- Docker Compose, GitHub Actions CI, 22+ unit tests

### Changed

- Restructured codebase into `rag/` Python package
- Documentation under `docs/`, optional deps under `requirements/`
