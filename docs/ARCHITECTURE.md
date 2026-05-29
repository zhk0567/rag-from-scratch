# 架构说明

## 流程

### 建索引（ingest）

1. 扫描 `data/`（`loader`）
2. 分块（`chunker`，Markdown 标题感知）
3. Ollama 嵌入（`embedder`）
4. 写入向量库（`store_factory` → local / Qdrant / Milvus）
5. 更新 `storage/manifest.json`、可选 `graph.json`、CLIP 索引

### 问答（query）

1. 查询扩展（历史 / HyDE / 改写，`query_rewrite`）
2. 混合检索 + 重排 + 阈值 + Graph 加权 + 可选 CLIP RRF
3. 组装 prompt（`prompts`）
4. Ollama 生成（流式可选）

## 模块

| 模块 | 职责 |
|------|------|
| `rag/pipeline.py` | ingest / query 编排 |
| `rag/retrieval.py` | BM25 + 向量 RRF、重排、去重 |
| `rag/store_factory.py` | 向量库后端选择 |
| `rag/multimodal.py` | 图片 / PDF 图描述 |
| `rag/auth.py` | API Key / JWT |
| `rag/backends/` | Qdrant、Milvus 实现 |

## 入口

| 文件 | 用途 |
|------|------|
| `ingest.py` | CLI 建索引 |
| `app.py` | Streamlit |
| `api.py` | REST API |
| `evaluate.py` | Hit@K 评估 |

## 向量后端

- **local**：NumPy + `storage/embeddings.npy`，支持混合检索（BM25）
- **qdrant / milvus**：远程向量库，仅向量检索（无 BM25 混合）
