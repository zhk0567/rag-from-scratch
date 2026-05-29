# 后续任务清单

**全部功能已完成**（核心 + 可选 + 生产化）。

---

## 生产化增强（本轮）

- [x] **API 鉴权**：`auth.py` — 环境变量 `API_KEY`，请求头 `X-API-Key`
- [x] **运行指标**：`metrics.py` — `GET /metrics`（查询/建索引次数、平均耗时）
- [x] **多租户隔离**：`TENANT_ID` — 数据与索引目录 `data/{tenant}/`、`storage/{tenant}/`
- [x] **API 扩展**：`POST /query/stream` 流式、`POST /evaluate` 评估

---

## 能力总览

| 类别 | 模块 |
|------|------|
| RAG 核心 | `rag.py`, `loader`, `retrieval`, `vector_store` |
| 多模态 | `multimodal`, `vision_cache`, `video_loader`, `clip_index` |
| 生产 API | `api.py`, `auth.py`, `metrics.py` |
| 运维 | `evaluate.py`, `profile_ingest.py`, Docker, CI |

---

## API 示例

```powershell
$headers = @{ "X-API-Key" = "your-secret" }

# 建索引
Invoke-RestMethod -Method Post -Uri http://localhost:8000/ingest `
  -Headers $headers -ContentType "application/json" `
  -Body '{"rebuild":false}'

# 问答
Invoke-RestMethod -Method Post -Uri http://localhost:8000/query `
  -Headers $headers -ContentType "application/json" `
  -Body '{"question":"RAG 有哪些步骤？"}'

# 指标
Invoke-RestMethod -Uri http://localhost:8000/metrics -Headers $headers
```

---

## 多租户

```env
TENANT_ID=team-a
```

文档放入 `data/team-a/`，索引写入 `storage/team-a/`（路径相对项目根目录的 `data` / `storage` 子目录）。

---

## 图文统一向量空间

启用 `USE_CLIP=true` 并安装 `requirements-clip.txt`，即使用 CLIP 在同一嵌入空间检索图片与文本（无需视觉描述中转）。

---

## 命令速查

```powershell
uvicorn api:app --port 8000
streamlit run app.py
python evaluate.py
```
