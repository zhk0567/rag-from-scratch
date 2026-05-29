# 后续任务清单

**项目已结项 v1.0.0**（2025-05-29）

**全部开发任务已完成**（含扩展生产化与向量库）。
---

## 扩展能力（本轮）

- [x] **Qdrant 向量库**：`VECTOR_BACKEND=qdrant`，`rag/backends/qdrant_store.py`
- [x] **Milvus 向量库**：`VECTOR_BACKEND=milvus`，`rag/backends/milvus_store.py`
- [x] **JWT 鉴权**：`POST /auth/token`，`AUTH_MODE=jwt|both`
- [x] **Prometheus 导出**：`GET /metrics/prometheus`

---

## 向量库切换

| 后端 | 配置 | 依赖 |
|------|------|------|
| local（默认） | `VECTOR_BACKEND=local` | 无 |
| Qdrant | `VECTOR_BACKEND=qdrant` | `requirements/qdrant.txt` |
| Milvus | `VECTOR_BACKEND=milvus` | `requirements/milvus.txt` |

```powershell
docker compose --profile qdrant up -d qdrant
pip install -r requirements/qdrant.txt
# .env: VECTOR_BACKEND=qdrant, QDRANT_URL=http://localhost:6333
python ingest.py --rebuild
```

---

## 鉴权

| AUTH_MODE | 说明 |
|-----------|------|
| `none` | 不鉴权 |
| `api_key` | 请求头 `X-API-Key` |
| `jwt` | `Authorization: Bearer <token>` |
| `both` | 二者任一有效即可 |

```powershell
# 获取 JWT
Invoke-RestMethod -Method Post -Uri http://localhost:8000/auth/token `
  -ContentType application/json `
  -Body '{"username":"admin","password":"admin"}'
```

---

## 监控

- JSON 指标：`GET /metrics`（需鉴权）
- Prometheus：`GET /metrics/prometheus`

---

## 命令速查

```powershell
uvicorn api:app --port 8000
streamlit run app.py
pytest tests/ -v
```
