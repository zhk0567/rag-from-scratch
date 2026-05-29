# 后续任务清单

全部核心路线已完成。

---

## P5 — 进阶 RAG

- [x] Agentic RAG
- [x] Graph RAG（轻量）
- [x] 评估集与指标
- [x] **多模态**：`multimodal.py` — PDF 内嵌图 + png/jpg/webp，Ollama 视觉模型描述后嵌入

---

## 多模态使用

1. 拉取视觉模型：`ollama pull llava`（或 `moondream`、`llava:13b` 等）
2. `.env` 设置：`USE_MULTIMODAL=true`，`VISION_MODEL=llava`
3. 将含图表的 PDF 或图片放入 `data/`，执行 `python ingest.py --rebuild`
4. 图片描述会与正文一并分块、嵌入，问答时可检索到图表内容

---

## 可选增强（未列入 MVP）

- [ ] 专用 CLIP / 多模态嵌入向量（与文本向量统一空间）
- [ ] 视频帧抽取与描述
- [ ] 在线增量更新视觉缓存，避免重复 describe

---

## 命令速查

```powershell
python ingest.py --rebuild
python evaluate.py
streamlit run app.py
uvicorn api:app --port 8000
```
