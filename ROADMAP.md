# 后续任务清单

基于当前 MVP 的演进路线。已完成项标记为 `[x]`。

---

## P0 — 稳定性与可用性

- [x] Ollama 健康检查
- [x] 增量建索引
- [x] 流式回答
- [x] 错误与日志
- [x] 单元测试

---

## P1 — 检索质量

- [x] 相似度阈值
- [x] 重排序（Rerank）
- [x] 混合检索
- [x] **HyDE / 查询改写**：`query_rewrite.py`（`USE_HYDE`、`USE_QUERY_REWRITE`）
- [x] Markdown 标题分块
- [x] 去重与合并

---

## P2 — 功能扩展

- [x] 多轮对话
- [x] FastAPI 服务
- [x] 更多文档格式
- [x] **扫描版 PDF OCR**：`pdf_ocr.py` + `requirements-ocr.txt`（可选依赖）
- [x] 网页抓取
- [x] 对话导出

---

## P3 — 工程与部署

- [x] Docker Compose
- [x] GitHub Actions CI
- [x] 配置校验
- [x] 索引版本化
- [x] **性能 profiling**：`profile_ingest.py` + `EMBED_WORKERS` 并发嵌入

---

## P4 — 体验与展示

- [x] 检索可视化
- [x] 侧栏知识库管理
- [x] 暗色主题
- [x] 示例问题
- [x] **中英文切换**：`i18n.py` + Streamlit 语言选择

---

## P5 — 进阶 RAG

- [x] **Agentic RAG**：低分时 `rewrite_query` 二次检索（`USE_AGENTIC_RAG`）
- [x] **Graph RAG（轻量）**：`graph_rag.py` 章节标题加权
- [x] **评估集与指标**：`data/eval_qa.json` + `evaluate.py`（Hit@K）
- [ ] **多模态**：图片 PDF 嵌入（需多模态模型，暂未实现）

---

## 使用新增能力

```powershell
# 评估检索质量（需已建索引且 Ollama 可用）
python evaluate.py

# 嵌入批大小性能探测
python profile_ingest.py --sizes 1 4 8 16

# 启用 HyDE（.env 中 USE_HYDE=true）

# OCR 可选依赖
pip install -r requirements-ocr.txt
```

---

## 里程碑状态

| 里程碑 | 状态 |
|--------|------|
| A 稳定日常使用 | 完成 |
| B 检索变准 | 完成 |
| C 可演示/集成 | 完成 |
| P5 多模态 | 待做 |
