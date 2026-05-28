# 后续任务清单



基于当前 MVP 的演进路线。已完成项标记为 `[x]`。



---



## P0 — 稳定性与可用性（建议优先）



- [x] **Ollama 健康检查**：`ollama_health.py`，启动时检测服务与模型（CLI / Streamlit）

- [x] **增量建索引**：`index_manifest.py` + `ingest(incremental=True)`，仅处理变更文件

- [x] **流式回答**：`query_with_stream()` + Streamlit `write_stream`

- [x] **错误与日志**：`logger.py`，记录 ingest / query 耗时

- [x] **单元测试**：`tests/` 覆盖 chunker、vector_store、prompts、retrieval



---



## P1 — 检索质量



- [x] **相似度阈值**：`SIMILARITY_THRESHOLD` + `filter_by_threshold`

- [x] **重排序（Rerank）**：`rerank_by_keywords` 轻量关键词重排

- [x] **混合检索**：`retrieval.py` BM25 + 向量 RRF 融合

- [ ] **HyDE / 查询改写**：需额外 LLM 调用（当前为多轮历史扩展查询）

- [x] **分块策略优化**：Markdown 按 `#` 标题切分

- [x] **去重与合并**：`merge_similar_chunks`



---



## P2 — 功能扩展



- [x] **多轮对话**：`CHAT_HISTORY_TURNS` + `history` 传入检索查询

- [x] **FastAPI 服务**：`api.py`（`/health`、`/ingest`、`/query`）

- [x] **更多文档格式**：`.docx`、`.html`（需 python-docx / beautifulsoup4）

- [ ] **扫描版 PDF**：OCR（Tesseract / PaddleOCR）未接入

- [x] **网页抓取**：`web_scrape.py` + Streamlit URL 输入

- [x] **对话导出**：Streamlit 导出 JSON



---



## P3 — 工程与部署



- [x] **Docker Compose**：`Dockerfile` + `docker-compose.yml`

- [x] **GitHub Actions CI**：`.github/workflows/ci.yml`

- [x] **配置校验**：`config.validate_config()`

- [x] **索引版本化**：`manifest.json` 记录 `embed_model`，`check_index_version()`

- [ ] **性能 profiling**：大批量 ingest 批大小/并发优化（待实测调参）



---



## P4 — 体验与展示



- [x] **检索可视化**：引用 expander 内 `bar_chart` 相似度

- [x] **侧栏知识库管理**：列出文件、单文件删除

- [x] **暗色主题**：Streamlit 自定义 CSS

- [x] **示例问题快捷按钮**：侧栏 DEMO_QUESTIONS

- [ ] **中英文切换**：界面 i18n 未实现



---



## P5 — 进阶 RAG（可选研究向）



- [ ] **Agentic RAG**：检索不足时自动换 query 再检索

- [ ] **Graph RAG**：实体关系抽取 + 图检索

- [ ] **评估集与指标**：Hit@K、LLM judge

- [ ] **多模态**：图片 PDF / 插图嵌入



---



## 建议实施顺序（3 个里程碑）



### 里程碑 A：能稳定日常使用 — 已完成

P0 全部 → 相似度阈值 → 增量建索引



### 里程碑 B：检索明显变准 — 基本完成

混合检索 → Rerank → Markdown 结构分块 → （评估集待做）



### 里程碑 C：可对外演示 / 集成 — 基本完成

FastAPI → Docker → CI → 流式回答



---



## 待办 Issue 标题



1. `feat: HyDE query expansion via Ollama`

2. `feat: OCR pipeline for scanned PDFs`

3. `perf: profile and tune ingest batch concurrency`

4. `feat: i18n for Streamlit UI`

