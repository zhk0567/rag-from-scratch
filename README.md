# 从零实现 RAG 系统

基于 **Ollama**（本地嵌入 + 对话）与 **Streamlit** 的检索增强生成（RAG）教学项目。不依赖 LangChain，核心检索逻辑使用 NumPy 自实现。

## 功能

- 从 `data/` 加载 `.txt`、`.md`、`.pdf`、`.docx`、`.html`
- Markdown 标题感知分块 + 增量建索引（仅处理变更文件）
- Ollama 嵌入 → 本地向量索引（`storage/` + `manifest.json`）
- **混合检索**：向量余弦相似度 + BM25，RRF 融合 + 关键词重排
- 相似度阈值过滤、相邻块去重合并
- 流式回答、多轮对话上下文、Ollama 健康检查
- Streamlit：知识库管理、网页抓取、引用可视化、对话导出
- FastAPI：`uvicorn api:app` 提供 `/health`、`/ingest`、`/query`

## 环境要求

- Windows 10/11
- Python 3.10+
- [Ollama](https://ollama.com) 已安装并运行

## 快速开始

### 1. 克隆 / 进入项目目录

```powershell
cd f:\commercial\rag-from-scratch
```

### 2. 创建虚拟环境并安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. 配置环境变量（可选）

```powershell
Copy-Item .env.example .env
```

按需修改 `.env` 中的模型名与路径。

### 4. 拉取 Ollama 模型

确保 Ollama 已在运行（系统托盘或 `ollama serve`），然后：

```powershell
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
```

可用 `ollama list` 确认模型已下载。

### 5. 建索引

项目已包含示例文档 `data/sample.md`：

```powershell
python ingest.py
```

强制清空并重建：

```powershell
python ingest.py --rebuild
```

增量更新（默认，仅处理新增/修改/删除的文件）：

```powershell
python ingest.py
```

成功后在 `storage/` 下会生成 `embeddings.npy`、`index.json`、`manifest.json`。

### 6. 启动 Web 界面

```powershell
streamlit run app.py
```

浏览器打开后，在侧栏可上传文档、**增量索引**或**全量重建**，在主界面流式问答。每次回答可展开查看引用片段与相似度条形图。

### 7. 启动 API（可选）

```powershell
uvicorn api:app --reload --port 8000
```

文档：http://localhost:8000/docs

### 8. 运行测试

```powershell
pytest tests/ -v
```

## 项目结构

```
├── app.py              # Streamlit 界面
├── api.py              # FastAPI REST
├── ingest.py           # CLI 建索引
├── rag.py              # RAG 流水线
├── retrieval.py        # 混合检索 / 重排 / 去重
├── ollama_health.py    # 健康检查
├── index_manifest.py   # 增量索引清单
├── prompts.py          # 提示词模板
├── tests/              # 单元测试
├── data/               # 知识库文档
└── storage/            # 向量索引（自动生成）
```

详见 [ROADMAP.md](ROADMAP.md) 了解已完成与待办项。

## 配置说明

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama 地址 |
| `EMBED_MODEL` | `nomic-embed-text` | 嵌入模型 |
| `CHAT_MODEL` | `qwen2.5:7b` | 对话模型 |
| `CHUNK_SIZE` | `500` | 分块字符数 |
| `CHUNK_OVERLAP` | `80` | 块间重叠 |
| `TOP_K` | `4` | 最终送入 LLM 的条数 |
| `RETRIEVE_N` | `12` | 混合检索候选数 |
| `SIMILARITY_THRESHOLD` | `0.3` | 相似度下限 |
| `HYBRID_SEARCH` | `true` | 是否启用 BM25+向量融合 |

更换嵌入模型后必须执行 `python ingest.py --rebuild` 重建索引。

## 演示问题

针对 `data/sample.md` 可尝试：

- RAG 流水线包含哪些步骤？
- 默认的嵌入模型是什么？
- 向量存储使用什么技术？

## 常见问题

**嵌入或生成报错 / 连接失败**

- 确认 Ollama 已启动：访问 http://localhost:11434
- 确认已 pull 对应模型：`ollama list`

**索引为空**

- 确认 `data/` 下有支持的文件且内容非空
- 重新运行 `python ingest.py --rebuild`

**PDF 检索效果差**

- 仅适合可选中文本的 PDF；扫描版需 OCR 预处理

## 许可证

MIT（可按需修改）
