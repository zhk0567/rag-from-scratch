# 从零实现 RAG 系统

基于 **Ollama**（本地嵌入 + 对话）与 **Streamlit** 的检索增强生成（RAG）教学项目。不依赖 LangChain，核心检索逻辑使用 NumPy 自实现。

## 功能

- 从 `data/` 加载 `.txt`、`.md`、`.pdf`
- 文本分块（可配置大小与重叠）
- Ollama 嵌入 → 本地向量索引（`storage/`）
- 余弦相似度 Top-K 检索
- 将检索上下文拼入提示词，由 Ollama 生成回答
- Streamlit 界面：上传文档、重建索引、聊天并展示引用来源

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

成功后在 `storage/` 下会生成 `embeddings.npy` 与 `index.json`。

### 6. 启动 Web 界面

```powershell
streamlit run app.py
```

浏览器打开后，在侧栏可上传文档、点击「重建索引」，在主界面进行问答。每次回答可展开查看引用片段与相似度。

## 项目结构

```
├── app.py              # Streamlit 界面
├── ingest.py           # CLI 建索引
├── rag.py              # RAG 流水线（ingest + query）
├── loader.py           # 文档加载
├── chunker.py          # 文本分块
├── embedder.py         # Ollama 嵌入
├── vector_store.py     # 向量存储与检索
├── config.py           # 配置
├── data/               # 知识库文档
├── storage/            # 向量索引（自动生成，已 gitignore）
├── requirements.txt
└── .env.example
```

## 配置说明

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama 地址 |
| `EMBED_MODEL` | `nomic-embed-text` | 嵌入模型 |
| `CHAT_MODEL` | `qwen2.5:7b` | 对话模型 |
| `CHUNK_SIZE` | `500` | 分块字符数 |
| `CHUNK_OVERLAP` | `80` | 块间重叠 |
| `TOP_K` | `4` | 检索条数 |

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
