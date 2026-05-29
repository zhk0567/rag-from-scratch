# 后续任务清单

**全部计划内能力已完成**（含可选增强）。

---

## 已完成：核心 + 可选增强

| 能力 | 模块 |
|------|------|
| RAG 全流程 | `rag.py`, `loader`, `chunker`, `embedder`, `vector_store` |
| 混合检索 / Agentic / Graph | `retrieval.py`, `query_rewrite.py`, `graph_rag.py` |
| 多模态（视觉描述） | `multimodal.py` |
| **视觉描述缓存** | `vision_cache.py` — 相同图片不重复调用视觉模型 |
| **视频帧描述** | `video_loader.py` — mp4/webm 等抽帧入库 |
| **CLIP 双索引** | `clip_index.py` — 图文跨模态 RRF 融合（可选） |
| 评估 / 性能探测 | `evaluate.py`, `profile_ingest.py` |
| API / Docker / CI | `api.py`, `Dockerfile`, `.github/workflows/ci.yml` |

---

## 配置速查

```env
# 视觉缓存（默认开）
USE_VISION_CACHE=true

# 视频（需 requirements-video.txt）
VIDEO_FRAME_INTERVAL_SEC=5
VIDEO_MAX_FRAMES=20

# CLIP 双索引（需 requirements-clip.txt）
USE_CLIP=false
CLIP_MODEL=clip-ViT-B-32
```

---

## 命令

```powershell
pip install -r requirements-video.txt   # 视频
pip install -r requirements-clip.txt    # CLIP
python ingest.py --rebuild
streamlit run app.py
```

---

## 未来可探索（超出当前仓库范围）

- 原生多模态嵌入 API（单向量空间、无需描述中转）
- 分布式向量库与多租户
- 生产级监控与鉴权
