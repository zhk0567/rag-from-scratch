"""Streamlit RAG 问答界面。"""

import json
from datetime import datetime
from pathlib import Path

import streamlit as st

import config
import rag
from index_manifest import check_index_version, scan_data_dir
from loader import SUPPORTED_EXTENSIONS
from ollama_health import get_health_status
from vector_store import VectorStore

st.set_page_config(page_title="RAG 问答", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    </style>
    """,
    unsafe_allow_html=True,
)

DEMO_QUESTIONS = [
    "RAG 流水线包含哪些步骤？",
    "默认的嵌入模型是什么？",
    "向量存储使用什么技术？",
]


@st.cache_resource
def _index_ready() -> bool:
    return rag.has_index()


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"引用片段（{len(sources)} 条）"):
        scores = [s["score"] for s in sources]
        st.bar_chart(
            {f"{s['source']} #{s['chunk_index']}": s["score"] for s in sources},
            horizontal=True,
        )
        for i, src in enumerate(sources, start=1):
            st.markdown(
                f"**[{i}]** `{src['source']}` "
                f"(相似度 {src['score']:.3f}, 块 #{src['chunk_index']})"
            )
            st.text(src["excerpt"])


st.title("检索增强生成（RAG）演示")
st.caption("基于本地 Ollama + 混合检索")

health = get_health_status()
if not health["ok"]:
    st.error("Ollama 未就绪")
    for msg in health["messages"]:
        st.warning(msg)
    st.info("请启动 Ollama 并执行 `ollama pull nomic-embed-text` 与 `ollama pull qwen2.5:7b`")
    st.stop()

with st.sidebar:
    st.header("设置")
    st.markdown(f"**嵌入**: `{config.EMBED_MODEL}`")
    st.markdown(f"**对话**: `{config.CHAT_MODEL}`")
    st.markdown(f"**混合检索**: `{config.HYBRID_SEARCH}`")
    st.markdown(f"**相似度阈值**: `{config.SIMILARITY_THRESHOLD}`")

    ok_ver, ver_msg = check_index_version()
    if not ok_ver:
        st.warning(ver_msg)

    st.divider()
    st.subheader("知识库文件")
    data_files = sorted(
        p for p in config.DATA_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    for fp in data_files:
        col1, col2 = st.columns([3, 1])
        rel = str(fp.relative_to(config.DATA_DIR))
        col1.text(rel)
        if col2.button("删", key=f"del_{rel}"):
            fp.unlink()
            st.rerun()

    uploaded = st.file_uploader("上传文档", type=["txt", "md", "pdf", "docx", "html"], accept_multiple_files=True)
    if uploaded and st.button("保存上传"):
        for f in uploaded:
            (config.DATA_DIR / f.name).write_bytes(f.getvalue())
        st.success(f"已保存 {len(uploaded)} 个文件")
        st.rerun()

    url = st.text_input("抓取网页 URL（可选）")
    if url and st.button("抓取并保存"):
        try:
            from web_scrape import fetch_url_text

            text = fetch_url_text(url)
            safe_name = url.replace("https://", "").replace("http://", "").replace("/", "_")[:80]
            (config.DATA_DIR / f"{safe_name}.txt").write_text(text, encoding="utf-8")
            st.success("已保存网页正文")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("增量索引"):
            with st.spinner("增量建索引..."):
                try:
                    stats = rag.ingest(rebuild=False, incremental=True)
                    st.success(f"{stats['mode']}: {stats['chunk_count']} 块")
                    _index_ready.clear()
                except Exception as e:
                    st.error(str(e))
    with col_b:
        if st.button("全量重建", type="primary"):
            rag.reset_store()
            with st.spinner("全量建索引..."):
                try:
                    stats = rag.ingest(rebuild=True)
                    st.success(f"{stats['chunk_count']} 块")
                    _index_ready.clear()
                except Exception as e:
                    st.error(str(e))

    if _index_ready():
        store = VectorStore()
        store.load(config.STORAGE_DIR)
        st.info(f"索引块数: {store.size}")
    else:
        st.warning("尚未建立索引")

    st.divider()
    st.subheader("示例问题")
    for q in DEMO_QUESTIONS:
        if st.button(q, key=f"demo_{q}"):
            st.session_state.pending_question = q

    if st.session_state.get("messages") and st.button("导出对话 JSON"):
        export = json.dumps(st.session_state.messages, ensure_ascii=False, indent=2)
        st.download_button(
            "下载",
            export,
            file_name=f"chat_{datetime.now():%Y%m%d_%H%M%S}.json",
            mime="application/json",
        )

if not _index_ready():
    st.info("请上传文档后点击 **增量索引** 或 **全量重建**。")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            _render_sources(msg["sources"])

prompt = st.session_state.pop("pending_question", None) or st.chat_input("输入您的问题...")

if prompt:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer = ""
        sources: list = []
        try:
            stream, sources = rag.query_with_stream(prompt, history=history)
            answer = st.write_stream(stream)
        except Exception as e:
            answer = f"错误：{e}"

        if sources:
            _render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer or "", "sources": sources}
    )
