"""Streamlit RAG 问答界面。"""

import streamlit as st

import config
import rag
from vector_store import VectorStore

st.set_page_config(page_title="RAG 问答", page_icon="📚", layout="wide")

st.title("检索增强生成（RAG）演示")
st.caption("基于本地 Ollama + 向量检索")


@st.cache_resource
def _index_ready() -> bool:
    return rag.has_index()


def _rebuild_index() -> dict:
    rag.reset_store()
    return rag.ingest(rebuild=True)


with st.sidebar:
    st.header("设置")
    st.markdown(f"**嵌入模型**: `{config.EMBED_MODEL}`")
    st.markdown(f"**对话模型**: `{config.CHAT_MODEL}`")
    st.markdown(f"**Ollama**: `{config.OLLAMA_HOST}`")
    st.markdown(f"**数据目录**: `{config.DATA_DIR}`")
    st.markdown(f"**索引目录**: `{config.STORAGE_DIR}`")

    st.divider()

    uploaded = st.file_uploader(
        "上传文档到 data/",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("保存上传文件"):
        for f in uploaded:
            dest = config.DATA_DIR / f.name
            dest.write_bytes(f.getvalue())
        st.success(f"已保存 {len(uploaded)} 个文件到 data/")
        st.rerun()

    if st.button("重建索引", type="primary"):
        with st.spinner("正在建索引（加载 → 分块 → 嵌入）..."):
            try:
                stats = _rebuild_index()
                st.success(
                    f"完成：{stats['doc_count']} 篇文档，"
                    f"{stats['chunk_count']} 个块，维度 {stats['dim']}"
                )
                _index_ready.clear()
            except Exception as e:
                st.error(str(e))

    if _index_ready():
        store = VectorStore()
        store.load(config.STORAGE_DIR)
        st.info(f"当前索引：{store.size} 个向量块")
    else:
        st.warning("尚未建立索引，请上传文档后点击「重建索引」")

if not _index_ready():
    st.info("请先向 `data/` 放入文档（或使用侧栏上传），然后点击 **重建索引**。")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"引用片段（{len(msg['sources'])} 条）"):
                for i, src in enumerate(msg["sources"], start=1):
                    st.markdown(
                        f"**[{i}]** `{src['source']}` "
                        f"(相似度 {src['score']:.3f}, 块 #{src['chunk_index']})"
                    )
                    st.text(src["excerpt"])

if prompt := st.chat_input("输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("检索并生成中..."):
            try:
                result = rag.query(prompt)
                answer = result["answer"]
                sources = result["sources"]
                st.write(f"检索到 {len(sources)} 条相关片段")
            except Exception as e:
                answer = f"错误：{e}"
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander(f"引用片段（{len(sources)} 条）"):
                for i, src in enumerate(sources, start=1):
                    st.markdown(
                        f"**[{i}]** `{src['source']}` "
                        f"(相似度 {src['score']:.3f}, 块 #{src['chunk_index']})"
                    )
                    st.text(src["excerpt"])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
