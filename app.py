"""Streamlit RAG 问答界面（中/英 i18n）。"""

import json
from datetime import datetime

import streamlit as st

import config
import rag
from i18n import DEMO_QUESTIONS, t
from index_manifest import check_index_version
from loader import SUPPORTED_EXTENSIONS
from ollama_health import get_health_status
from vector_store import VectorStore

if "ui_lang" not in st.session_state:
    st.session_state.ui_lang = config.UI_LANG

lang = st.session_state.ui_lang

st.set_page_config(page_title=t("page_title", lang), page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def _index_ready() -> bool:
    return rag.has_index()


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(t("citations", lang, n=len(sources))):
        st.bar_chart(
            {f"{s['source']} #{s['chunk_index']}": s["score"] for s in sources},
            horizontal=True,
        )
        for i, src in enumerate(sources, start=1):
            st.markdown(
                f"**[{i}]** `{src['source']}` "
                f"(score {src['score']:.3f}, #{src['chunk_index']})"
            )
            st.text(src["excerpt"])


st.title(t("title", lang))
st.caption(t("caption", lang))

health = get_health_status()
if not health["ok"]:
    st.error(t("ollama_error", lang))
    for msg in health["messages"]:
        st.warning(msg)
    st.info(t("ollama_hint", lang))
    st.stop()

with st.sidebar:
    st.selectbox(
        t("lang_label", lang),
        options=["zh", "en"],
        format_func=lambda x: "中文" if x == "zh" else "English",
        key="ui_lang",
    )
    lang = st.session_state.ui_lang

    st.header(t("settings", lang))
    st.markdown(f"**{t('embed', lang)}**: `{config.EMBED_MODEL}`")
    st.markdown(f"**{t('chat', lang)}**: `{config.CHAT_MODEL}`")
    st.markdown(f"**{t('hybrid', lang)}**: `{config.HYBRID_SEARCH}`")
    st.markdown(f"**{t('threshold', lang)}**: `{config.SIMILARITY_THRESHOLD}`")
    st.caption(
        f"HyDE={config.USE_HYDE} | Agentic={config.USE_AGENTIC_RAG} | "
        f"Graph={config.USE_GRAPH_RAG} | Multimodal={config.USE_MULTIMODAL}"
    )
    if config.USE_MULTIMODAL:
        st.caption(f"Vision: `{config.VISION_MODEL}` (max {config.MULTIMODAL_MAX_IMAGES} imgs/doc)")

    ok_ver, ver_msg = check_index_version()
    if not ok_ver:
        st.warning(ver_msg)

    st.divider()
    st.subheader(t("kb_files", lang))
    data_files = sorted(
        p for p in config.DATA_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    for fp in data_files:
        col1, col2 = st.columns([3, 1])
        rel = str(fp.relative_to(config.DATA_DIR))
        col1.text(rel)
        if col2.button(t("delete", lang), key=f"del_{rel}"):
            fp.unlink()
            st.rerun()

    uploaded = st.file_uploader(
        t("upload", lang),
        type=["txt", "md", "pdf", "docx", "html", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )
    if uploaded and st.button(t("save_upload", lang)):
        for f in uploaded:
            (config.DATA_DIR / f.name).write_bytes(f.getvalue())
        st.success(t("upload_ok", lang, n=len(uploaded)))
        st.rerun()

    url = st.text_input(t("url_label", lang))
    if url and st.button(t("url_fetch", lang)):
        try:
            from web_scrape import fetch_url_text

            text = fetch_url_text(url)
            safe_name = url.replace("https://", "").replace("http://", "").replace("/", "_")[:80]
            (config.DATA_DIR / f"{safe_name}.txt").write_text(text, encoding="utf-8")
            st.success(t("url_ok", lang))
            st.rerun()
        except Exception as e:
            st.error(str(e))

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button(t("incr_index", lang)):
            with st.spinner("..."):
                try:
                    stats = rag.ingest(rebuild=False, incremental=True)
                    st.success(f"{stats['mode']}: {stats['chunk_count']}")
                    _index_ready.clear()
                except Exception as e:
                    st.error(str(e))
    with col_b:
        if st.button(t("full_rebuild", lang), type="primary"):
            rag.reset_store()
            with st.spinner("..."):
                try:
                    stats = rag.ingest(rebuild=True)
                    st.success(str(stats["chunk_count"]))
                    _index_ready.clear()
                except Exception as e:
                    st.error(str(e))

    if _index_ready():
        store = VectorStore()
        store.load(config.STORAGE_DIR)
        st.info(t("index_blocks", lang, n=store.size))
    else:
        st.warning(t("no_index", lang))

    st.divider()
    st.subheader(t("demo_q", lang))
    for q in DEMO_QUESTIONS.get(lang, DEMO_QUESTIONS["zh"]):
        if st.button(q, key=f"demo_{lang}_{q}"):
            st.session_state.pending_question = q

    if st.session_state.get("messages") and st.button(t("export_chat", lang)):
        export = json.dumps(st.session_state.messages, ensure_ascii=False, indent=2)
        st.download_button(
            t("download", lang),
            export,
            file_name=f"chat_{datetime.now():%Y%m%d_%H%M%S}.json",
            mime="application/json",
        )

if not _index_ready():
    st.info(t("no_index_hint", lang))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            _render_sources(msg["sources"])

prompt = st.session_state.pop("pending_question", None) or st.chat_input(t("chat_input", lang))

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
            answer = f"Error: {e}"

        if sources:
            _render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer or "", "sources": sources}
    )
