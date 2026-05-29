"""Streamlit 界面文案（中/英）。"""

from typing import Any

STRINGS: dict[str, dict[str, str]] = {
    "zh": {
        "page_title": "RAG 问答",
        "title": "检索增强生成（RAG）演示",
        "caption": "基于本地 Ollama + 混合检索",
        "ollama_error": "Ollama 未就绪",
        "ollama_hint": "请启动 Ollama 并执行 `ollama pull nomic-embed-text` 与 `ollama pull qwen2.5:7b`",
        "settings": "设置",
        "embed": "嵌入",
        "chat": "对话",
        "hybrid": "混合检索",
        "threshold": "相似度阈值",
        "kb_files": "知识库文件",
        "upload": "上传文档",
        "save_upload": "保存上传",
        "upload_ok": "已保存 {n} 个文件",
        "url_label": "抓取网页 URL（可选）",
        "url_fetch": "抓取并保存",
        "url_ok": "已保存网页正文",
        "incr_index": "增量索引",
        "full_rebuild": "全量重建",
        "index_blocks": "索引块数: {n}",
        "no_index": "尚未建立索引",
        "demo_q": "示例问题",
        "export_chat": "导出对话 JSON",
        "download": "下载",
        "no_index_hint": "请上传文档后点击 **增量索引** 或 **全量重建**。",
        "citations": "引用片段（{n} 条）",
        "chat_input": "输入您的问题...",
        "lang_label": "界面语言",
        "delete": "删",
    },
    "en": {
        "page_title": "RAG Chat",
        "title": "Retrieval-Augmented Generation (RAG) Demo",
        "caption": "Local Ollama + hybrid retrieval",
        "ollama_error": "Ollama is not ready",
        "ollama_hint": "Start Ollama and run `ollama pull nomic-embed-text` and `ollama pull qwen2.5:7b`",
        "settings": "Settings",
        "embed": "Embedding",
        "chat": "Chat",
        "hybrid": "Hybrid search",
        "threshold": "Similarity threshold",
        "kb_files": "Knowledge base files",
        "upload": "Upload documents",
        "save_upload": "Save uploads",
        "upload_ok": "Saved {n} file(s)",
        "url_label": "Fetch URL (optional)",
        "url_fetch": "Fetch and save",
        "url_ok": "Page text saved",
        "incr_index": "Incremental index",
        "full_rebuild": "Full rebuild",
        "index_blocks": "Indexed chunks: {n}",
        "no_index": "No index yet",
        "demo_q": "Sample questions",
        "export_chat": "Export chat JSON",
        "download": "Download",
        "no_index_hint": "Upload documents, then click **Incremental index** or **Full rebuild**.",
        "citations": "Citations ({n})",
        "chat_input": "Ask a question...",
        "lang_label": "UI language",
        "delete": "Del",
    },
}

DEMO_QUESTIONS: dict[str, list[str]] = {
    "zh": [
        "RAG 流水线包含哪些步骤？",
        "默认的嵌入模型是什么？",
        "向量存储使用什么技术？",
    ],
    "en": [
        "What steps does the RAG pipeline include?",
        "What is the default embedding model?",
        "What technology is used for vector storage?",
    ],
}


def t(key: str, lang: str = "zh", **kwargs: Any) -> str:
    text = STRINGS.get(lang, STRINGS["zh"]).get(key, key)
    if kwargs:
        return text.format(**kwargs)
    return text
