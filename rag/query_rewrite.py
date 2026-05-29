"""查询改写与 HyDE（假设性文档嵌入）。"""

import ollama

from . import config
from .logger import get_logger

log = get_logger()


def _client():
    return ollama.Client(host=config.OLLAMA_HOST)


def hyde_expand(question: str) -> str:
    """
    生成假设性回答段落，用于改善检索召回（HyDE）。
  失败时回退为原问题。
    """
    prompt = (
        "请用中文写一段 80~150 字的假设性参考资料，内容应能回答下列问题。"
        "只输出正文，不要标题或解释。\n\n问题："
        f"{question}"
    )
    try:
        resp = _client().chat(
            model=config.CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        passage = (resp.get("message") or {}).get("content", "").strip()
        if passage:
            log.info("HyDE 扩展查询成功 (%d 字)", len(passage))
            return f"{question}\n{passage}"
    except Exception as e:
        log.warning("HyDE 失败，使用原问题: %s", e)
    return question


def rewrite_query(question: str) -> str:
    """将口语化问题改写为更适合检索的查询。"""
    prompt = (
        "将下列用户问题改写为简洁的检索查询（保留关键实体与术语，"
        "一行中文，不要解释）：\n\n"
        f"{question}"
    )
    try:
        resp = _client().chat(
            model=config.CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        rewritten = (resp.get("message") or {}).get("content", "").strip()
        if rewritten:
            log.info("查询改写: %s -> %s", question[:40], rewritten[:40])
            return rewritten
    except Exception as e:
        log.warning("查询改写失败: %s", e)
    return question


def build_search_query(
    question: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    """组合历史、HyDE、改写，得到最终检索用查询。"""
    parts = [question]
    if history:
        recent = [
            m["content"] for m in history if m.get("role") == "user"
        ][-config.CHAT_HISTORY_TURNS :]
        if recent:
            parts.insert(0, " ".join(recent))

    base = " ".join(parts)

    if config.USE_HYDE:
        base = hyde_expand(base)

    if config.USE_QUERY_REWRITE:
        base = rewrite_query(base)

    return base
