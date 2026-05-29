"""Ollama 服务与模型可用性检查。"""

from typing import Any

import ollama

from . import config


def check_ollama_running() -> tuple[bool, str]:
    try:
        client = ollama.Client(host=config.OLLAMA_HOST)
        client.list()
        return True, "Ollama 服务正常"
    except Exception as e:
        return False, (
            f"无法连接 Ollama（{config.OLLAMA_HOST}）。"
            f"请确认已启动 Ollama。错误: {e}"
        )


def _model_names(models_response: Any) -> set[str]:
    names: set[str] = set()
    for m in models_response.get("models", []):
        name = m.get("name") or m.get("model") or ""
        if name:
            names.add(name)
            names.add(name.split(":")[0])
    return names


def check_model_pulled(model: str) -> tuple[bool, str]:
    try:
        client = ollama.Client(host=config.OLLAMA_HOST)
        resp = client.list()
        names = _model_names(resp)
        base = model.split(":")[0]
        if model in names or base in names or any(n.startswith(base) for n in names):
            return True, f"模型 {model} 已就绪"
        return False, f"未找到模型 {model}，请执行: ollama pull {model}"
    except Exception as e:
        return False, f"检查模型失败: {e}"


def get_health_status() -> dict[str, Any]:
    """返回整体健康状态，供 CLI / Streamlit 使用。"""
    ok, msg = check_ollama_running()
    status: dict[str, Any] = {
        "ok": ok,
        "ollama": msg,
        "embed_model": None,
        "chat_model": None,
        "messages": [msg],
    }
    if not ok:
        status["ok"] = False
        return status

    embed_ok, embed_msg = check_model_pulled(config.EMBED_MODEL)
    chat_ok, chat_msg = check_model_pulled(config.CHAT_MODEL)
    status["embed_model"] = {"ok": embed_ok, "message": embed_msg}
    status["chat_model"] = {"ok": chat_ok, "message": chat_msg}
    status["messages"].extend([embed_msg, chat_msg])
    status["ok"] = embed_ok and chat_ok

    if config.USE_MULTIMODAL:
        vision_ok, vision_msg = check_model_pulled(config.VISION_MODEL)
        status["vision_model"] = {"ok": vision_ok, "message": vision_msg}
        status["messages"].append(vision_msg)
        status["ok"] = status["ok"] and vision_ok

    return status
