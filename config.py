"""RAG 系统配置，从环境变量读取，路径相对于项目根目录。"""

import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen2.5:7b")

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "./storage"))

if not DATA_DIR.is_absolute():
    DATA_DIR = _ROOT / DATA_DIR
if not STORAGE_DIR.is_absolute():
    STORAGE_DIR = _ROOT / STORAGE_DIR

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
TOP_K = int(os.getenv("TOP_K", "4"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
HYBRID_SEARCH = os.getenv("HYBRID_SEARCH", "true").lower() in ("1", "true", "yes")
RETRIEVE_N = int(os.getenv("RETRIEVE_N", "12"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
CHAT_HISTORY_TURNS = int(os.getenv("CHAT_HISTORY_TURNS", "3"))


def validate_config() -> None:
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError(f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) 必须小于 CHUNK_SIZE ({CHUNK_SIZE})")
    if TOP_K < 1:
        raise ValueError("TOP_K 必须 >= 1")
    if not 0 <= SIMILARITY_THRESHOLD <= 1:
        raise ValueError("SIMILARITY_THRESHOLD 须在 0~1 之间")


validate_config()

DATA_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
