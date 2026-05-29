"""RAG 系统配置，从环境变量读取，路径相对于项目根目录。"""

import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen2.5:7b")

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "./storage"))

if not DATA_DIR.is_absolute():
    DATA_DIR = _PROJECT_ROOT / DATA_DIR
if not STORAGE_DIR.is_absolute():
    STORAGE_DIR = _PROJECT_ROOT / STORAGE_DIR

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
TOP_K = int(os.getenv("TOP_K", "4"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
HYBRID_SEARCH = os.getenv("HYBRID_SEARCH", "true").lower() in ("1", "true", "yes")
RETRIEVE_N = int(os.getenv("RETRIEVE_N", "12"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
EMBED_WORKERS = int(os.getenv("EMBED_WORKERS", "4"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
CHAT_HISTORY_TURNS = int(os.getenv("CHAT_HISTORY_TURNS", "3"))

USE_HYDE = os.getenv("USE_HYDE", "false").lower() in ("1", "true", "yes")
USE_QUERY_REWRITE = os.getenv("USE_QUERY_REWRITE", "false").lower() in ("1", "true", "yes")
USE_AGENTIC_RAG = os.getenv("USE_AGENTIC_RAG", "true").lower() in ("1", "true", "yes")
AGENTIC_MIN_SCORE = float(os.getenv("AGENTIC_MIN_SCORE", "0.35"))
USE_GRAPH_RAG = os.getenv("USE_GRAPH_RAG", "true").lower() in ("1", "true", "yes")

OCR_ENABLED = os.getenv("OCR_ENABLED", "true").lower() in ("1", "true", "yes")
OCR_MIN_TEXT_LEN = int(os.getenv("OCR_MIN_TEXT_LEN", "50"))
UI_LANG = os.getenv("UI_LANG", "zh")

USE_MULTIMODAL = os.getenv("USE_MULTIMODAL", "false").lower() in ("1", "true", "yes")
VISION_MODEL = os.getenv("VISION_MODEL", "llava")
MULTIMODAL_MAX_IMAGES = int(os.getenv("MULTIMODAL_MAX_IMAGES", "10"))

USE_VISION_CACHE = os.getenv("USE_VISION_CACHE", "true").lower() in ("1", "true", "yes")
USE_CLIP = os.getenv("USE_CLIP", "false").lower() in ("1", "true", "yes")
CLIP_MODEL = os.getenv("CLIP_MODEL", "clip-ViT-B-32")

VIDEO_FRAME_INTERVAL_SEC = int(os.getenv("VIDEO_FRAME_INTERVAL_SEC", "5"))
VIDEO_MAX_FRAMES = int(os.getenv("VIDEO_MAX_FRAMES", "20"))

# 生产化：多租户隔离、API 鉴权
TENANT_ID = os.getenv("TENANT_ID", "").strip()
API_KEY = os.getenv("API_KEY", "").strip()
METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() in ("1", "true", "yes")

# 向量库后端: local | qdrant | milvus
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "local")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_chunks")
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_chunks")

# 鉴权: none | api_key | jwt | both
AUTH_MODE = os.getenv("AUTH_MODE", "api_key")
JWT_SECRET = os.getenv("JWT_SECRET", "").strip()
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")

PROMETHEUS_ENABLED = os.getenv("PROMETHEUS_ENABLED", "true").lower() in ("1", "true", "yes")


def validate_config() -> None:
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError(f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) 必须小于 CHUNK_SIZE ({CHUNK_SIZE})")
    if TOP_K < 1:
        raise ValueError("TOP_K 必须 >= 1")
    if not 0 <= SIMILARITY_THRESHOLD <= 1:
        raise ValueError("SIMILARITY_THRESHOLD 须在 0~1 之间")
    if EMBED_WORKERS < 1:
        raise ValueError("EMBED_WORKERS 必须 >= 1")
    if MULTIMODAL_MAX_IMAGES < 1:
        raise ValueError("MULTIMODAL_MAX_IMAGES 必须 >= 1")
    if VIDEO_MAX_FRAMES < 1:
        raise ValueError("VIDEO_MAX_FRAMES 必须 >= 1")
    if VIDEO_FRAME_INTERVAL_SEC < 1:
        raise ValueError("VIDEO_FRAME_INTERVAL_SEC 必须 >= 1")
    if VECTOR_BACKEND not in ("local", "qdrant", "milvus"):
        raise ValueError("VECTOR_BACKEND 须为 local | qdrant | milvus")
    if AUTH_MODE not in ("none", "api_key", "jwt", "both"):
        raise ValueError("AUTH_MODE 须为 none | api_key | jwt | both")


validate_config()

if TENANT_ID:
    DATA_DIR = DATA_DIR / TENANT_ID
    STORAGE_DIR = STORAGE_DIR / TENANT_ID

DATA_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
