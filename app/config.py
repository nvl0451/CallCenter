from __future__ import annotations
import os
from pydantic import BaseModel

class Settings(BaseModel):
    # OpenAI settings
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    openai_use_responses: bool = bool(int(os.getenv("OPENAI_USE_RESPONSES", "1")))
    openai_embed_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    sqlite_path: str = os.getenv("SQLITE_PATH", "./callcenter.sqlite3")
    chroma_dir: str = os.getenv("CHROMA_DIR", "./chroma_db")

    # Token/decoding controls
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    # Backwards-compat: REPLY_MAX_TOKENS overrides LLM_MAX_TOKENS
    reply_max_tokens: int = int(os.getenv("REPLY_MAX_TOKENS", os.getenv("LLM_MAX_TOKENS", "256")))
    classify_max_tokens: int = int(os.getenv("CLASSIFY_MAX_TOKENS", "16"))
    classify_strict: bool = bool(int(os.getenv("CLASSIFY_STRICT", "1")))
    classify_model: str = os.getenv("CLASSIFY_MODEL", "gpt-4o-mini")

    # Feature flags
    enable_rag: bool = bool(int(os.getenv("ENABLE_RAG", "0")))
    enable_vision: bool = bool(int(os.getenv("ENABLE_VISION", "0")))

    # Vision
    openai_vision_model: str = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
    vision_backend: str = os.getenv("VISION_BACKEND", "openai")  # openai | clip
    vision_allow_insecure_download: bool = bool(int(os.getenv("VISION_ALLOW_INSECURE_DOWNLOAD", "0")))

    # RAG chunking/embeddings
    rag_use_openai_embeddings: bool = bool(int(os.getenv("RAG_USE_OPENAI_EMBEDDINGS", "1")))
    rag_chunk_chars: int = int(os.getenv("RAG_CHUNK_CHARS", "800"))
    rag_chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))

    # Admin / storage / testing
    admin_token: str = os.getenv("ADMIN_TOKEN", "")
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "8"))
    rag_storage_dir: str = os.getenv("RAG_STORAGE_DIR", "./storage/rag")
    rag_hard_delete: bool = bool(int(os.getenv("RAG_HARD_DELETE", "0")))
    openai_mock: bool = bool(int(os.getenv("OPENAI_MOCK", "0")))

settings = Settings()
