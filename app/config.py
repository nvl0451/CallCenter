from __future__ import annotations
import os
from pydantic import BaseModel

class Settings(BaseModel):
    # OpenAI settings
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    openai_fallback_model: str = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o-mini")
    openai_use_responses: bool = bool(int(os.getenv("OPENAI_USE_RESPONSES", "1")))

    sqlite_path: str = os.getenv("SQLITE_PATH", "./callcenter.sqlite3")
    chroma_dir: str = os.getenv("CHROMA_DIR", "./chroma_db")

    # Token/decoding controls
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    # Backwards-compat: REPLY_MAX_TOKENS overrides LLM_MAX_TOKENS
    reply_max_tokens: int = int(os.getenv("REPLY_MAX_TOKENS", os.getenv("LLM_MAX_TOKENS", "256")))
    classify_max_tokens: int = int(os.getenv("CLASSIFY_MAX_TOKENS", "16"))

    # Feature flags
    enable_rag: bool = bool(int(os.getenv("ENABLE_RAG", "0")))
    enable_vision: bool = bool(int(os.getenv("ENABLE_VISION", "0")))

settings = Settings()
