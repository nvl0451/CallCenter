from __future__ import annotations
from fastapi import FastAPI
from ..core.config import settings
from ..data import db
from ..data.bootstrap import bootstrap_defaults, ensure_default_system_base, load_active_inline_docs, mark_inline_indexed
from ..data.repos.rag_docs_repo import ensure_storage_dirs
from ..services import cache as app_cache
from ..services.llm import llm_client  # noqa: F401  # ensure client constructed

from ..api.health import router as health_router
from ..api.features import router as features_router
from ..api.dialog import router as dialog_router
from ..api.rag import router as rag_router
from ..api.vision import router as vision_router
from ..api.admin import router as admin_router
from ..api.agent import router as agent_router


def create_app() -> FastAPI:
    app = FastAPI(title="CallCenter LLM + RAG + Vision")

    @app.on_event("startup")
    def _startup():
        try:
            db.run_migrations()
        except Exception:
            pass
        try:
            ensure_default_system_base()
        except Exception:
            pass
        try:
            inserted = bootstrap_defaults()
        except Exception:
            inserted = None
        try:
            ensure_storage_dirs()
        except Exception:
            pass
        try:
            app_cache.reload_caches()
        except Exception:
            pass
        try:
            from ..services.classifier_service import _sbert_classify as _probe  # type: ignore
            _ = _probe  # suppress unused
        except Exception:
            pass
        try:
            if settings.enable_rag and inserted and isinstance(inserted, dict) and inserted.get("rag_docs", 0) > 0:
                from ..services.rag_service import ingest_texts
                texts = load_active_inline_docs()
                if texts:
                    ingest_texts(texts, metadatas=[{"source": "bootstrap"} for _ in texts])
                    try:
                        mark_inline_indexed(getattr(settings, "openai_embed_model", ""))
                    except Exception:
                        pass
        except Exception:
            pass

    app.include_router(health_router)
    app.include_router(features_router)
    app.include_router(dialog_router)
    app.include_router(rag_router)
    app.include_router(vision_router)
    app.include_router(admin_router)
    app.include_router(agent_router)
    return app
