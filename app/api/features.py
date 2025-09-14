from __future__ import annotations
from fastapi import APIRouter
from ..core.config import settings
from ..data.repos.rag_docs_repo import rag_doc_metrics
from ..services.vision_service import vision_available, vision_unavailable_reason
from ..services.rag_service import rag_available, rag_unavailable_reason
from ..services import cache as app_cache

router = APIRouter()

@router.get("/features")
def features():
    try:
        import multipart  # type: ignore
        mp = True
    except Exception:
        mp = False
    try:
        metrics = rag_doc_metrics()
    except Exception:
        metrics = {"active": 0, "dirty": 0}
    return {
        "model": getattr(settings, "openai_model", None),
        "use_responses": getattr(settings, "openai_use_responses", None),
        "reply_max_tokens": getattr(settings, "reply_max_tokens", None),
        "classifier": {
            "backend": getattr(settings, "classifier_backend", "sbert"),
            "sbert_model": getattr(settings, "sbert_model", None),
        },
        "rag": {"enabled": settings.enable_rag, "available": rag_available, "reason": rag_unavailable_reason},
        "vision": {
            "enabled": settings.enable_vision,
            "available": vision_available,
            "reason": vision_unavailable_reason,
            "multipart": mp,
            "backend": getattr(settings, "vision_backend", None),
        },
        "tools": {
            "enabled": True,
            "max_calls": 3,
            "tool_timeout_s": 8,
            "total_budget_s": 20,
        },
        "admin": {
            "caches": {
                "classes": len(app_cache.classes_cache()),
                "vision_labels": len(app_cache.vision_labels_cache()),
            },
            "rag_docs": metrics,
        },
    }
