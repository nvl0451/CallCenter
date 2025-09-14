from fastapi import APIRouter
from .classes import router as classes_router
from .vision import router as vision_router
from .rag_docs import router as rag_router
from .settings import router as settings_router

router = APIRouter(prefix="/admin", tags=["admin"])
router.include_router(classes_router)
router.include_router(vision_router)
router.include_router(rag_router)
router.include_router(settings_router)

