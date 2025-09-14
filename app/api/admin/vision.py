from __future__ import annotations
from fastapi import APIRouter, Request
from ...services import cache as app_cache
from ...models.schemas import AdminVisionIn, AdminVisionOut
from ._util import require_admin
from ...data.repos import vision_labels_repo as repo

router = APIRouter()

@router.get("/vision", response_model=list[AdminVisionOut])
def admin_list_vision(req: Request):
    require_admin(req)
    return repo.fetch_vision_labels(active_only=False)

@router.post("/vision", response_model=AdminVisionOut)
def admin_create_vision(req: Request, body: AdminVisionIn):
    require_admin(req)
    new_id = repo.insert_vision_label(body.name, body.synonyms, body.templates, body.priority, body.active)
    app_cache.reload_caches()
    rows = repo.fetch_vision_labels(active_only=False)
    return next(r for r in rows if r["id"] == new_id)

@router.put("/vision/{lbl_id}", response_model=AdminVisionOut)
def admin_update_vision(req: Request, lbl_id: int, body: AdminVisionIn):
    require_admin(req)
    repo.update_vision_label(lbl_id, name=body.name, synonyms=body.synonyms, templates=body.templates, priority=body.priority, active=body.active)
    app_cache.reload_caches()
    rows = repo.fetch_vision_labels(active_only=False)
    return next(r for r in rows if r["id"] == int(lbl_id))

@router.delete("/vision/{lbl_id}")
def admin_delete_vision(req: Request, lbl_id: int):
    require_admin(req)
    changed = repo.soft_delete_vision_label(lbl_id)
    app_cache.reload_caches()
    return {"deleted": changed}

