from __future__ import annotations
from fastapi import APIRouter, Request
from ...services import cache as app_cache
from ...models.schemas import AdminClassIn, AdminClassOut
from ._util import require_admin
from ...data.repos import classes_repo as repo
from ...data.bootstrap import update_default_stems

router = APIRouter()

@router.post("/update_stems")
def admin_update_stems(req: Request):
    require_admin(req)
    updated = update_default_stems()
    app_cache.reload_caches()
    return {"updated_rows": updated}

@router.get("/classes", response_model=list[AdminClassOut])
def admin_list_classes(req: Request):
    require_admin(req)
    return repo.fetch_classes(active_only=False)

@router.post("/classes", response_model=AdminClassOut)
def admin_create_class(req: Request, body: AdminClassIn):
    require_admin(req)
    new_id = repo.insert_class(body.name, body.synonyms, body.stems, body.system_prompt, body.priority, body.active)
    app_cache.reload_caches()
    rows = repo.fetch_classes(active_only=False)
    return next(r for r in rows if r["id"] == new_id)

@router.put("/classes/{cls_id}", response_model=AdminClassOut)
def admin_update_class(req: Request, cls_id: int, body: AdminClassIn):
    require_admin(req)
    repo.update_class(
        cls_id,
        name=body.name,
        synonyms=body.synonyms,
        stems=body.stems,
        system_prompt=body.system_prompt,
        priority=body.priority,
        active=body.active,
    )
    app_cache.reload_caches()
    rows = repo.fetch_classes(active_only=False)
    return next(r for r in rows if r["id"] == int(cls_id))

@router.delete("/classes/{cls_id}")
def admin_delete_class(req: Request, cls_id: int):
    require_admin(req)
    changed = repo.soft_delete_class(cls_id)
    app_cache.reload_caches()
    return {"deleted": changed}

