from __future__ import annotations
from fastapi import APIRouter, Request
from ...services import cache as app_cache
from ._util import require_admin
from ...data.bootstrap import bootstrap_defaults, ensure_default_system_base

router = APIRouter()

@router.post("/bootstrap")
def admin_bootstrap(req: Request):
    require_admin(req)
    ensure_default_system_base()
    ins = bootstrap_defaults()
    app_cache.reload_caches()
    return {"inserted": ins, "caches": {"classes": len(app_cache.classes_cache()), "vision_labels": len(app_cache.vision_labels_cache())}}

@router.get("/settings")
def admin_get_settings(req: Request):
    require_admin(req)
    return {"system_base": app_cache.system_base()}

@router.put("/settings/system_base")
async def admin_put_system_base(req: Request, payload: dict):
    require_admin(req)
    from ...data.repos.settings_repo import set_setting
    text = str(payload.get("system_base", ""))
    if not text.strip():
        from fastapi import HTTPException
        raise HTTPException(400, "system_base required")
    set_setting("system_base", text)
    app_cache.reload_caches()
    return {"ok": True, "system_base": app_cache.system_base()}

