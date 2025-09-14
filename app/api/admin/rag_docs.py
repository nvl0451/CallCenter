from __future__ import annotations
import os, time as _time, re as _re, hashlib
from fastapi import APIRouter, Request, HTTPException
from ...core.config import settings
from ...services import cache as app_cache
from ...models.schemas import AdminRagInlineIn, AdminRagDocOut
from ._util import require_admin
from ...data.repos import rag_docs_repo as repo
from ...data.bootstrap import sync_seed_kb_from_files
from ...services import rag_service as rag_mod

router = APIRouter()

@router.get("/rag/docs", response_model=list[AdminRagDocOut])
def admin_list_rag_docs(req: Request):
    require_admin(req)
    return repo.list_rag_documents(active_only=False)

@router.post("/rag/inline", response_model=AdminRagDocOut)
def admin_create_rag_inline(req: Request, body: AdminRagInlineIn):
    require_admin(req)
    doc_id = repo.insert_rag_inline(body.title, body.content_text, source="admin-inline")
    if settings.enable_rag and rag_mod.rag_available and body.index_now:
        try:
            meta = {"source": "admin-inline", "doc_id": str(doc_id)}
            rag_mod.ingest_texts([body.content_text], metadatas=[meta])
            repo.update_rag_doc(doc_id, indexed_at=_time.time(), embed_model=rag_mod.get_embed_note(), chunks_count=rag_mod.chunk_count(body.content_text), dirty=0)
        except Exception:
            pass
    row = repo.get_rag_doc(doc_id)
    return row  # type: ignore

@router.put("/rag/docs/{doc_id}", response_model=AdminRagDocOut)
def admin_update_rag_inline(req: Request, doc_id: int, payload: dict):
    require_admin(req)
    row = repo.get_rag_doc(doc_id)
    if not row:
        raise HTTPException(404, "not found")
    if row.get("kind") != "inline":
        raise HTTPException(400, "only inline docs can be updated via this endpoint")
    content = str(payload.get("content_text", ""))
    if not content:
        raise HTTPException(400, "content_text required")
    repo.update_rag_doc(doc_id, content_text=content, sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(), bytes=len(content.encode("utf-8")), dirty=1)
    return repo.get_rag_doc(doc_id)  # type: ignore

try:
    from fastapi import UploadFile, File, Form
    _mp_ok = True
except Exception:
    _mp_ok = False

def _slugify(name: str) -> str:
    s = name.lower().strip()
    s = _re.sub(r"[^a-z0-9а-яё_.-]+", "-", s)
    s = _re.sub(r"-+", "-", s).strip("-")
    return s or "doc"

if _mp_ok:
    @router.post("/rag/file", response_model=AdminRagDocOut)
    async def admin_upload_rag_file(req: Request, file: UploadFile = File(...), index_now: int = Form(1)):
        require_admin(req)
        fname = file.filename or "upload.txt"
        ext = (fname.rsplit(".", 1)[-1] or "").lower()
        if ext not in ("txt", "md"):
            raise HTTPException(400, "Only .txt and .md are allowed")
        raw = await file.read()
        if len(raw) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"File too large, limit {settings.max_upload_mb} MB")
        try:
            text = raw.decode("utf-8", errors="ignore")
        except Exception:
            text = ""
        sha = hashlib.sha256(raw).hexdigest()
        mime = "text/markdown" if ext == "md" else "text/plain"
        new_id = repo.insert_rag_file(title=fname, rel_path="", bytes_len=len(raw), sha256=sha, mime=mime, source="admin-upload")
        repo.ensure_storage_dirs()
        slug = _slugify(fname)
        rel_name = f"{new_id}_{slug}"
        abs_dir = settings.rag_storage_dir
        abs_path = os.path.join(abs_dir, rel_name)
        with open(abs_path, "wb") as f:
            f.write(raw)
        repo.update_rag_doc(new_id, rel_path=rel_name)
        if settings.enable_rag and rag_mod.rag_available and int(index_now) == 1:
            try:
                meta = {"source": "admin-upload", "doc_id": str(new_id), "filename": fname}
                rag_mod.ingest_texts([text], metadatas=[meta])
                repo.update_rag_doc(new_id, indexed_at=_time.time(), embed_model=rag_mod.get_embed_note(), chunks_count=rag_mod.chunk_count(text), dirty=0)
            except Exception:
                pass
        row = repo.get_rag_doc(new_id)
        return row  # type: ignore

@router.delete("/rag/docs/{doc_id}")
def admin_delete_rag_doc(req: Request, doc_id: int):
    require_admin(req)
    row = repo.get_rag_doc(doc_id)
    if not row:
        raise HTTPException(404, "not found")
    try:
        if settings.enable_rag and rag_mod.rag_available:
            rag_mod.delete_by_doc_id(doc_id)
    except Exception:
        pass
    changed = repo.soft_delete_rag_doc(doc_id)
    if settings.rag_hard_delete and row.get("kind") == "file" and row.get("rel_path"):
        try:
            os.remove(os.path.join(settings.rag_storage_dir, row["rel_path"]))
        except Exception:
            pass
    return {"deleted": changed}

@router.post("/rag/reindex/{doc_id}")
def admin_reindex_one(req: Request, doc_id: int):
    require_admin(req)
    row = repo.get_rag_doc(doc_id)
    if not row or int(row.get("active", 1)) != 1:
        raise HTTPException(404, "doc not found or inactive")
    return _reindex_doc(row)

def _reindex_doc(doc: dict) -> dict:
    if not (settings.enable_rag and rag_mod.rag_available):
        raise HTTPException(503, "RAG unavailable")
    import time as _t
    t0 = _t.perf_counter()
    content = None
    if doc.get("kind") == "inline":
        content = doc.get("content_text") or ""
    else:
        rel = doc.get("rel_path") or ""
        try:
            with open(os.path.join(settings.rag_storage_dir, rel), "rb") as f:
                raw = f.read()
            content = raw.decode("utf-8", errors="ignore")
        except Exception as e:
            raise HTTPException(500, f"failed to read file content: {e}")
    try:
        rag_mod.delete_by_doc_id(doc.get("id"))
    except Exception:
        pass
    chunks = 0
    try:
        meta = {"source": doc.get("source") or "admin", "doc_id": str(doc.get("id"))}
        rag_mod.ingest_texts([content], metadatas=[meta])
        chunks = rag_mod.chunk_count(content)
    except Exception as e:
        raise HTTPException(500, f"embed/ingest failed: {e}")
    try:
        repo.update_rag_doc(int(doc.get("id")), indexed_at=_time.time(), embed_model=rag_mod.get_embed_note(), chunks_count=chunks, dirty=0)
    except Exception:
        pass
    ms = int((_t.perf_counter() - t0) * 1000)
    return {"id": doc.get("id"), "chunks": chunks, "ms": ms, "ok": True}

@router.post("/rag/reindex_dirty")
def admin_reindex_dirty(req: Request, limit: int | None = None):
    require_admin(req)
    rows = repo.list_dirty_docs(limit=limit)
    out = []; errs = 0
    for r in rows:
        try:
            out.append(_reindex_doc(r))
        except Exception as e:
            errs += 1
            out.append({"id": r.get("id"), "ok": False, "error": str(e)})
    return {"processed": len(rows), "errors": errs, "results": out}

@router.post("/rag/mark_all_dirty")
def admin_mark_all_dirty(req: Request):
    require_admin(req)
    return {"marked": repo.mark_all_dirty()}

@router.post("/rag/sync_seed_kb")
def admin_sync_seed_kb(req: Request):
    require_admin(req)
    res = sync_seed_kb_from_files()
    return res

@router.post("/rag/reset_index")
def admin_reset_index(req: Request):
    require_admin(req)
    if not (settings.enable_rag and rag_mod.rag_available):
        raise HTTPException(503, "RAG unavailable")
    rag_mod.reset_index(); return {"ok": True}

@router.post("/rag/sync_docs")
def admin_sync_docs(req: Request, body: dict | None = None):
    require_admin(req)
    base = None
    try:
        if body and isinstance(body, dict):
            base = str(body.get("base") or "").strip() or None
    except Exception:
        base = None
    if not base:
        base = getattr(settings, "docs_dir", "./docs")
    res = repo.sync_docs_dir(base)
    return {"synced": res, "base": base}

