from __future__ import annotations
import uuid
from fastapi import FastAPI, HTTPException, Request
import time as _time
from fastapi.middleware.cors import CORSMiddleware
from .schemas import *
from .config import settings
from . import cache as app_cache
from .constants import DEFAULT_CLASS_PROMPTS
from .memory import add_message, get_session_messages
from .llm import llm_client
from .local_classifier import classify as local_classify
from .rag import rag_available, rag_unavailable_reason, ingest_texts, search
from .vision import vision_available, vision_unavailable_reason, classify_image
from . import db
from . import sales
from . import rag as rag_mod

app = FastAPI(title="CallCenter LLM + RAG + Vision")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/features")
def features():
    try:
        import multipart  # type: ignore
        mp = True
    except Exception:
        mp = False
    # Cache and RAG metrics
    try:
        metrics = db.rag_doc_metrics()
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

# ---------- Диалог ----------
@app.post("/dialog/start", response_model=StartDialogResponse)
def start_dialog(req: StartDialogRequest):
    session_id = str(uuid.uuid4())
    add_message(session_id, "system", app_cache.system_base())
    if req.metadata:
        add_message(session_id, "system", f"meta:{req.metadata}")
    return StartDialogResponse(session_id=session_id)

def _classify_heuristic(text: str) -> tuple[str, float]:
    t = text.lower()
    if any(k in t for k in ["оплат", "карта", "счет", "счёт", "купить", "тариф", "подписк"]):
        return "продажи", 0.8
    if any(k in t for k in ["жалоб", "недоволен", "недовольна", "возврат", "refun", "обман"]):
        return "жалоба", 0.8
    return "техподдержка", 0.7

def _normalize_category(cat: str) -> str:
    cat_l = (cat or "").strip().lower()
    if not cat_l:
        return ""
    mp = app_cache.classes_normalization_map()
    if mp:
        # Only return a value if we have a direct match; otherwise let caller fallback to heuristic
        return mp.get(cat_l, "")
    # Fallback to baked-in mapping; return empty if unknown
    return {"техподдержка":"техподдержка", "поддержка":"техподдержка", "support":"техподдержка",
            "sales":"продажи", "продажи":"продажи",
            "жалоба":"жалоба", "complaint":"жалоба"}.get(cat_l, "")

async def _classify(text: str) -> tuple[str, float, dict]:
    # Prefer fast local SBERT classifier; fallback to heuristic if unavailable
    t0 = _time.perf_counter()
    try:
        cat, conf, meta = local_classify(text)
        meta = {"backend": "sbert", **meta}
        return cat, conf, meta
    except Exception:
        cat, conf = _classify_heuristic(text)
        wall_ms = int((_time.perf_counter() - t0) * 1000)
        return cat, conf, {"backend": "heuristic", "api_ms": 0, "wall_ms": wall_ms}

@app.post("/dialog/message", response_model=MessageResponse)
async def send_message(req: MessageRequest):
    session_id, user_text = req.session_id, req.message
    req_t0 = _time.perf_counter()
    # контекст
    history = get_session_messages(session_id)
    if not history:
        raise HTTPException(400, "unknown session_id")
    add_message(session_id, "user", user_text)

    # классификация
    cls_t0 = _time.perf_counter()
    category, confidence, cls_meta = await _classify(user_text)
    cls_wall_ms = int((_time.perf_counter() - cls_t0) * 1000)
    # Choose system prompt: DB category prompt if available, else fallback
    db_prompts = app_cache.classes_prompts_map()
    sys_prompt = db_prompts.get(category) or DEFAULT_CLASS_PROMPTS.get(category, "")

    messages = [{"role": "system", "content": app_cache.system_base() + "\n" + sys_prompt}]
    # Подмешиваем краткий контекст (последние 8 сообщений)
    for role, content in history[-8:]:
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_text})

    rep_t0 = _time.perf_counter()
    reply, reply_api_ms, cost = await llm_client.chat(messages)
    reply_wall_ms = int((_time.perf_counter() - rep_t0) * 1000)
    # Если пришёл оффлайн/ошибочный ответ — добавим дружественное сообщение
    if reply.startswith("[offline]"):
        reply = (
            "Извините, сейчас недоступен генеративный ответ. "
            "Мы зафиксировали ваше сообщение и свяжемся с вами.\n\n"
            f"Техническая справка: {reply}"
        )
    add_message(session_id, "assistant", reply)

    total_ms = int((_time.perf_counter() - req_t0) * 1000)
    latencies = {
        "classify_backend": getattr(settings, "classifier_backend", "sbert"),
        "classify_api_ms": int(cls_meta.get("api_ms", 0)),
        "classify_wall_ms": cls_wall_ms,
        "reply_api_ms": reply_api_ms,
        "reply_wall_ms": reply_wall_ms,
        "total_ms": total_ms,
    }
    return MessageResponse(type=category, confidence=confidence, reply=reply, cost_estimate_usd=cost, latency_ms=reply_api_ms, latencies=latencies)

# ---------- RAG ----------
@app.post("/rag/ingest")
def rag_ingest(req: IngestRequest):
    if not settings.enable_rag:
        raise HTTPException(503, "RAG disabled via ENABLE_RAG=0")
    if not rag_available:
        raise HTTPException(503, rag_unavailable_reason or "RAG dependencies unavailable")
    texts = req.documents or []
    # Если не передали — зальём демо файлы
    if not texts:
        demo = []
        for p in ["data/kb/pricing.md", "data/kb/refund_policy.md", "data/kb/troubleshooting.md"]:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    demo.append(f.read())
            except FileNotFoundError:
                pass
        texts = demo
    ids = ingest_texts(texts, metadatas=[{"source":"ingest"} for _ in texts])
    return {"count": len(ids), "ids": ids}

@app.post("/rag/ask", response_model=AskResponse)
async def rag_ask(req: AskRequest):
    if not settings.enable_rag:
        raise HTTPException(503, "RAG disabled via ENABLE_RAG=0")
    if not rag_available:
        raise HTTPException(503, rag_unavailable_reason or "RAG dependencies unavailable")
    try:
        docs = search(req.question, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(500, f"RAG search failed: {e}")
    # Сформируем подсказку с цитатами
    if not docs:
        return AskResponse(answer="В базе знаний пока нет документов.", sources=[])
    context = "\n\n".join([f"[DOC {i+1}]\n" + (d.get("document") or "") for i, d in enumerate(docs)])
    sys = app_cache.system_base() + "\nОтвечай, используя только факты из [DOC]. Если чего‑то нет в документах — честно скажи."
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": f"Вопрос: {req.question}\n\nКонтекст:\n{context}"},
    ]
    try:
        answer, _, _ = await llm_client.chat(messages)
    except Exception as e:
        raise HTTPException(502, f"LLM call failed: {e}")
    return AskResponse(answer=answer or "", sources=[{"id": d.get("id"), "score": d.get("score")} for d in docs])

# ---------- Vision ----------
# Only register upload endpoint if vision enabled AND deps for multipart are present
try:
    import multipart  # type: ignore
    multipart_available = True
except Exception:
    multipart_available = False

if settings.enable_vision and vision_available and multipart_available:
    from fastapi import UploadFile, File

    @app.post("/vision/classify", response_model=VisionResponse)
    async def classify(file: UploadFile = File(...)):
        if not file.content_type.startswith("image/"):
            raise HTTPException(400, "upload an image file")
        from PIL import Image  # lazy import to avoid startup crash if missing
        img = Image.open(file.file).convert("RGB")
        try:
            category, conf, logits, labels = classify_image(img)
        except Exception as e:
            raise HTTPException(503, f"Vision backend error: {e}")
        return VisionResponse(category=category, confidence=conf, logits=logits, labels=labels)
else:
    @app.post("/vision/classify", response_model=VisionResponse)
    async def classify_disabled():
        if not settings.enable_vision:
            raise HTTPException(503, "Vision disabled via ENABLE_VISION=0")
        if not vision_available:
            raise HTTPException(503, vision_unavailable_reason or "Vision dependencies unavailable")
        if not multipart_available:
            raise HTTPException(503, 'python-multipart not installed in current interpreter')
        raise HTTPException(503, "Vision unavailable")


@app.on_event("startup")
def _startup():
    # Run migrations, ensure storage dirs, warm caches
    try:
        db.run_migrations()
    except Exception:
        pass
    # Ensure default editable settings
    try:
        db.ensure_default_system_base()
    except Exception:
        pass
    # Ensure default editable system_base exists
    try:
        db.ensure_default_system_base()
    except Exception:
        pass
    # Seed defaults if tables empty
    inserted = None
    try:
        inserted = db.bootstrap_defaults()
    except Exception:
        inserted = None
    try:
        db.ensure_storage_dirs()
    except Exception:
        pass
    try:
        app_cache.reload_caches()
    except Exception:
        pass
    # Warm SBERT classifier to avoid first-use latency
    try:
        if getattr(settings, "classifier_backend", "sbert") == "sbert":
            from . import local_classifier as _lc
            _lc.warm()
    except Exception:
        pass
    # Optional initial RAG ingest for seeded inline docs
    try:
        if settings.enable_rag and rag_available and inserted and inserted.get("rag_docs", 0) > 0:
            texts = db.load_active_inline_docs()
            if texts:
                ingest_texts(texts, metadatas=[{"source": "bootstrap"} for _ in texts])
                try:
                    db.mark_inline_indexed(getattr(settings, "openai_embed_model", ""))
                except Exception:
                    pass
    except Exception:
        pass


def _require_admin(req: Request):
    token = req.headers.get("authorization") or req.headers.get("Authorization")
    expected = settings.admin_token.strip()
    if not expected:
        raise HTTPException(401, "ADMIN_TOKEN is not configured")
    if not token or not token.lower().startswith("bearer "):
        raise HTTPException(401, "Missing Bearer token")
    value = token.split(" ", 1)[1].strip()
    if value != expected:
        raise HTTPException(403, "Invalid admin token")


@app.post("/admin/bootstrap")
def admin_bootstrap(req: Request):
    _require_admin(req)
    db.run_migrations()
    ins = db.bootstrap_defaults()
    app_cache.reload_caches()
    rag_ingested = 0
    if settings.enable_rag and rag_available and ins.get("rag_docs", 0) > 0:
        try:
            texts = db.load_active_inline_docs()
            if texts:
                ingest_texts(texts, metadatas=[{"source": "bootstrap"} for _ in texts])
                rag_ingested = len(texts)
                try:
                    db.mark_inline_indexed(getattr(settings, "openai_embed_model", ""))
                except Exception:
                    pass
        except Exception:
            rag_ingested = 0
    return {
        "inserted": ins,
        "caches": {
            "classes": len(app_cache.classes_cache()),
            "vision_labels": len(app_cache.vision_labels_cache()),
        },
        "rag": {"ingested": rag_ingested, "enabled": settings.enable_rag, "available": rag_available},
    }


@app.post("/admin/update_stems")
def admin_update_stems(req: Request):
    _require_admin(req)
    from . import db as _db
    updated = _db.update_default_stems()
    app_cache.reload_caches()
    return {"updated_rows": updated}

# -------- Admin CRUD: Classes --------
@app.get("/admin/classes", response_model=list[AdminClassOut])
def admin_list_classes(req: Request):
    _require_admin(req)
    return db.fetch_classes(active_only=False)

@app.post("/admin/classes", response_model=AdminClassOut)
def admin_create_class(req: Request, body: AdminClassIn):
    _require_admin(req)
    new_id = db.insert_class(body.name, body.synonyms, body.stems, body.system_prompt, body.priority, body.active)
    app_cache.reload_caches()
    rows = db.fetch_classes(active_only=False)
    return next(r for r in rows if r["id"] == new_id)

@app.put("/admin/classes/{cls_id}", response_model=AdminClassOut)
def admin_update_class(req: Request, cls_id: int, body: AdminClassIn):
    _require_admin(req)
    db.update_class(
        cls_id,
        name=body.name,
        synonyms=body.synonyms,
        stems=body.stems,
        system_prompt=body.system_prompt,
        priority=body.priority,
        active=body.active,
    )
    app_cache.reload_caches()
    rows = db.fetch_classes(active_only=False)
    return next(r for r in rows if r["id"] == int(cls_id))

@app.delete("/admin/classes/{cls_id}")
def admin_delete_class(req: Request, cls_id: int):
    _require_admin(req)
    changed = db.soft_delete_class(cls_id)
    app_cache.reload_caches()
    return {"deleted": changed}

# -------- Admin CRUD: Vision Labels --------
@app.get("/admin/vision", response_model=list[AdminVisionOut])
def admin_list_vision(req: Request):
    _require_admin(req)
    return db.fetch_vision_labels(active_only=False)

@app.post("/admin/vision", response_model=AdminVisionOut)
def admin_create_vision(req: Request, body: AdminVisionIn):
    _require_admin(req)
    new_id = db.insert_vision_label(body.name, body.synonyms, body.templates, body.priority, body.active)
    app_cache.reload_caches()
    rows = db.fetch_vision_labels(active_only=False)
    return next(r for r in rows if r["id"] == new_id)

@app.put("/admin/vision/{lbl_id}", response_model=AdminVisionOut)
def admin_update_vision(req: Request, lbl_id: int, body: AdminVisionIn):
    _require_admin(req)
    db.update_vision_label(lbl_id, name=body.name, synonyms=body.synonyms, templates=body.templates, priority=body.priority, active=body.active)
    app_cache.reload_caches()
    rows = db.fetch_vision_labels(active_only=False)
    return next(r for r in rows if r["id"] == int(lbl_id))

@app.delete("/admin/vision/{lbl_id}")
def admin_delete_vision(req: Request, lbl_id: int):
    _require_admin(req)
    changed = db.soft_delete_vision_label(lbl_id)
    app_cache.reload_caches()
    return {"deleted": changed}

# -------- Admin CRUD: RAG Documents --------
@app.get("/admin/rag/docs", response_model=list[AdminRagDocOut])
def admin_list_rag_docs(req: Request):
    _require_admin(req)
    return db.list_rag_documents(active_only=False)

@app.post("/admin/rag/inline", response_model=AdminRagDocOut)
def admin_create_rag_inline(req: Request, body: AdminRagInlineIn):
    _require_admin(req)
    doc_id = db.insert_rag_inline(body.title, body.content_text, source="admin-inline")
    # Try to index synchronously if enabled and requested
    if settings.enable_rag and rag_available and body.index_now:
        try:
            meta = {"source": "admin-inline", "doc_id": str(doc_id)}
            rag_mod.ingest_texts([body.content_text], metadatas=[meta])
            db.update_rag_doc(doc_id, indexed_at=_time.time(), embed_model=rag_mod.get_embed_note(), chunks_count=rag_mod.chunk_count(body.content_text), dirty=0)
        except Exception:
            pass
    row = db.get_rag_doc(doc_id)
    return row  # type: ignore

@app.put("/admin/rag/docs/{doc_id}", response_model=AdminRagDocOut)
def admin_update_rag_inline(req: Request, doc_id: int, payload: dict):
    _require_admin(req)
    row = db.get_rag_doc(doc_id)
    if not row:
        raise HTTPException(404, "not found")
    if row.get("kind") != "inline":
        raise HTTPException(400, "only inline docs can be updated via this endpoint")
    content = str(payload.get("content_text", ""))
    if not content:
        raise HTTPException(400, "content_text required")
    db.update_rag_doc(doc_id, content_text=content, sha256=db._sha256_text(content), bytes=len(content.encode("utf-8")), dirty=1)
    return db.get_rag_doc(doc_id)  # type: ignore

# Multipart upload for .txt/.md
try:
    from fastapi import UploadFile, File, Form
    _mp_ok = True
except Exception:
    _mp_ok = False

if _mp_ok:
    import os, re as _re, hashlib
    def _slugify(name: str) -> str:
        s = name.lower().strip()
        s = _re.sub(r"[^a-z0-9а-яё_.-]+", "-", s)
        s = _re.sub(r"-+", "-", s).strip("-")
        return s or "doc"

    @app.post("/admin/rag/file", response_model=AdminRagDocOut)
    async def admin_upload_rag_file(req: Request, file: UploadFile = File(...), index_now: int = Form(1)):
        _require_admin(req)
        # Validate ext
        fname = file.filename or "upload.txt"
        ext = (fname.rsplit(".", 1)[-1] or "").lower()
        if ext not in ("txt", "md"):
            raise HTTPException(400, "Only .txt and .md are allowed")
        # Read bytes with limit
        raw = await file.read()
        if len(raw) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"File too large, limit {settings.max_upload_mb} MB")
        try:
            text = raw.decode("utf-8", errors="ignore")
        except Exception:
            text = ""
        sha = hashlib.sha256(raw).hexdigest()
        mime = "text/markdown" if ext == "md" else "text/plain"
        # Insert placeholder to obtain id
        new_id = db.insert_rag_file(title=fname, rel_path="", bytes_len=len(raw), sha256=sha, mime=mime, source="admin-upload")
        # Build final storage path
        db.ensure_storage_dirs()
        slug = _slugify(fname)
        rel_name = f"{new_id}_{slug}"
        abs_dir = settings.rag_storage_dir
        abs_path = os.path.join(abs_dir, rel_name)
        with open(abs_path, "wb") as f:
            f.write(raw)
        db.update_rag_doc(new_id, rel_path=rel_name)
        # Optional indexing
        if settings.enable_rag and rag_available and int(index_now) == 1:
            try:
                meta = {"source": "admin-upload", "doc_id": str(new_id), "filename": fname}
                rag_mod.ingest_texts([text], metadatas=[meta])
                db.update_rag_doc(new_id, indexed_at=_time.time(), embed_model=rag_mod.get_embed_note(), chunks_count=rag_mod.chunk_count(text), dirty=0)
            except Exception:
                pass
        row = db.get_rag_doc(new_id)
        return row  # type: ignore

@app.delete("/admin/rag/docs/{doc_id}")
def admin_delete_rag_doc(req: Request, doc_id: int):
    _require_admin(req)
    row = db.get_rag_doc(doc_id)
    if not row:
        raise HTTPException(404, "not found")
    # Try to delete from vector DB by metadata filter
    try:
        if settings.enable_rag and rag_available:
            rag_mod.delete_by_doc_id(doc_id)
    except Exception:
        pass
    # Soft delete row
    changed = db.soft_delete_rag_doc(doc_id)
    # Optionally physically delete file
    if settings.rag_hard_delete and row.get("kind") == "file" and row.get("rel_path"):
        try:
            import os
            os.remove(os.path.join(settings.rag_storage_dir, row["rel_path"]))
        except Exception:
            pass
    return {"deleted": changed}


def _reindex_doc(doc: dict) -> dict:
    if not (settings.enable_rag and rag_available):
        raise HTTPException(503, "RAG unavailable")
    import time as _t
    t0 = _t.perf_counter()
    content = None
    if doc.get("kind") == "inline":
        content = doc.get("content_text") or ""
    else:
        # file
        rel = doc.get("rel_path") or ""
        import os
        try:
            with open(os.path.join(settings.rag_storage_dir, rel), "rb") as f:
                raw = f.read()
            content = raw.decode("utf-8", errors="ignore")
        except Exception as e:
            raise HTTPException(500, f"failed to read file content: {e}")
    # delete previous vectors
    try:
        rag_mod.delete_by_doc_id(doc.get("id"))
    except Exception:
        pass
    # ingest again
    chunks = 0
    try:
        meta = {"source": doc.get("source") or "admin", "doc_id": str(doc.get("id"))}
        rag_mod.ingest_texts([content], metadatas=[meta])
        chunks = rag_mod.chunk_count(content)
    except Exception as e:
        raise HTTPException(500, f"embed/ingest failed: {e}")
    # update DB
    try:
        db.update_rag_doc(int(doc.get("id")), indexed_at=_time.time(), embed_model=rag_mod.get_embed_note(), chunks_count=chunks, dirty=0)
    except Exception:
        pass
    ms = int((_t.perf_counter() - t0) * 1000)
    return {"id": doc.get("id"), "chunks": chunks, "ms": ms, "ok": True}

@app.post("/admin/rag/reindex/{doc_id}")
def admin_reindex_one(req: Request, doc_id: int):
    _require_admin(req)
    row = db.get_rag_doc(doc_id)
    if not row or int(row.get("active", 1)) != 1:
        raise HTTPException(404, "doc not found or inactive")
    return _reindex_doc(row)

@app.post("/admin/rag/reindex_dirty")
def admin_reindex_dirty(req: Request, limit: int | None = None):
    _require_admin(req)
    rows = db.list_dirty_docs(limit=limit)
    out = []
    errs = 0
    for r in rows:
        try:
            out.append(_reindex_doc(r))
        except Exception as e:
            errs += 1
            out.append({"id": r.get("id"), "ok": False, "error": str(e)})
    return {"processed": len(rows), "errors": errs, "results": out}

@app.post("/admin/rag/mark_all_dirty")
def admin_mark_all_dirty(req: Request):
    _require_admin(req)
    return {"marked": db.mark_all_dirty()}


# ---------- Agent ----------
@app.post("/agent/message", response_model=AgentMessageResponse)
async def agent_message(req: AgentMessageRequest):
    import time as _t
    t0 = _t.perf_counter()
    tools_used: list[str] = []
    max_calls = 3
    calls = 0
    session_id = req.session_id or str(uuid.uuid4())
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "text required")
    # Sales session state and opportunistic slot extraction
    st_before = sales.get(session_id)
    try:
        slot_guess = sales.parse_message_slots(text)
    except Exception:
        slot_guess = {}
    if slot_guess:
        sales.update(session_id, **slot_guess)
    st = sales.get(session_id)
    # Track material info changes (affects re-pitch)
    material_keys = {k for k in (slot_guess or {}).keys() if k in ("seats", "channels", "sso_required", "analytics_required", "billing")}
    new_email = (st_before.get("admin_email") is None and bool(st.get("admin_email")))
    new_billing = (st_before.get("billing") not in ("monthly", "annual") and st.get("billing") in ("monthly", "annual"))
    # Persist user message for conversational context
    try:
        add_message(session_id, "user", text)
    except Exception:
        pass

    def budget_ok():
        return calls < max_calls and ((_t.perf_counter() - t0) < 20.0)

    # 1) Intent classification (reuse strict classifier): expect 'продажи' for purchase intent
    intent = None
    try:
        if budget_ok():
            calls += 1; tools_used.append("classify_text")
            intent, _, _m = await _classify(text)
    except HTTPException:
        intent = None
    if intent:
        sales.update(session_id, intent=intent)

    # 2) If purchase intent, try to detect plan (Basic/Plus/Business)
    plan = None; plan_conf = 0.0
    locked = bool(st.get("plan_locked"))
    if locked and not material_keys:
        plan = st.get("plan_candidate"); plan_conf = float(st.get("plan_confidence") or 0.9)
    elif intent == "продажи" and budget_ok():
        labels = ["Basic", "Plus", "Business"]
        # Prefer SBERT-based quick classifier among labels
        try:
            from .local_classifier import classify_among as _cls_among
            plan, plan_conf = _cls_among(text, [
                "Basic plan", "Plus plan", "Business plan"
            ])
            plan = plan.split()[0]
        except Exception:
            # Heuristic fallback
            low = text.lower()
            if any(k in low for k in ["basic", "starter", "базов"]):
                plan, plan_conf = "Basic", 0.6
            elif any(k in low for k in ["plus", "плюс", "pro"]):
                plan, plan_conf = "Plus", 0.6
            elif any(k in low for k in ["business", "бизнес", "enterprise"]):
                plan, plan_conf = "Business", 0.6

    if plan:
        sales.update(session_id, plan_candidate=plan, plan_confidence=plan_conf)
    # If unsure but we have some slots, recommend a plan heuristically
    try:
        st_now = sales.get(session_id)
        if (not plan or plan_conf < 0.65) and intent == "продажи":
            rec = sales.recommend_plan(st_now.get("seats"), st_now.get("channels") or [], st_now.get("sso_required"), st_now.get("analytics_required"))
            plan = rec; plan_conf = max(plan_conf, 0.7)
            sales.update(session_id, plan_candidate=plan, plan_confidence=plan_conf)
    except Exception:
        pass

    # 3) If confident plan, optionally fetch KB snippet via RAG (avoid re-pitch when locked and no new info)
    docs = []
    rag_ms = 0
    do_pitch = bool(plan and plan_conf >= 0.65 and settings.enable_rag and rag_available and budget_ok())
    if do_pitch and locked and not material_keys and not new_billing and not new_email:
        do_pitch = False
    if do_pitch:
        calls += 1; tools_used.append("rag_search")
        ru = {"Basic":"Базовый","Plus":"Плюс","Business":"Бизнес"}.get(plan, plan)
        q = f"Тариф {ru} (план {plan}) цена, ключевые функции, ограничения"
        try:
            _rt0 = _t.perf_counter()
            docs = search(q, top_k=5)
            rag_ms = int((_t.perf_counter() - _rt0) * 1000)
        except Exception:
            docs = []

    # Build reply using LLM or templates
    reply = ""
    sources = None
    llm_fallback = False
    llm_ms = 0
    price_ms = 0
    quote_ms = 0
    quote_obj = None

    def _extract_plan_section(txt: str, plan_en: str, plan_ru: str) -> str:
        import re as _re
        t = txt or ""
        # Try headings like ## Плюс (Plus) or ## Plus
        patterns = [
            rf"^##\s*{plan_ru}.*$[\s\S]*?(?=^##\s|\Z)",
            rf"^##\s*{plan_en}.*$[\s\S]*?(?=^##\s|\Z)",
        ]
        for pat in patterns:
            m = _re.search(pat, t, flags=_re.MULTILINE)
            if m:
                return m.group(0)
        return t
    # Confident plan path (with or without DOC depending on availability)
    if plan and plan_conf >= 0.65:
        ru = {"Basic":"Базовый","Plus":"Плюс","Business":"Бизнес"}.get(plan, plan)
        import re as _re
        sections = []
        if docs:
            for d in docs:
                raw = d.get("document", "") or ""
                sec = _extract_plan_section(raw, plan, ru)
                # Remove generic inheritance lines to avoid repetition
                sec_lines = [ln for ln in sec.splitlines() if not _re.search(r"(?i)(вс.? из базов|everything in basic)", ln)]
                sec_clean = "\n".join(sec_lines)
                sections.append(sec_clean[:1500])
        context = "\n\n".join(sections)
        included_hint = {
            "Basic": "(до 3 пользователей включено)",
            "Plus": "(до 10 пользователей включено)",
            "Business": "(до 25 пользователей включено)",
        }.get(plan, "")
        st_now_for_prompt = sales.get(session_id)
        greeted = bool(st_now_for_prompt.get("greeted"))
        has_billing = st_now_for_prompt.get("billing") in ("monthly", "annual")
        has_email = bool(st_now_for_prompt.get("admin_email"))
        # Personalize with compact slot header
        slot_header = []
        if st_now_for_prompt.get("seats"):
            slot_header.append(f"Пользователи: {st_now_for_prompt.get('seats')}")
        if st_now_for_prompt.get("channels"):
            slot_header.append("Каналы: " + ", ".join(st_now_for_prompt.get("channels") or []))
        if st_now_for_prompt.get("billing"):
            slot_header.append("Оплата: " + ("месяц" if st_now_for_prompt.get("billing")=="monthly" else "год"))
        slot_header_txt = ("; ".join(slot_header)) if slot_header else ""
        # Build CTA text depending on what is already provided
        if has_billing and has_email:
            bill_txt = "помесячной" if st_now_for_prompt.get("billing") == "monthly" else "годовой"
            cta_text = f"Оформляю по {bill_txt} оплате и отправлю предложение на {st_now_for_prompt.get('admin_email')}."
        elif has_billing and not has_email:
            bill_txt = "помесячной" if st_now_for_prompt.get("billing") == "monthly" else "годовой"
            cta_text = f"Оформлю по {bill_txt} оплате. Укажите e‑mail администратора для приглашения команды {included_hint}."
        elif (not has_billing) and has_email:
            cta_text = f"Приглашу команду на {st_now_for_prompt.get('admin_email')}. Удобнее платить помесячно или за год (−20%)?"
        else:
            cta_text = f"Хотите выбрать период оплаты (месяц/год; за год — скидка 20%)? И пришлите e‑mail администратора для приглашения команды {included_hint}."
        if not greeted:
            # First pitch: only include benefits if docs fetched; personalize with slot header
            sys = app_cache.system_base() + (
                "\nСтиль: тёплый, дружелюбный, уверенный менеджер по продажам мужского пола. Пиши по‑русски, живым человеческим языком (без канцеляризмов и англицизмов)." \
                "\nФормат ответа:" \
                + ("\n— Короткий приветственный лид (без общих рассуждений). " + (slot_header_txt if slot_header_txt else "")) \
                + ("\n— Цена: 1 короткая фраза." if docs else "") \
                + ("\n— Ключевые выгоды плана: 3–5 маркеров только при наличии [DOC]." if docs else "") \
                + "\n— Заверши дружелюбной фразой‑призывом: " + cta_text + " (не используй слова ‘Следующий шаг’)." \
                + ("\nИспользуй только факты из [DOC]." if docs else "")
            )
        else:
            # Follow-up: confirm fit briefly; avoid repeating benefits unless we fetched new DOC
            sys = app_cache.system_base() + (
                "\nСтиль: тёплый, дружелюбный, уверенный менеджер по продажам мужского пола. Пиши по‑русски, живым человеческим языком (без канцеляризмов и англицизмов)." \
                "\nФормат ответа:" \
                + ("\n— Коротко подтвердить соответствие плана без приветствия." + ("\n— Цена: 1 короткая фраза." if docs else "")) \
                + "\n— Заверши дружелюбной фразой‑призывом: " + cta_text + " (без повторения выгод)."
            )
        # Build conversational context (last 8 turns)
        convo = [{"role": "system", "content": sys}]
        try:
            hist = get_session_messages(session_id)
        except Exception:
            hist = []
        for role, content in hist[-8:]:
            if role in ("user", "assistant"):
                convo.append({"role": role, "content": content})
        # Append task instruction depending on availability of DOC
        if docs:
            task = f"Пользователь хочет оформить план {plan}. На основе [DOC] опиши цену, ключевые выгоды и мягко предложи, что сделать дальше (без слов ‘Следующий шаг’).\n\n[DOC]\n{context}"
        else:
            task = f"Пользователь хочет оформить план {plan}. Коротко подтверди соответствие плана и предложи понятное действие без повторения списка выгод."
        convo.append({"role": "user", "content": task})
        messages = convo
        try:
            _lt0 = _t.perf_counter()
            reply, _, _ = await llm_client.chat(messages)
            llm_ms = int((_t.perf_counter() - _lt0) * 1000)
        except Exception:
            reply = f"План {plan}: см. детали в документации."
            llm_fallback = True
        # Mark greeted after first pitch
        if not greeted:
            try:
                sales.update(session_id, greeted=True)
            except Exception:
                pass
        def _src_item(d: dict) -> dict:
            md = d.get("metadata") or {}
            doc = (d.get("document", "") or "").replace("\n", " ")
            # collapse whitespace and cut to nearest word boundary around 200-240 chars
            import re as _re
            doc = _re.sub(r"\s+", " ", doc).strip()
            snip = doc[:240]
            if len(doc) > 240:
                # try not to cut mid-word
                cut = snip.rfind(" ")
                if cut > 160:
                    snip = snip[:cut] + "…"
            return {
                "score": d.get("score"),
                "id": d.get("id"),
                "source": md.get("source"),
                "doc_id": md.get("doc_id"),
                "snippet": snip,
            }
        sources = [_src_item(d) for d in docs]
        # If user already provided billing + admin_email, emit a quote
        try:
            st_now = sales.get(session_id)
            if st_now.get("admin_email") and st_now.get("billing") in ("monthly", "annual") and calls < max_calls:
                # price_calc (stub)
                calls += 1; tools_used.append("price_calc")
                _pt0 = _t.perf_counter()
                monthly = {"Basic":29.0, "Plus":99.0, "Business":299.0}.get(plan, 0.0)
                if st_now.get("billing") == "annual":
                    base = monthly * 12 * 0.8
                else:
                    base = monthly
                total = base
                price_ms = int((_t.perf_counter() - _pt0) * 1000)
                # create_quote (stub)
                import uuid as _uuid, datetime as _dt
                calls += 1; tools_used.append("create_quote")
                _qt0 = _t.perf_counter()
                qid = str(_uuid.uuid4())
                quote_obj = {
                    "quote_id": qid,
                    "url": f"https://example.local/quote/{qid}",
                    "plan": plan,
                    "billing": st_now.get("billing"),
                    "seats": st_now.get("seats"),
                    "total_usd": round(total, 2),
                    "sent_to": st_now.get("admin_email"),
                }
                quote_ms = int((_t.perf_counter() - _qt0) * 1000)
                reply = (reply + "\n\n— Оформил предложение: " + quote_obj["url"] +
                         f" (отправил на {quote_obj['sent_to']}).")
                sales.update(session_id, stage="quoted", plan_locked=True)
            else:
                sales.update(session_id, stage="confirm", plan_locked=True)
        except Exception:
            pass
    else:
        # Uncertain plan → LLM negotiатор (strict ru-only): exactly 2 questions + friendly CTA
        greeted = bool(sales.get(session_id).get("greeted"))
        if not greeted:
            sys = (
                app_cache.system_base() +
                "\nСтиль: тёплый, дружелюбный, уверенный менеджер по продажам мужского пола. Только по‑русски, без эмодзи и иностранных символов." \
                "\nФормат ответа:" \
                "\n— Короткая приветственная фраза (1 строка)." \
                "\n— Два уточняющих вопроса (каждый с маркером «— …», очень коротко)." \
                "\n— Заверши 1 дружелюбной фразой‑призывом (не используй слова ‘Следующий шаг’), например: ‘Ответьте — подберу план и пришлю ссылку на оплату/счёт сегодня’."
            )
        else:
            sys = (
                app_cache.system_base() +
                "\nСтиль: тёплый, дружелюбный, уверенный менеджер по продажам мужского пола. Только по‑русски, без эмодзи и иностранных символов." \
                "\nФормат ответа:" \
                "\n— Два уточняющих вопроса (каждый с маркером «— …», очень коротко)." \
                "\n— Заверши 1 дружелюбной фразой‑призывом (не используй слова ‘Следующий шаг’), например: ‘Ответьте — подберу план и пришлю ссылку на оплату/счёт сегодня’."
            )
        user_msg = (
            "Клиент не уверен в плане (Basic/Plus/Business). Задай ровно 2 лаконичных вопроса, "
            "которые помогут выбрать план по числу пользователей, каналам (email/чат/WhatsApp) и требованиям (SSO/аналитика). "
            "Затем предложи один следующий шаг. Текст клиента: " + text
        )
        # Build conversational context (last 8 turns)
        convo = [{"role": "system", "content": sys}]
        try:
            hist = get_session_messages(session_id)
        except Exception:
            hist = []
        for role, content in hist[-8:]:
            if role in ("user", "assistant"):
                convo.append({"role": role, "content": content})
        convo.append({"role": "user", "content": user_msg})
        messages = convo
        try:
            _lt0 = _t.perf_counter()
            reply, _, _ = await llm_client.chat(messages)
            llm_ms = int((_t.perf_counter() - _lt0) * 1000)
        except Exception:
            reply = (
                "1) Вопрос 1: Сколько будет пользователей?\n"
                "2) Вопрос 2: Какие каналы нужны (email/чат/WhatsApp)? Нужны ли SSO/аналитика?\n"
                "Следующий шаг: после ответа предложу подходящий план и ссылку на оформление."
            )
            llm_fallback = True
        # mark greeted to avoid repeated salutations in subsequent turns
        try:
            if not greeted:
                sales.update(session_id, greeted=True)
        except Exception:
            pass

    lat = {"rag_ms": rag_ms, "llm_ms": llm_ms, "price_ms": price_ms, "quote_ms": quote_ms, "total_ms": int((_t.perf_counter() - t0) * 1000)}
    st_out = sales.get(session_id)
    # Persist assistant reply
    try:
        add_message(session_id, "assistant", reply or "")
    except Exception:
        pass
    return AgentMessageResponse(session_id=session_id, tools_used=tools_used, reply=reply or "", sources=sources, latencies=lat, llm_fallback=llm_fallback, stage=st_out.get("stage"), slots={k: st_out.get(k) for k in ("seats","channels","billing","admin_email","plan_candidate")}, quote=quote_obj)


# ---------- Admin: KB reseed from files ----------
@app.post("/admin/rag/sync_seed_kb")
def admin_sync_seed_kb(req: Request):
    _require_admin(req)
    res = db.sync_seed_kb_from_files()
    return res

@app.post("/admin/rag/reset_index")
def admin_reset_index(req: Request):
    _require_admin(req)
    if not (settings.enable_rag and rag_available):
        raise HTTPException(503, "RAG unavailable")
    rag_mod.reset_index()
    return {"ok": True}

@app.post("/admin/rag/sync_docs")
def admin_sync_docs(req: Request, body: dict | None = None):
    _require_admin(req)
    base = None
    try:
        if body and isinstance(body, dict):
            base = str(body.get("base") or "").strip() or None
    except Exception:
        base = None
    if not base:
        base = getattr(settings, "docs_dir", "./docs")
    res = db.sync_docs_dir(base)
    return {"synced": res, "base": base}


# -------- Admin: App Settings (system_base) --------
@app.get("/admin/settings")
def admin_get_settings(req: Request):
    _require_admin(req)
    return {"system_base": app_cache.system_base()}

@app.put("/admin/settings/system_base")
async def admin_put_system_base(req: Request, payload: dict):
    _require_admin(req)
    text = str(payload.get("system_base", ""))
    if not text.strip():
        raise HTTPException(400, "system_base required")
    db.set_setting("system_base", text)
    app_cache.reload_caches()
    return {"ok": True, "system_base": app_cache.system_base()}
