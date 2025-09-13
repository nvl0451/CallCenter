from __future__ import annotations
import uuid, json, re
from fastapi import FastAPI, HTTPException, Request
import json as _json
import time as _time
from fastapi.middleware.cors import CORSMiddleware
from .schemas import *
from .config import settings
from .prompts import SYSTEM_BASE, CLASSIFY_PROMPT, PROMPTS_BY_TYPE
from .memory import add_message, get_session_messages
from .llm import llm_client
from .rag import rag_available, rag_unavailable_reason, ingest_texts, search
from .vision import vision_available, vision_unavailable_reason, classify_image
from . import db
from . import cache as app_cache

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
        "classify_max_tokens": getattr(settings, "classify_max_tokens", None),
        "rag": {"enabled": settings.enable_rag, "available": rag_available, "reason": rag_unavailable_reason},
        "vision": {
            "enabled": settings.enable_vision,
            "available": vision_available,
            "reason": vision_unavailable_reason,
            "multipart": mp,
            "backend": getattr(settings, "vision_backend", None),
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
    add_message(session_id, "system", SYSTEM_BASE)
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

def _extract_json(text: str) -> dict | None:
    s = text.strip()
    # remove code fences if present
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.IGNORECASE|re.MULTILINE).strip()
    try:
        return json.loads(s)
    except Exception:
        return None

async def _classify(text: str) -> tuple[str, float]:
    # Если нет ключа — используем локальную эвристику
    if not settings.openai_api_key:
        return _classify_heuristic(text)
    # Иначе пробуем LLM c JSON-ответом, при ошибке — эвристика
    # Build category list from DB cache if present
    names = app_cache.classes_names()
    if names:
        cats_str = ", ".join(names)
        prompt = (
            "Классифицируй пользовательский запрос в одну категорию из: "
            f"[{cats_str}]. Верни строго JSON без пояснений: {{\\\"category\\\":\\\"<категория>\\\",\\\"confidence\\\":<0..1>}}.\n\n"
            f"Запрос: {text}"
        )
    else:
        prompt = CLASSIFY_PROMPT.format(text=text)
    # Try Responses (gpt-5) minimal reasoning; if model rejects or 5xx occurs, use Chat with classify_model for this microtask
    out = ""
    lat_ms = 0
    use_responses = bool(getattr(settings, "openai_use_responses", True)) and str(getattr(settings, "openai_model", "")).startswith("gpt-5")
    err_note = None
    if use_responses:
        schema = {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": names if names else ["техподдержка", "продажи", "жалоба"]},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["category", "confidence"],
            "additionalProperties": False,
        }
        out, lat_ms = await llm_client.classify_json(prompt, max_tokens=settings.classify_max_tokens, json_schema=schema)
        if out.startswith("[offline]") or out.startswith("[incomplete:"):
            err_note = out
    if (not use_responses) or err_note:
        # Chat fallback for classifier ONLY (no heuristics): require JSON object
        cls_model = getattr(settings, "classify_model", "gpt-4o-mini")
        chat_messages = [
            {"role": "system", "content": "Верни строго JSON объект без пояснений."},
            {"role": "user", "content": prompt},
        ]
        out, lat_ms = await llm_client.classify_chat_json(chat_messages, max_tokens=settings.classify_max_tokens, model=cls_model)
    data = _extract_json(out) or {}
    raw_cat = str(data.get("category", ""))
    cat = _normalize_category(raw_cat)
    try:
        conf = float(data.get("confidence", 0.7))
    except Exception:
        conf = 0.7
    conf = max(0.0, min(1.0, conf))
    # Debug output for classification
    try:
        debug = {
            "cls_strict": settings.classify_strict,
            "model": getattr(settings, "openai_model", None),
            "cats_from_db": names,
            "prompt": prompt,
            "raw_output": out,
            "parsed": data,
            "raw_category": raw_cat,
            "normalized_category": cat,
            "llm_latency_ms": lat_ms,
        }
        print("[cls-debug] " + _json.dumps(debug, ensure_ascii=False))
    except Exception:
        pass
    if not cat:
        # Strict: no heuristic; fail clearly
        try:
            print("[cls-debug-final] " + _json.dumps({"mode":"strict-fail","reason": err_note or "normalize_empty", "raw": out}, ensure_ascii=False))
        except Exception:
            pass
        raise HTTPException(502, "classifier_failed")
    try:
        print("[cls-debug-final] " + _json.dumps({"mode":"llm","final_category": cat, "confidence": conf}, ensure_ascii=False))
    except Exception:
        pass
    return cat, conf

@app.post("/dialog/message", response_model=MessageResponse)
async def send_message(req: MessageRequest):
    session_id, user_text = req.session_id, req.message
    # контекст
    history = get_session_messages(session_id)
    if not history:
        raise HTTPException(400, "unknown session_id")
    add_message(session_id, "user", user_text)

    # классификация
    category, confidence = await _classify(user_text)
    # Choose system prompt: DB category prompt if available, else fallback
    db_prompts = app_cache.classes_prompts_map()
    sys_prompt = db_prompts.get(category) or PROMPTS_BY_TYPE.get(category, "")

    messages = [{"role": "system", "content": SYSTEM_BASE + "\n" + sys_prompt}]
    # Подмешиваем краткий контекст (последние 8 сообщений)
    for role, content in history[-8:]:
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_text})

    reply, latency_ms, cost = await llm_client.chat(messages)
    # Если пришёл оффлайн/ошибочный ответ — добавим дружественное сообщение
    if reply.startswith("[offline]"):
        reply = (
            "Извините, сейчас недоступен генеративный ответ. "
            "Мы зафиксировали ваше сообщение и свяжемся с вами.\n\n"
            f"Техническая справка: {reply}"
        )
    add_message(session_id, "assistant", reply)

    return MessageResponse(type=category, confidence=confidence, reply=reply, cost_estimate_usd=cost, latency_ms=latency_ms)

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
    sys = SYSTEM_BASE + "\nОтвечай, используя только факты из [DOC]. Если чего‑то нет в документах — честно скажи."
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
