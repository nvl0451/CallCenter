from __future__ import annotations
import uuid, json, re
from fastapi import FastAPI, HTTPException, Request
import json as _json
import time as _time
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from .schemas import *
from .config import settings
from .prompts import SYSTEM_BASE, CLASSIFY_PROMPT, PROMPTS_BY_TYPE
from .memory import add_message, get_session_messages
from .llm import llm_client
from .local_classifier import classify as local_classify
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
        "classifier": {
            "backend": getattr(settings, "classifier_backend", "responses"),
            "sbert_model": getattr(settings, "sbert_model", None) if getattr(settings, "classifier_backend", "responses") == "sbert" else None,
        },
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

async def _classify(text: str) -> tuple[str, float, dict]:
    # Если нет ключа — используем локальную эвристику
    if not settings.openai_api_key:
        cat, conf = _classify_heuristic(text)
        return cat, conf, {"mode": "offline", "wall_ms": 0, "api_ms": 0, "attempts": 0}
    # Local SBERT classifier (fast, on-CPU) if configured
    if getattr(settings, "classifier_backend", "responses") == "sbert":
        cat, conf, meta = local_classify(text)
        return cat, conf, meta
    # Иначе — строго Responses (gpt‑5) с компактным индекс‑JSON. Попытаться с двумя форматами input.
    names = app_cache.classes_names()
    labels = names if names else ["техподдержка", "продажи", "жалоба"]
    label_map = {i: lbl for i, lbl in enumerate(labels)}
    idx_range = "|".join(str(i) for i in range(len(labels)))
    # Build class lines with stems (from DB if available)
    try:
        from .cache import classes_stems_map as _stems_map
        stems_mp = _stems_map()
    except Exception:
        stems_mp = {}
    _defaults_stems = {
        "техподдержка": ["ошибк", "не запуска", "проблем", "support"],
        "продажи": ["куп", "тариф", "цен", "оплат", "счёт", "подписк"],
        "жалоба": ["жалоб", "возврат", "refund", "дважд"],
    }
    _class_lines = []
    for i, lbl in label_map.items():
        _stems = stems_mp.get(lbl) or _defaults_stems.get(lbl, [])
        _kw = ", ".join(_stems[:6]) if _stems else ""
        _class_lines.append(f"{i}={lbl}{' (стемы: ' + _kw + ')' if _kw else ''}")
    cls_prompt = (
        "Категории: " + "; ".join(_class_lines) + ". "
        f"Выбери ровно одну. Ответь строго одним JSON без пробелов: {{\\\"index\\\":<{idx_range}>,\\\"confidence\\\":<0.00-1.00>}}. "
        "Только JSON, без пояснений. Запрос: " + text
    )
    out = ""; lat_ms = 0; err_note = None
    attempts = 0
    data = {}
    raw_cat = ""; cat = ""; conf = 0.0
    wall_t0 = _time.perf_counter()
    api_ms_total = 0
    used_path = "responses"
    deadline_ms = max(200, int(getattr(settings, "classify_deadline_ms", 1200)))
    if settings.classify_use_responses:
        for shape in ("string", "content"):
            # Respect overall deadline across attempts
            elapsed = int((_time.perf_counter() - wall_t0) * 1000)
            remaining = max(50, deadline_ms - elapsed)
            attempts += 1
            try:
                out, lat_ms = await asyncio.wait_for(
                    llm_client.classify_json(cls_prompt, max_tokens=settings.classify_max_tokens, json_schema={}, shape=shape),
                    timeout=remaining / 1000.0,
                )
            except asyncio.TimeoutError:
                out, lat_ms = "[timeout]", remaining
            api_ms_total += int(lat_ms or 0)
            if out.startswith("[offline]") or out.startswith("[incomplete:") or out.startswith("[timeout]"):
                err_note = out
                continue
            data = _extract_json(out) or {}
            try:
                _idx = data.get("index", data.get("i", None))
                idx = int(_idx) if _idx is not None else None
            except Exception:
                idx = None
            val = data.get("confidence", data.get("c", data.get("p", 0.0)))
            try:
                p = float(val)
            except Exception:
                p = 0.0
            p = max(0.0, min(1.0, p))
            if isinstance(idx, int) and idx in label_map:
                cat = label_map[idx]
                conf = p
                raw_cat = cat
                break
            err_note = f"parse_error:{shape}"
    else:
        # If explicitly disabled responses (shouldn't be per your directive), still run one shot responses
        attempts = 1
        try:
            out, lat_ms = await asyncio.wait_for(
                llm_client.classify_json(cls_prompt, max_tokens=settings.classify_max_tokens, json_schema={}, shape="string"),
                timeout=deadline_ms / 1000.0,
            )
        except asyncio.TimeoutError:
            out, lat_ms = "[timeout]", deadline_ms
        api_ms_total += int(lat_ms or 0)
        if out.startswith("[offline]") or out.startswith("[incomplete:"):
            err_note = out
        else:
            data = _extract_json(out) or {}
            try:
                idx = int(data.get("index", data.get("i", 0)))
            except Exception:
                idx = None
            val = data.get("confidence", data.get("c", data.get("p", 0.0)))
            try:
                p = float(val)
            except Exception:
                p = 0.0
            p = max(0.0, min(1.0, p))
            if isinstance(idx, int) and idx in label_map:
                cat = label_map[idx]
                conf = p
                raw_cat = cat
            else:
                err_note = "parse_error:string"

    # Опционально: fallback на Chat (OpenAI) только если включено
    if not cat and getattr(settings, "fallback_classifier", False):
        cls_model = getattr(settings, "classify_chat_model", "gpt-4o-mini")
        chat_messages = [
            {"role": "system", "content": "Верни строго JSON объект без пояснений."},
            {"role": "user", "content": cls_prompt},
        ]
        out, lat_ms = await llm_client.classify_chat_json(chat_messages, max_tokens=settings.classify_max_tokens, model=cls_model)
        api_ms_total += int(lat_ms or 0)
        data = _extract_json(out) or {}
        try:
            _idx = data.get("index", data.get("i", None))
            idx = int(_idx) if _idx is not None else None
        except Exception:
            idx = None
        val = data.get("confidence", data.get("c", data.get("p", 0.0)))
        try:
            p = float(val)
        except Exception:
            p = 0.0
        p = max(0.0, min(1.0, p))
        if isinstance(idx, int) and idx in label_map:
            cat = label_map[idx]
            conf = p
    # Debug output for classification (hidden by default)
    wall_ms = int((_time.perf_counter() - wall_t0) * 1000)
    if getattr(settings, "debug_verbose", False):
        try:
            debug = {
                "cls_strict": settings.classify_strict,
                "model": getattr(settings, "openai_model", None),
                "cats_from_db": names,
                "prompt": cls_prompt,
                "raw_output": out,
                "parsed": data,
                "raw_category": raw_cat,
                "normalized_category": cat,
                "llm_latency_ms": lat_ms,
                "attempts": attempts,
                "api_ms_total": api_ms_total,
                "wall_ms": wall_ms,
                "path": used_path,
            }
            print("[cls-debug] " + _json.dumps(debug, ensure_ascii=False))
        except Exception:
            pass
    if not cat:
        # Strict: no heuristic; fail clearly
        if getattr(settings, "debug_verbose", False):
            try:
                print("[cls-debug-final] " + _json.dumps({"mode":"strict-fail","reason": err_note or "normalize_empty", "raw": out}, ensure_ascii=False))
            except Exception:
                pass
        raise HTTPException(502, "classifier_failed")
    if getattr(settings, "debug_verbose", False):
        try:
            print("[cls-debug-final] " + _json.dumps({"mode":"llm","final_category": cat, "confidence": conf}, ensure_ascii=False))
        except Exception:
            pass
    return cat, conf, {"attempts": attempts, "api_ms": api_ms_total, "wall_ms": wall_ms}

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
    sys_prompt = db_prompts.get(category) or PROMPTS_BY_TYPE.get(category, "")

    messages = [{"role": "system", "content": SYSTEM_BASE + "\n" + sys_prompt}]
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
        "classify_backend": getattr(settings, "classifier_backend", "responses"),
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


@app.post("/admin/update_stems")
def admin_update_stems(req: Request):
    _require_admin(req)
    from . import db as _db
    updated = _db.update_default_stems()
    app_cache.reload_caches()
    return {"updated_rows": updated}
