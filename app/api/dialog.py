from __future__ import annotations
import uuid
import time as _time
from fastapi import APIRouter, HTTPException
from ..models.schemas import StartDialogRequest, StartDialogResponse, MessageRequest, MessageResponse
from ..services import cache as app_cache
from ..services.llm import llm_client
from ..services.classifier_service import classify_text
from ..core.config import settings
from ..data.messages import add_message, get_session_messages

router = APIRouter(prefix="/dialog", tags=["dialog"])


@router.post("/start", response_model=StartDialogResponse)
def start_dialog(req: StartDialogRequest):
    session_id = str(uuid.uuid4())
    add_message(session_id, "system", app_cache.system_base())
    if req.metadata:
        add_message(session_id, "system", f"meta:{req.metadata}")
    return StartDialogResponse(session_id=session_id)


@router.post("/message", response_model=MessageResponse)
async def send_message(req: MessageRequest):
    session_id, user_text = req.session_id, req.message
    req_t0 = _time.perf_counter()
    history = get_session_messages(session_id)
    if not history:
        raise HTTPException(400, "unknown session_id")
    add_message(session_id, "user", user_text)

    cls_t0 = _time.perf_counter()
    category, confidence, cls_meta = classify_text(user_text)
    cls_wall_ms = int((_time.perf_counter() - cls_t0) * 1000)

    db_prompts = app_cache.classes_prompts_map()
    from ..core.constants import DEFAULT_CLASS_PROMPTS
    sys_prompt = db_prompts.get(category) or DEFAULT_CLASS_PROMPTS.get(category, "")

    messages = [{"role": "system", "content": app_cache.system_base() + "\n" + sys_prompt}]
    for role, content in history[-8:]:
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_text})

    rep_t0 = _time.perf_counter()
    reply, reply_api_ms, cost = await llm_client.chat(messages)
    reply_wall_ms = int((_time.perf_counter() - rep_t0) * 1000)
    if reply.startswith("[offline]"):
        reply = (
            "Извините, сейчас недоступен генеративный ответ. "
            "Мы зафиксировали ваше сообщение и свяжемся с вами.\n\n" f"Техническая справка: {reply}"
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

