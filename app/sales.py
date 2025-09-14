from __future__ import annotations
import re
from typing import Dict, Any, Tuple

# In-memory sales state by session_id. For production, move to DB.
_STATE: Dict[str, Dict[str, Any]] = {}


def get(session_id: str) -> Dict[str, Any]:
    return _STATE.setdefault(session_id, {
        "stage": "discover",  # discover | recommend | confirm | checkout | quoted
        "intent": None,
        "plan_candidate": None,
        "plan_confidence": 0.0,
        "plan_locked": False,
        "seats": None,
        "channels": [],
        "sso_required": None,
        "analytics_required": None,
        "billing": None,  # monthly | annual
        "admin_email": None,
        "greeted": False,
    })


def update(session_id: str, **fields):
    st = get(session_id)
    st.update({k: v for k, v in fields.items() if v is not None})


def parse_message_slots(text: str) -> Dict[str, Any]:
    t = (text or "").lower()
    out: Dict[str, Any] = {}
    # seats
    m = re.search(r"(\d{1,3})\s*(польз|чел|user|people)", t)
    if m:
        try:
            out["seats"] = int(m.group(1))
        except Exception:
            pass
    else:
        # lone number fallback
        m2 = re.search(r"\b(\d{1,3})\b", t)
        if m2:
            try:
                out["seats"] = int(m2.group(1))
            except Exception:
                pass
    # channels
    ch = []
    if "email" in t or "почт" in t:
        ch.append("email")
    if "чат" in t or "chat" in t:
        ch.append("chat")
    if "whatsapp" in t or "ватсап" in t:
        ch.append("whatsapp")
    if ch:
        out["channels"] = sorted(list(set(ch)))
    # sso / analytics
    if "sso" in t:
        out["sso_required"] = True
    if "аналитик" in t or "analytics" in t:
        out["analytics_required"] = True
    # billing
    if "год" in t or "annual" in t or "year" in t:
        out["billing"] = "annual"
    elif "мес" in t or "month" in t:
        out["billing"] = "monthly"
    # email
    em = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text or "")
    if em:
        out["admin_email"] = em.group(0)
    return out


def recommend_plan(seats: int | None, channels: list[str], sso: bool | None, analytics: bool | None) -> str:
    # Simple heuristic: enterprise needs → Business; multi-channel → Plus; else Basic
    if sso or analytics:
        return "Business"
    if channels and len(channels) > 1:
        return "Plus"
    # fallback by seats
    if seats is not None and seats >= 12:
        return "Business"
    if seats is not None and seats >= 4:
        return "Plus"
    return "Basic"
