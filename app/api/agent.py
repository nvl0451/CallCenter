from __future__ import annotations
from fastapi import APIRouter, HTTPException
import uuid
import time as _t
from ..models.schemas import AgentMessageRequest, AgentMessageResponse
from ..services.classifier_service import classify_text, classify_among
from ..services.rag_service import rag_available, search
from ..services.llm import llm_client
from ..services import cache as app_cache
from ..core.config import settings
from ..data.messages import add_message, get_session_messages
from ..services import sales as sales

router = APIRouter(prefix="/agent", tags=["agent"])

# Prototype sales funnel Negotiator agent


@router.post("/message", response_model=AgentMessageResponse)
async def agent_message(req: AgentMessageRequest):
    t0 = _t.perf_counter()
    tools_used: list[str] = []
    max_calls = 3
    calls = 0
    session_id = req.session_id or str(uuid.uuid4())
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "text required")
    st_before = sales.get(session_id)
    try:
        slot_guess = sales.parse_message_slots(text)
    except Exception:
        slot_guess = {}
    if slot_guess:
        sales.update(session_id, **slot_guess)
    st = sales.get(session_id)
    material_keys = {k for k in (slot_guess or {}).keys() if k in ("seats", "channels", "sso_required", "analytics_required", "billing")}
    new_email = (st_before.get("admin_email") is None and bool(st.get("admin_email")))
    new_billing = (st_before.get("billing") not in ("monthly", "annual") and st.get("billing") in ("monthly", "annual"))
    try:
        add_message(session_id, "user", text)
    except Exception:
        pass

    def budget_ok():
        return calls < max_calls and ((_t.perf_counter() - t0) < 20.0)

    intent = None
    try:
        if budget_ok():
            calls += 1; tools_used.append("classify_text")
            intent, _, _m = classify_text(text)
    except Exception:
        intent = None
    if intent:
        sales.update(session_id, intent=intent)

    plan = None; plan_conf = 0.0
    locked = bool(st.get("plan_locked"))
    if locked and not material_keys:
        plan = st.get("plan_candidate"); plan_conf = float(st.get("plan_confidence") or 0.9)
    elif intent == "продажи" and budget_ok():
        try:
            best, conf = classify_among(text, ["Basic plan", "Plus plan", "Business plan"])
            plan = best.split()[0]; plan_conf = conf
        except Exception:
            low = text.lower()
            if any(k in low for k in ["basic", "starter", "базов"]):
                plan, plan_conf = "Basic", 0.6
            elif any(k in low for k in ["plus", "плюс", "pro"]):
                plan, plan_conf = "Plus", 0.6
            elif any(k in low for k in ["business", "бизнес", "enterprise"]):
                plan, plan_conf = "Business", 0.6

    if plan:
        sales.update(session_id, plan_candidate=plan, plan_confidence=plan_conf)
    try:
        st_now = sales.get(session_id)
        if (not plan or plan_conf < 0.65) and intent == "продажи":
            rec = sales.recommend_plan(st_now.get("seats"), st_now.get("channels") or [], st_now.get("sso_required"), st_now.get("analytics_required"))
            plan = rec; plan_conf = max(plan_conf, 0.7)
            sales.update(session_id, plan_candidate=plan, plan_confidence=plan_conf)
    except Exception:
        pass

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

    reply = ""; sources = None; llm_fallback = False; llm_ms = 0
    price_ms = 0; quote_ms = 0; quote_obj = None

    def _extract_plan_section(txt: str, plan_en: str, plan_ru: str) -> str:
        import re as _re
        t = txt or ""
        patterns = [rf"^##\s*{plan_ru}.*$[\s\S]*?(?=^##\s|\Z)", rf"^##\s*{plan_en}.*$[\s\S]*?(?=^##\s|\Z)"]
        for pat in patterns:
            m = _re.search(pat, t, flags=_re.MULTILINE)
            if m:
                return m.group(0)
        return t

    if plan and plan_conf >= 0.65:
        ru = {"Basic":"Базовый","Plus":"Плюс","Business":"Бизнес"}.get(plan, plan)
        import re as _re
        sections = []
        if docs:
            for d in docs:
                raw = d.get("document", "") or ""
                sec = _extract_plan_section(raw, plan, ru)
                sec_lines = [ln for ln in sec.splitlines() if not _re.search(r"(?i)(вс.? из базов|everything in basic)", ln)]
                sec_clean = "\n".join(sec_lines)
                sections.append(sec_clean[:1500])
        context = "\n\n".join(sections)
        included_hint = {"Basic": "(до 3 пользователей включено)", "Plus": "(до 10 пользователей включено)", "Business": "(до 25 пользователей включено)"}.get(plan, "")
        st_now_for_prompt = sales.get(session_id)
        greeted = bool(st_now_for_prompt.get("greeted"))
        has_billing = st_now_for_prompt.get("billing") in ("monthly", "annual")
        has_email = bool(st_now_for_prompt.get("admin_email"))
        slot_header = []
        if st_now_for_prompt.get("seats"):
            slot_header.append(f"Пользователи: {st_now_for_prompt.get('seats')}")
        if st_now_for_prompt.get("channels"):
            slot_header.append("Каналы: " + ", ".join(st_now_for_prompt.get("channels") or []))
        if st_now_for_prompt.get("billing"):
            slot_header.append("Оплата: " + ("месяц" if st_now_for_prompt.get("billing")=="monthly" else "год"))
        slot_header_txt = ("; ".join(slot_header)) if slot_header else ""
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
            sys = app_cache.system_base() + ("\nСтиль: тёплый, дружелюбный, уверенный менеджер по продажам мужского пола. Пиши по‑русски, живым человеческим языком (без канцеляризмов и англицизмов)."
                "\nФормат ответа:" + ("\n— Короткий приветственный лид (без общих рассуждений). " + (slot_header_txt if slot_header_txt else ""))
                + ("\n— Цена: 1 короткая фраза." if docs else "")
                + ("\n— Ключевые выгоды плана: 3–5 маркеров только при наличии [DOC]." if docs else "")
                + "\n— Заверши дружелюбной фразой‑призывом: " + cta_text + " (не используй слова ‘Следующий шаг’)."
                + ("\nИспользуй только факты из [DOC]." if docs else ""))
        else:
            sys = app_cache.system_base() + ("\nСтиль: тёплый, дружелюбный, уверенный менеджер по продажам мужского пола. Пиши по‑русски, живым человеческим языком (без канцеляризмов и англицизмов)."
                "\nФормат ответа:" + ("\n— Коротко подтвердить соответствие плана без приветствия." + ("\n— Цена: 1 короткая фраза." if docs else ""))
                + "\n— Заверши дружелюбной фразой‑призывом: " + cta_text + " (без повторения выгод).")
        convo = [{"role": "system", "content": sys}]
        try:
            hist = get_session_messages(session_id)
        except Exception:
            hist = []
        for role, content in hist[-8:]:
            if role in ("user", "assistant"):
                convo.append({"role": role, "content": content})
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
        if not greeted:
            try:
                sales.update(session_id, greeted=True)
            except Exception:
                pass
        def _src_item(d: dict) -> dict:
            md = d.get("metadata") or {}
            doc = (d.get("document", "") or "").replace("\n", " ")
            import re as _re
            doc = _re.sub(r"\s+", " ", doc).strip()
            snip = doc[:240]
            if len(doc) > 240:
                cut = snip.rfind(" ")
                if cut > 160:
                    snip = snip[:cut] + "…"
            return {"score": d.get("score"), "id": d.get("id"), "source": md.get("source"), "doc_id": md.get("doc_id"), "snippet": snip}
        sources = [_src_item(d) for d in docs]
        try:
            st_now = sales.get(session_id)
            if st_now.get("admin_email") and st_now.get("billing") in ("monthly", "annual") and calls < max_calls:
                calls += 1; tools_used.append("price_calc")
                _pt0 = _t.perf_counter()
                monthly = {"Basic":29.0, "Plus":99.0, "Business":299.0}.get(plan, 0.0)
                if st_now.get("billing") == "annual":
                    base = monthly * 12 * 0.8
                else:
                    base = monthly
                total = base
                price_ms = int((_t.perf_counter() - _pt0) * 1000)
                import uuid as _uuid
                calls += 1; tools_used.append("create_quote")
                _qt0 = _t.perf_counter()
                qid = str(_uuid.uuid4())
                quote_obj = {"quote_id": qid, "url": f"https://example.local/quote/{qid}", "plan": plan, "billing": st_now.get("billing"), "seats": st_now.get("seats"), "total_usd": round(total, 2), "sent_to": st_now.get("admin_email")}
                quote_ms = int((_t.perf_counter() - _qt0) * 1000)
                reply = (reply + "\n\n— Оформил предложение: " + quote_obj["url"] + f" (отправил на {quote_obj['sent_to']}).")
                sales.update(session_id, stage="quoted", plan_locked=True)
            else:
                sales.update(session_id, stage="confirm", plan_locked=True)
        except Exception:
            pass
    else:
        # Guided path
        greeted = bool(sales.get(session_id).get("greeted"))
        if not greeted:
            sys = (app_cache.system_base() + "\nСтиль: тёплый, дружелюбный, уверенный менеджер по продажам мужского пола. Только по‑русски, без эмодзи и иностранных символов.\nФормат ответа:\n— Короткая приветственная фраза (1 строка).\n— Два уточняющих вопроса (каждый с маркером «— …», очень коротко).\n— Заверши 1 дружелюбной фразой‑призывом (не используй слова ‘Следующий шаг’), например: ‘Ответьте — подберу план и пришлю ссылку на оплату/счёт сегодня’.")
        else:
            sys = (app_cache.system_base() + "\nСтиль: тёплый, дружелюбный, уверенный менеджер по продажам мужского пола. Только по‑русски, без эмодзи и иностранных символов.\nФормат ответа:\n— Два уточняющих вопроса (каждый с маркером «— …», очень коротко).\n— Заверши 1 дружелюбной фразой‑призывом (не используй слова ‘Следующий шаг’), например: ‘Ответьте — подберу план и пришлю ссылку на оплату/счёт сегодня’.")
        user_msg = ("Клиент не уверен в плане (Basic/Plus/Business). Задай ровно 2 лаконичных вопроса, которые помогут выбрать план по числу пользователей, каналам (email/чат/WhatsApp) и требованиям (SSO/аналитика). Затем предложи один следующий шаг. Текст клиента: " + text)
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
            reply = ("1) Вопрос 1: Сколько будет пользователей?\n2) Вопрос 2: Какие каналы нужны (email/чат/WhatsApp)? Нужны ли SSO/аналитика?\nСледующий шаг: после ответа предложу подходящий план и ссылку на оформление.")
            llm_fallback = True
        try:
            if not greeted:
                sales.update(session_id, greeted=True)
        except Exception:
            pass

    lat = {"rag_ms": rag_ms, "llm_ms": llm_ms, "price_ms": price_ms, "quote_ms": quote_ms, "total_ms": int((_t.perf_counter() - t0) * 1000)}
    st_out = sales.get(session_id)
    try:
        add_message(session_id, "assistant", reply or "")
    except Exception:
        pass
    return AgentMessageResponse(session_id=session_id, tools_used=tools_used, reply=reply or "", sources=sources, latencies=lat, llm_fallback=llm_fallback, stage=st_out.get("stage"), slots={k: st_out.get(k) for k in ("seats","channels","billing","admin_email","plan_candidate")}, quote=quote_obj)

