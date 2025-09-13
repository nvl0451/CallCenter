#!/usr/bin/env python3
import os, json, time, textwrap
from pathlib import Path
from typing import Dict
import logging

import httpx
from telegram import Update
from dotenv import load_dotenv
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters


load_dotenv()  # load .env if present

def _env_first(*keys: str, default: str = "") -> str:
    for k in keys:
        v = os.getenv(k)
        if v:
            return v
    return default

API_BASE = _env_first("API_BASE", default="http://localhost:8000")
BOT_TOKEN = _env_first("TELEGRAM_BOT_TOKEN", "TELEGRAM_TOKEN", "BOT_TOKEN", default="")
SESS_FILE = os.getenv("TG_SESS_FILE", ".tg_sessions.json")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tg-bot")


def load_sessions() -> Dict[str, str]:
    p = Path(SESS_FILE)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def save_sessions(mp: Dict[str, str]):
    try:
        Path(SESS_FILE).write_text(json.dumps(mp))
    except Exception:
        pass


SESSIONS: Dict[str, str] = load_sessions()


async def _post_json(path: str, payload: dict) -> dict:
    url = API_BASE.rstrip("/") + path
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        return r.json()


def _chunk(text: str, n: int = 3800):
    text = text or ""
    while text:
        yield text[:n]
        text = text[n:]


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    SESSIONS.pop(chat_id, None)
    save_sessions(SESSIONS)
    msg = (
        "Привет! Я помогу подобрать тариф и оформить подписку. "
        "Напишите, что вам нужно — например: ‘хочу оформить Plus на 8 человек’."
    )
    await update.message.reply_text(msg)


async def cmd_features(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(API_BASE.rstrip("/") + "/features")
            d = r.json()
        rag = d.get("rag", {})
        clf = d.get("classifier", {})
        tools = d.get("tools", {})
        txt = (
            f"RAG: enabled={rag.get('enabled')} available={rag.get('available')}\n"
            f"Classifier: backend={clf.get('backend')}\n"
            f"Tools: max_calls={tools.get('max_calls')} total_budget_s={tools.get('total_budget_s')}"
        )
    except Exception as e:
        txt = f"Не удалось получить /features: {e}"
    await update.message.reply_text(txt)


async def cmd_reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    SESSIONS.pop(chat_id, None)
    save_sessions(SESSIONS)
    await update.message.reply_text("Сессия сброшена. Напишите сообщение, чтобы начать сначала.")


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    text = update.message.text or ""
    sess = SESSIONS.get(chat_id)
    payload = {"text": text}
    if sess:
        payload["session_id"] = sess
    try:
        t0 = time.perf_counter()
        resp = await _post_json("/agent/message", payload)
        t1 = time.perf_counter()
    except httpx.HTTPStatusError as e:
        await update.message.reply_text(f"Ошибка API: {e.response.status_code}")
        return
    except Exception as e:
        await update.message.reply_text(f"Сбой: {e}")
        return

    sid = resp.get("session_id")
    if sid:
        SESSIONS[chat_id] = sid
        save_sessions(SESSIONS)

    reply = resp.get("reply") or ""
    meta = resp.get("latencies") or {}
    llm_fb = bool(resp.get("llm_fallback"))
    tail = f"\n\n— ⏱ llm={meta.get('llm_ms','?')}ms rag={meta.get('rag_ms','?')}ms" + (" (fallback)" if llm_fb else "")
    out = reply + tail
    for chunk in _chunk(out):
        await update.message.reply_text(chunk, disable_web_page_preview=True)


def main():
    if not BOT_TOKEN:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN env var")
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("features", cmd_features))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), on_text))
    log.info("Bot up. API_BASE=%s", API_BASE)
    app.run_polling()


if __name__ == "__main__":
    main()
