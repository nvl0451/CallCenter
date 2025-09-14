from typing import List, Dict
from openai import OpenAI
from ...core.config import settings
import asyncio

class LLMClient:
    def __init__(self):
        self.model = settings.openai_model
        self.available = bool(settings.openai_api_key)
        self.client = OpenAI(api_key=settings.openai_api_key) if self.available else None

    def _chat_sync(self, messages: List[Dict[str, str]], max_tokens: int | None, temperature: float | None):
        import time as _time
        t0 = _time.perf_counter()
        if not self.available or not self.client:
            content = "[offline] Нет ключа OPENAI_API_KEY — LLM отключён."
            t1 = _time.perf_counter()
            return content, int((t1 - t0) * 1000), 0.0

        use_responses = bool(getattr(settings, "openai_use_responses", True)) and self.model.startswith("gpt-5")
        try:
            if use_responses:
                # Use simple string input for Responses API (most compatible)
                text = "\n\n".join(f"{m.get('role','user').upper()}: {m.get('content','')}" for m in messages)
                want_tokens = max_tokens or settings.reply_max_tokens
                if want_tokens < 128:
                    want_tokens = 128  # avoid incomplete for gpt-5 nano family
                resp = self.client.responses.create(
                    model=self.model,
                    input=text,
                    max_output_tokens=want_tokens,
                    text={"verbosity": "low"},
                    reasoning={"effort": "minimal"},
                )
                t1 = _time.perf_counter()
                try:
                    content = (resp.output_text or "").strip()
                except Exception:
                    content = ""
                # Debug summary for Responses
                try:
                    d = resp.to_dict()
                except Exception:
                    d = {}
                if getattr(settings, "debug_verbose", False):
                    try:
                        import json as _json
                        summary = {
                            "status": d.get("status"),
                            "incomplete": (d.get("incomplete_details") or {}).get("reason"),
                            "max_output_tokens": want_tokens,
                            "output_len": len(content),
                        }
                        print("[llm-reply-debug] " + _json.dumps(summary, ensure_ascii=False))
                    except Exception:
                        pass
                if not content:
                    # Fallback to dict walk
                    try:
                        d = resp.to_dict()
                    except Exception:
                        d = {}
                    # Try to extract from output list
                    if not content:
                        for item in (d.get("output") or []):
                            if isinstance(item, dict) and item.get("type") == "message":
                                for part in (item.get("content") or []):
                                    if isinstance(part, dict) and part.get("type") in ("output_text", "text") and isinstance(part.get("text"), str):
                                        content = (part.get("text") or "").strip()
                                        if content:
                                            break
                                if content:
                                    break
                    if not content and d.get("status") == "incomplete":
                        reason = (d.get("incomplete_details") or {}).get("reason")
                        content = f"[incomplete:{reason or 'unknown'}]"
                    # Emit debug if still empty
                    if not content and getattr(settings, "debug_verbose", False):
                        try:
                            import json as _json
                            print("[llm-debug] empty_output resp= " + _json.dumps(d)[:1500])
                        except Exception:
                            pass
                return content or "", int((t1 - t0) * 1000), 0.0
            else:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens or settings.reply_max_tokens,
                    temperature=settings.llm_temperature if temperature is None else temperature,
                )
                t1 = _time.perf_counter()
                content = (resp.choices[0].message.content or "").strip()
                return content, int((t1 - t0) * 1000), 0.0
        except Exception as e:
            t1 = _time.perf_counter()
            return f"[offline] Ошибка обращения к LLM: {e}", int((t1 - t0) * 1000), 0.0

    async def chat(self, messages: List[Dict[str, str]], max_tokens: int = None, temperature: float = None):
        return await asyncio.to_thread(self._chat_sync, messages, max_tokens, temperature)

llm_client = LLMClient()
