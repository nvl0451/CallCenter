from typing import List, Dict
from openai import OpenAI
from .config import settings
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

    def _responses_json_sync(self, prompt: str, max_tokens: int, json_schema: dict, shape: str = "string"):
        import time as _time
        t0 = _time.perf_counter()
        if not self.available or not self.client:
            t1 = _time.perf_counter()
            return "[offline] Нет ключа OPENAI_API_KEY — LLM отключён.", int((t1 - t0) * 1000)
        try:
            if shape == "content":
                _input = [{"type": "input_text", "text": prompt}]
            else:
                _input = prompt
            resp = self.client.responses.create(
                model=self.model,
                input=_input,
                max_output_tokens=max_tokens,
                text={"verbosity": "low"},
                # No reasoning for classifier to minimize latency
            )
            t1 = _time.perf_counter()
            try:
                content = (resp.output_text or "").strip()
            except Exception:
                content = ""
            if not content:
                try:
                    d = resp.to_dict()
                except Exception:
                    d = {}
                if d.get("status") == "incomplete":
                    reason = (d.get("incomplete_details") or {}).get("reason")
                    content = f"[incomplete:{reason or 'unknown'}]"
                if not content and getattr(settings, "debug_verbose", False):
                    try:
                        import json as _json
                        print("[llm-debug] classify empty_output resp= " + _json.dumps(d)[:1500])
                    except Exception:
                        pass
            return content, int((t1 - t0) * 1000)
        except Exception as e:
            t1 = _time.perf_counter()
            return f"[offline] Ошибка обращения к LLM: {e}", int((t1 - t0) * 1000)

    async def classify_json(self, prompt: str, max_tokens: int, json_schema: dict, shape: str = "string"):
        return await asyncio.to_thread(self._responses_json_sync, prompt, max_tokens, json_schema, shape)

    def _classify_chat_sync(self, messages: List[Dict[str, str]], max_tokens: int, model: str):
        import time as _time
        t0 = _time.perf_counter()
        if not self.available or not self.client:
            t1 = _time.perf_counter()
            return "[offline] Нет ключа OPENAI_API_KEY — LLM отключён.", int((t1 - t0) * 1000)
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                temperature=0,
            )
            t1 = _time.perf_counter()
            content = (resp.choices[0].message.content or "").strip()
            return content, int((t1 - t0) * 1000)
        except Exception as e:
            t1 = _time.perf_counter()
            return f"[offline] Ошибка chat classify: {e}", int((t1 - t0) * 1000)

    async def classify_chat_json(self, messages: List[Dict[str, str]], max_tokens: int, model: str):
        return await asyncio.to_thread(self._classify_chat_sync, messages, max_tokens, model)

llm_client = LLMClient()
