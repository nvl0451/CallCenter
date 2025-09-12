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
                # Flatten conversation into single input string
                text = "\n\n".join(f"{m.get('role','user').upper()}: {m.get('content','')}" for m in messages)
                # First attempt with explicitly requested minimal reasoning
                try:
                    resp = self.client.responses.create(
                        model=self.model,
                        input=text,
                        max_output_tokens=max_tokens or settings.reply_max_tokens,
                        reasoning={"effort": "minimal"},
                        text={"verbosity": "low"},
                    )
                except Exception as e_min:
                    # If the backend rejects "minimal", fall back to "low" transparently
                    msg = str(e_min)
                    if "Invalid value" in msg and "effort" in msg:
                        resp = self.client.responses.create(
                            model=self.model,
                            input=text,
                            max_output_tokens=max_tokens or settings.reply_max_tokens,
                            reasoning={"effort": "low"},
                            text={"verbosity": "low"},
                        )
                    else:
                        raise
                t1 = _time.perf_counter()
                try:
                    content = (resp.output_text or "").strip()
                except Exception:
                    # Fallback to dict walk
                    d = resp.to_dict()
                    content = str(d.get("output_text") or "").strip()
                    if not content:
                        for item in d.get("output") or []:
                            if isinstance(item, dict) and item.get("type") == "message":
                                for part in item.get("content") or []:
                                    if isinstance(part, dict) and part.get("type") in ("output_text", "text") and isinstance(part.get("text"), str):
                                        content = part.get("text").strip()
                                        break
                                if content:
                                    break
                    if not content and d.get("status") == "incomplete":
                        reason = (d.get("incomplete_details") or {}).get("reason")
                        content = f"[incomplete:{reason or 'unknown'}]"
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
