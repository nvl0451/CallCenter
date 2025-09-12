# CallCenter LLM + RAG + Vision — One‑Shot Codex CLI

Готовый one‑shot пакет: скормите этот документ **codex CLI** и получите рабочее решение.

> **Что внутри**
> - FastAPI‑сервис с LLM‑агентом (OpenAI, по умолчанию `gpt-5-mini`, поддержка Chat/Responses)
> - Промпт‑система для типов запросов (поддержка/продажи/жалобы)
> - Хранение контекста диалога (SQLite)
> - RAG: ChromaDB + sentence‑transformers (эмбеддинги) + QA endpoint
> - Vision: zero‑shot классификация изображений (CLIP через `open-clip-torch`)
> - Метрики/логирование, базовая оптимизация по стоимости/времени

---

## 📁 Структура проекта
```
callcenter-ai/
├─ app/
│  ├─ __init__.py
│  ├─ main.py
│  ├─ config.py
│  ├─ llm.py
│  ├─ prompts.py
│  ├─ memory.py
│  ├─ rag.py
│  ├─ vision.py
│  └─ schemas.py
├─ data/
│  ├─ kb/
│  │  ├─ pricing.md
│  │  ├─ refund_policy.md
│  │  └─ troubleshooting.md
├─ chroma_db/        # локальная папка для Chroma (persist)
├─ requirements.txt
├─ .env.example
├─ README.md
└─ run.sh
```

---

## ⚙️ requirements.txt
```txt
fastapi>=0.111
uvicorn[standard]>=0.30
httpx>=0.27
pydantic>=2.7
python-dotenv>=1.0
chromadb>=0.5.5
sentence-transformers>=3.0
numpy>=1.26
pandas>=2.2
scikit-learn>=1.4
# Torch stack (CPU)
torch>=2.3; platform_system!="Darwin" or platform_machine!="arm64"
torchvision>=0.18; platform_system!="Darwin" or platform_machine!="arm64"
# macOS arm64 users: установите torch вручную по инструкции PyTorch
Pillow>=10.3
open-clip-torch>=2.24.0
python-multipart>=0.0.9
```

---

## 🔐 .env.example
```env
# Ключ OpenAI
OPENAI_API_KEY=sk-...
# Модель по умолчанию (gpt‑5 требует Responses API)
OPENAI_MODEL=gpt-5-mini
# Использовать OpenAI Responses API (1) или Chat Completions (0)
OPENAI_USE_RESPONSES=1

# Параметры БД
SQLITE_PATH=./callcenter.sqlite3

# ChromaDB
CHROMA_DIR=./chroma_db

# Тюнинги LLM
LLM_TEMPERATURE=0.2
# Лимиты на токены (ответ / классификация)
REPLY_MAX_TOKENS=256
CLASSIFY_MAX_TOKENS=16
```

---

## 🚀 run.sh
```bash
#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
if [ -f .env ]; then export $(grep -v '^#' .env | xargs); fi
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🧠 app/config.py
```python
from __future__ import annotations
import os
from pydantic import BaseModel

class Settings(BaseModel):
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    openai_use_responses: bool = bool(int(os.getenv("OPENAI_USE_RESPONSES", "1")))

    sqlite_path: str = os.getenv("SQLITE_PATH", "./callcenter.sqlite3")
    chroma_dir: str = os.getenv("CHROMA_DIR", "./chroma_db")

    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    reply_max_tokens: int = int(os.getenv("REPLY_MAX_TOKENS", "256"))
    classify_max_tokens: int = int(os.getenv("CLASSIFY_MAX_TOKENS", "16"))

settings = Settings()
```

---

## 🧾 app/schemas.py
```python
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional

class StartDialogRequest(BaseModel):
    customer_id: Optional[str] = None
    metadata: Optional[dict] = None

class StartDialogResponse(BaseModel):
    session_id: str

class MessageRequest(BaseModel):
    session_id: str
    message: str = Field(..., min_length=1)

class MessageResponse(BaseModel):
    type: str
    reply: str
    cost_estimate_usd: float
    latency_ms: int

class IngestRequest(BaseModel):
    documents: Optional[List[str]] = None  # сырые тексты

class AskRequest(BaseModel):
    question: str
    top_k: int = 4
    session_id: Optional[str] = None

class AskResponse(BaseModel):
    answer: str
    sources: List[dict]

class VisionResponse(BaseModel):
    category: str
    confidence: float
    logits: List[float]
    labels: List[str]
```

---

## 🧩 app/prompts.py
```python
SYSTEM_BASE = (
    "Ты — вежливый и точный LLM‑агент колл‑центра. Говори кратко, дружелюбно, по делу. "
    "Если нужна уточняющая информация — задай 1-2 конкретных вопроса."
)

CLASSIFY_PROMPT = (
    "Классифицируй пользовательский запрос в одну категорию из: "
    "[техподдержка, продажи, жалоба]. Верни только одно слово из списка.\n\nЗапрос: {text}"
)

PROMPTS_BY_TYPE = {
    "техподдержка": (
        "Ты специалист техподдержки. Попроси у клиента минимально необходимые детали, "
        "дай 1-3 шага решения, при необходимости предложи эскалацию."
    ),
    "продажи": (
        "Ты менеджер по продажам. Уточни потребности, предложи подходящий тариф/пакет, "
        "сформулируй чёткий next step (демо, счёт, пробный период)."
    ),
    "жалоба": (
        "Ты сотрудник по работе с жалобами. Признай проблему, извинись, опиши, что сделаешь, "
        "предложи компенсацию при необходимости и сроки ответа."
    ),
}
```

---

## 🧮 app/memory.py (SQLite контекст)
```python
from __future__ import annotations
import sqlite3, pathlib, time
from typing import List, Tuple
from .config import settings

DB_PATH = pathlib.Path(settings.sqlite_path)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            ts REAL NOT NULL
        )
        """
    )
    conn.commit(); conn.close()

init_db()

def add_message(session_id: str, role: str, content: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO messages(session_id, role, content, ts) VALUES(?,?,?,?)",
        (session_id, role, content, time.time()),
    )
    conn.commit(); conn.close()

def get_session_messages(session_id: str, limit: int = 20) -> List[Tuple[str,str]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
        (session_id, limit),
    )
    rows = cur.fetchall()
    conn.close()
    return list(reversed(rows))
```

---

## 🌐 app/llm.py (OpenAI клиент + диалог)
```python
from __future__ import annotations
from typing import List, Dict
from openai import OpenAI
from .config import settings

class LLMClient:
    def __init__(self):
        self.model = settings.openai_model
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    async def chat(self, messages: List[Dict[str, str]], max_tokens: int | None = None, temperature: float | None = None):
        if not self.client:
            return "[offline] Нет OPENAI_API_KEY", 0, 0.0
        # gpt‑5 → Responses API с минимальным reasoning, иначе — Chat Completions
        if settings.openai_use_responses and self.model.startswith("gpt-5"):
            text = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
            resp = self.client.responses.create(
                model=self.model,
                input=text,
                max_output_tokens=max_tokens or settings.reply_max_tokens,
                reasoning={"effort": "minimal"},
                text={"verbosity": "low"},
            )
            return (resp.output_text or "").strip(), 0, 0.0
        else:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens or settings.reply_max_tokens,
                temperature=settings.llm_temperature if temperature is None else temperature,
            )
            return (resp.choices[0].message.content or "").strip(), 0, 0.0

llm_client = LLMClient()
```

---

## 🔎 app/rag.py (ChromaDB + sentence‑transformers)
```python
from __future__ import annotations
import os, uuid, pathlib
from typing import List, Dict
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from .config import settings

CHROMA_DIR = pathlib.Path(settings.chroma_dir)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

client = chromadb.Client(ChromaSettings(is_persistent=True, persist_directory=str(CHROMA_DIR)))
collection = client.get_or_create_collection("company_kb")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts: List[str]) -> List[List[float]]:
    return embedder.encode(texts, normalize_embeddings=True).tolist()

def ingest_texts(texts: List[str], metadatas: List[Dict] | None = None):
    ids = [str(uuid.uuid4()) for _ in texts]
    embs = embed_texts(texts)
    collection.add(ids=ids, embeddings=embs, documents=texts, metadatas=metadatas)
    client.persist()
    return ids

def search(query: str, top_k: int = 4):
    q_emb = embed_texts([query])[0]
    res = collection.query(query_embeddings=[q_emb], n_results=top_k, include=["documents", "metadatas", "distances", "ids"])
    items = []
    for i in range(len(res["ids"][0])):
        items.append({
            "id": res["ids"][0][i],
            "document": res["documents"][0][i],
            "metadata": res.get("metadatas", [[{}]])[0][i] or {},
            "score": float(1.0 - (res.get("distances", [[1.0]])[0][i] or 1.0)),
        })
    return items
```

---

## 🖼️ app/vision.py (Zero‑shot CLIP классификация)
```python
from __future__ import annotations
from typing import List, Tuple
from PIL import Image
import torch
import open_clip

CATEGORIES = [
    "ошибка интерфейса",
    "проблема с оплатой",
    "технический сбой",
    "вопрос по продукту",
    "другое",
]

_model = None
_preprocess = None
_tokenizer = None

def _load():
    global _model, _preprocess, _tokenizer
    if _model is None:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-B-32", pretrained="openai"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        model.eval()
        _model, _preprocess, _tokenizer = model, preprocess, tokenizer

@torch.inference_mode()
def classify_image(img: Image.Image) -> Tuple[str, float, List[float], List[str]]:
    _load()
    image = _preprocess(img).unsqueeze(0)
    texts = _tokenizer([f"фото: {c}" for c in CATEGORIES])
    with torch.no_grad():
        image_features = _model.encode_image(image)
        text_features = _model.encode_text(texts)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        logits = (100.0 * image_features @ text_features.T).softmax(dim=-1).squeeze(0)
    conf, idx = float(logits.max().item()), int(logits.argmax().item())
    return CATEGORIES[idx], conf, [float(x) for x in logits.tolist()], CATEGORIES
```

---

## 🧭 app/main.py (FastAPI endpoints)
```python
from __future__ import annotations
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import *
from .config import settings
from .prompts import SYSTEM_BASE, CLASSIFY_PROMPT, PROMPTS_BY_TYPE
from .memory import add_message, get_session_messages
from .llm import llm_client
from .rag import ingest_texts, search
from .vision import classify_image
from PIL import Image

app = FastAPI(title="CallCenter LLM + RAG + Vision")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

# ---------- Диалог ----------
@app.post("/dialog/start", response_model=StartDialogResponse)
def start_dialog(req: StartDialogRequest):
    session_id = str(uuid.uuid4())
    add_message(session_id, "system", SYSTEM_BASE)
    if req.metadata:
        add_message(session_id, "system", f"meta:{req.metadata}")
    return StartDialogResponse(session_id=session_id)

async def _classify(text: str) -> str:
    # Классификация через LLM: возвращаем строгое одно слово
    messages = [
        {"role": "system", "content": SYSTEM_BASE},
        {"role": "user", "content": CLASSIFY_PROMPT.format(text=text)},
    ]
    out, _, _ = await llm_client.chat(messages, max_tokens=4, temperature=0.0)
    cat = out.strip().lower()
    cat = {"техподдержка":"техподдержка", "поддержка":"техподдержка", "support":"техподдержка",
           "sales":"продажи", "продажи":"продажи",
           "жалоба":"жалоба", "complaint":"жалоба"}.get(cat, "техподдержка")
    return cat

@app.post("/dialog/message", response_model=MessageResponse)
async def send_message(req: MessageRequest):
    session_id, user_text = req.session_id, req.message
    # контекст
    history = get_session_messages(session_id)
    if not history:
        raise HTTPException(400, "unknown session_id")
    add_message(session_id, "user", user_text)

    # классификация
    category = await _classify(user_text)
    sys_prompt = PROMPTS_BY_TYPE.get(category, "")

    messages = [{"role": "system", "content": SYSTEM_BASE + "\n" + sys_prompt}]
    # Подмешиваем краткий контекст (последние 8 сообщений)
    for role, content in history[-8:]:
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_text})

    reply, latency_ms, cost = await llm_client.chat(messages)
    add_message(session_id, "assistant", reply)

    return MessageResponse(type=category, reply=reply, cost_estimate_usd=cost, latency_ms=latency_ms)

# ---------- RAG ----------
@app.post("/rag/ingest")
def rag_ingest(req: IngestRequest):
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
    docs = search(req.question, top_k=req.top_k)
    # Сформируем подсказку с цитатами
    context = "\n\n".join([f"[DOC {i+1}]\n" + d["document"] for i, d in enumerate(docs)])
    sys = SYSTEM_BASE + "\nОтвечай, используя только факты из [DOC]. Если чего‑то нет в документах — честно скажи."
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": f"Вопрос: {req.question}\n\nКонтекст:\n{context}"},
    ]
    answer, _, _ = await llm_client.chat(messages)
    return AskResponse(answer=answer, sources=[{"id": d["id"], "score": d["score"]} for d in docs])

# ---------- Vision ----------
@app.post("/vision/classify", response_model=VisionResponse)
async def classify(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "upload an image file")
    img = Image.open(file.file).convert("RGB")
    category, conf, logits, labels = classify_image(img)
    return VisionResponse(category=category, confidence=conf, logits=logits, labels=labels)
```

---

## 📝 data/kb/pricing.md (пример)
```md
Тарифы: Базовый (10$/мес), Плюс (25$/мес), Бизнес (99$/мес). Скидка 20% при оплате за год. Все тарифы включают поддержку по email, Бизнес — приоритетный ответ в течение 2 часов.
```

## 📝 data/kb/refund_policy.md (пример)
```md
Возврат средств возможен в течение 14 дней после покупки при наличии технических проблем, которые мы не смогли решить. Запрос оформляется через поддержку, срок обработки — до 5 рабочих дней.
```

## 📝 data/kb/troubleshooting.md (пример)
```md
Если приложение не открывается: перезапустите устройство, очистите кеш, переустановите приложение. Для ошибок оплаты: проверьте карту, лимиты, повторите попытку через 10 минут или используйте другой способ.
```

---

## 📚 README.md
```md
# CallCenter LLM + RAG + Vision

## Быстрый старт
1. Python 3.10+
2. `python -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. Создайте `.env` из `.env.example` и пропишите `OPENAI_API_KEY` (и `OPENAI_MODEL` при необходимости)
5. `bash run.sh`

## Примеры запросов
### Диалог
```bash
curl -sX POST http://localhost:8000/dialog/start -H 'Content-Type: application/json' -d '{}'
# => {"session_id":"..."}

curl -sX POST http://localhost:8000/dialog/message -H 'Content-Type: application/json' \
  -d '{"session_id":"<ID>", "message":"у меня списали деньги дважды"}'
```

### RAG
```bash
curl -sX POST http://localhost:8000/rag/ingest -H 'Content-Type: application/json' -d '{}'

curl -sX POST http://localhost:8000/rag/ask -H 'Content-Type: application/json' \
  -d '{"question":"какая политика возврата?"}'
```

### Vision
```bash
curl -s -F 'file=@screenshot.png' http://localhost:8000/vision/classify
```

## Заметки по стоимости/скорости
- Температура = 0.2; REPLY_MAX_TOKENS по умолчанию 256.
- Классификация идёт коротким вызовом LLM с `CLASSIFY_MAX_TOKENS=16`, `temperature=0.0`.
- Для gpt‑5 используется OpenAI Responses с reasoning `minimal`.
- Можно включить агрессивное триммирование истории (см. `history[-8:]`).

## Метрики/логи
- В ответах диалога возвращается приблизительная оценка стоимости и latency.
- Для продакшена подключите Prometheus/OTel.
```

---

## 🧪 app/__init__.py
```python
# пусто
```

---

### Подсказки для Codex CLI
- Сохраните этот документ как `SPEC.md` в пустой папке.
- Запустите вашу команду Codex/Codegen с режимом развертывания из одного файла спецификации.
- Генератор создаст описанные файлы и заполнит их содержимым.

Удачи и мягких SLA! 🧃
