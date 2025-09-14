# CallCenter Platform: LLM, RAG, Vision и SBERT для Умного Колл-Центра

Добро пожаловать в **CallCenter Platform** — мощную FastAPI-платформу, интегрирующую передовые AI-технологии для автоматизации колл-центра. Это полноценная экосистема с LLM-агентами (OpenAI по умолчанию), RAG для поиска по знаниям, Vision для анализа изображений, SBERT для локальной классификации и эмбеддингов, а также прототипом **Negotiator** — умным sales-агентом с кастомными инструментами.

Negotiator фокусируется на технике воронки продаж: использует "мягкие скрипты" для реактивного динамического общения — базовый скрипт для уверенных пользователей (decisive path: быстрый pitch, цена, CTA), и мягкую guidance для неуверенных (guided path: 2 вопроса, рекомендация плана). Всё настраивается через admin API, с DB-driven промптами/лейблами в SQLite, resilient RAG из DOCS_DIR (Chroma + OpenAI/SBERT embeddings) и graceful handling отключённых фич.

Платформа production ready: separation of concerns (api/services/backends/data), smoke-тесты, телеметрия (/features), Telegram-бот для demo Negotiator. Scope заморожен (см. SCOPE_FREEZE.md), дальнейшая работа — advanced sales и persistence.

## Быстрый Старт

1. **Окружение**  
   `python -m venv .venv && source .venv/bin/activate`

2. **Зависимости**  
   `pip install -U pip setuptools wheel`  
   `pip install -r requirements.txt`  
   (Для SBERT/CLIP: добавьте `sentence-transformers torch torchvision open-clip-torch`)

3. **Настройка `.env`**  
   `cp .env.example .env` и заполните:

   - `OPENAI_API_KEY` (обязательно для LLM/RAG)
   - `ADMIN_TOKEN` (для /admin/\*)
   - `TELEGRAM_BOT_TOKEN` (для tg bot demo)
   - Флаги: `ENABLE_RAG=1`, `ENABLE_VISION=1`, `CLASSIFIER_BACKEND=sbert` (локальный SBERT или responses)

4. **Запуск**  
   `bash run.sh`  
   Сервер на http://localhost:8000

Проверка:  
`curl -s http://localhost:8000/health` → `{"ok": true}`  
`/features` — флаги, метрики RAG (active docs, dirty count), кэши (classes/labels), tools budget.

## API Эндпоинты

### Диалог (/dialog/\*)

Базовый LLM-чат с классификацией (SBERT) и контекстом из SQLite.

- Start: `curl -sX POST http://localhost:8000/dialog/start -H 'Content-Type: application/json' -d '{}'` → `{"session_id": "..."}`
- Message: `curl -sX POST http://localhost:8000/dialog/message -H 'Content-Type: application/json' -d '{"session_id":"...", "message":"у меня списали деньги дважды"}'`  
  Ответ: `{"type": "жалоба", "reply": "...", "cost_estimate_usd": 0.001, "latency_ms": 150}`

### RAG (/rag/\*)

Retrieval-Augmented Generation: ingest из DOCS_DIR, search from sources.

- Ingest: `curl -sX POST http://localhost:8000/rag/ingest -d '{}'` (загружает KB файлы)
- Ask: `curl -sX POST http://localhost:8000/rag/ask -H 'Content-Type: application/json' -d '{"question":"политика возврата?","top_k":4}'`  
  Ответ: `{"answer": "...", "sources": [{"id": "...", "score": 0.85, "snippet": "..."}]}`

### Vision (/vision/classify)

Zero-shot классификация изображений (локальный CLIP с DB-лейблами).  
`curl -s -F 'file=@image.png' http://localhost:8000/vision/classify`  
Ответ: `{"category": "ui_error", "confidence": 0.92, "logits": [...], "labels": [...]}`

### Negotiator (/agent/message)

Прототип sales-агента: классифицирует intent, использует tools (classify_text, rag_search, vision_classify; ≤3/turn, ≤20s total). Мягкие скрипты в зависимости от считанной BERT уверенности намерений пользователя: decisive (pitch + CTA) или guided (вопросы + recommend). In-memory state card (slots: intent, plan, seats, email). RAG sources в ответах.  
`curl -sX POST http://localhost:8000/agent/message -H 'Content-Type: application/json' -d '{"text":"интересует бизнес-тариф"}'`  
Ответ: `{"session_id": "...", "tools_used": ["classify_text", "rag_search"], "reply": "Для бизнеса рекомендую план Business...", "sources": [...]}`

См. NEGOTIATOR.md для полного видения агента Negotiator - в текущей форме он представлен как прототип

### Admin API (/admin/\*; Bearer ADMIN_TOKEN)

CRUD и lifecycle: редактируйте промпты/лейблы в DB, инвалидация кэшей auto.

- Classes: `GET/POST/PUT/DELETE /admin/classes` (категории: name, synonyms, system_prompt)
- Vision: `GET/POST/PUT/DELETE /admin/vision` (labels: synonyms, templates)
- RAG Docs: `POST /admin/rag/files` (upload), `POST /admin/rag/inline` (текст), `GET /admin/rag/docs` (с dirty), `DELETE /admin/rag/docs/{id}`
- Lifecycle: `POST /admin/rag/reset_index`, `POST /admin/rag/sync_docs` (из DOCS_DIR), `POST /admin/rag/reindex_dirty`
- Settings: `GET/PUT /admin/settings` (system_base), `POST /admin/bootstrap` (seed defaults)

## Архитектура

Чистый модульный дизайн для AI-агентов и людей (см. AGENTS.md):

- **api/**: Routers (no business logic) — health, features, dialog, rag, vision, agent, admin.
- **services/**: Facades — cache (DB-backed), classifier (SBERT-first + OpenAI fallback), rag (ingest/search/reset), vision (OpenAI/CLIP), llm (OpenAI client), sales (negotiator slots/state).
- **backends/**: Impl — llm_backend (OpenAI), rag_backend (Chroma + OpenAI/SBERT embeddings), vision_backend (DB labels), classifier_backend (SBERT/responses), sales_backend (in-memory sessions, tool orchestration).
- **data/**: Repos (pure SQL: classes_repo, vision_labels_repo, rag_docs_repo), db (SQLite migrations), bootstrap (seed defaults on first boot).
- **core/**: Startup (migrations → bootstrap → cache warmup → routers), config/constants (env-driven, re-export).
- **models/**: Pydantic schemas для API.

Data flow: Runtime reads from DB via caches (no fallbacks). RAG: ingest → dirty flag → reindex (batches, metadata with doc_id/embed_model). Tools: JSON-strict, budgets enforced. Sessions: in-memory for negotiator (future: DB). SBERT for локальной classification/embeddings (device: cpu/gpu). Всё resilient: no crashes если фичи off.

## Тестирование и Demo

Smoke scripts (bash/python; OPENAI optional — offline mode):

- `scripts/run_crud_strict.sh` — admin CRUD (classes/vision/rag) + checks.
- `scripts/run_reindex_strict.sh` — RAG reset/sync/reindex + label dynamics.
- `scripts/run_agent_purchase.sh` — negotiator paths (decisive/unsure) + sources.
- `scripts/run_negotiator_smoke.sh` — 3-step funnel (discovery → plan → quote/email; no early quote).
- `scripts/run_demo_strict.sh` — full e2e (dialog + rag + vision).

**Telegram Bot Demo для Negotiator**:

- `bash scripts/run_tg.sh` — запускает бота (нужен TELEGRAM_TOKEN в .env).
- Или `python scripts/run_telegram_bot.py` — интерактивный чат с агентом в TG. Идеально для testing sales flows.

## Параметры Окружения (ключевые)

- **LLM**: `OPENAI_API_KEY`, `OPENAI_MODEL` (def: gpt-4o-mini), `OPENAI_USE_RESPONSES` (1 для gpt-4o с reasoning minimal), `REPLY_MAX_TOKENS=256`, `LLM_TEMPERATURE=0.2`.
- **RAG**: `ENABLE_RAG=1`, `RAG_USE_OPENAI_EMBEDDINGS=1` (или SBERT), `OPENAI_EMBED_MODEL=text-embedding-3-small`, `CHROMA_DIR=./chroma`, `DOCS_DIR=./data/kb`, `RAG_STORAGE_DIR=./storage/rag`, `RAG_HARD_DELETE=0`.
- **Vision**: `ENABLE_VISION=1`, `VISION_BACKEND=openai|clip`, `OPENAI_VISION_MODEL=gpt-4o-mini`, `VISION_ALLOW_INSECURE_DOWNLOAD=0`.
- **Classifier**: `CLASSIFIER_BACKEND=responses|sbert` (SBERT для local/low-latency), `SBERT_MODEL=all-MiniLM-L6-v2`, `SBERT_DEVICE=cpu|cuda`.
- **Admin/DB**: `ADMIN_TOKEN` (Bearer для /admin), `SQLITE_PATH=./app.db`.
- **Tools**: Budgets ≤3 calls/turn, 8s/tool, 20s total (enforced в backends).

См. .env.example для полного списка.

## Будущая Работа

- DB-persistence для negotiator sessions (сейчас in-memory).
- Advanced sales tools: haggling, plan_compare, discount_policy, compliance_info.
- Full pytest suite с mocks (OPENAI_MOCK), load testing.
- Admin UI (React?), RBAC/multi-tenant.
- Интеграция с telephony (Twilio?), analytics dashboard.

## Примечания и Отладка

- **Стоимость/Скорость**: gpt-4o-mini + SBERT/CLIP для low-latency (Responses minimal reasoning). Tools budgets минимизируют calls. /features показывает metrics.
- **Offline Mode**: Без OPENAI_KEY — graceful "offline" replies; SBERT/CLIP работают locally.
- **RAG Lifecycle**: Sync из DOCS_DIR auto на boot; reindex dirty docs (triggers: CRUD, embed_model change). Sources с {id, score, snippet} в ответах.
- **Отладка**:
  - Vision silent? Проверьте file path, используйте `-v` в curl.
  - Admin 401? Добавьте `-H "Authorization: Bearer $ADMIN_TOKEN"`.
  - Logs: Structured (tools ms/ok, slots changes, latencies).
  - Фичи: `/features` для snapshots (flags, caches, rag stats).

Исходники: SCOPE_FREEZE.md (scope), NEGOTIATOR.md (agent guide), SPEC_2.md (admin-first spec), AGENTS.md (dev rules).  
Epic платформа для AI-driven call centers — fork, contribute, scale!
