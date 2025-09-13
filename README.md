# CallCenter LLM + RAG + Vision (OpenAI)

FastAPI‑сервис с диалогом, RAG и классификацией изображений. По умолчанию — OpenAI.

## Быстрый старт
1) Окружение
- `python -m venv .venv && source .venv/bin/activate`
2) Зависимости
- `pip install -U pip setuptools wheel`
- `pip install -r requirements.txt`
3) Настройка `.env`
- `cp .env.example .env` и укажите `OPENAI_API_KEY`
- (при необходимости) `OPENAI_MODEL`, `OPENAI_USE_RESPONSES`, `REPLY_MAX_TOKENS`, `CLASSIFY_MAX_TOKENS`
- Для RAG: `ENABLE_RAG=1`, `RAG_USE_OPENAI_EMBEDDINGS=1`
- Для Vision: `ENABLE_VISION=1`, `VISION_BACKEND=openai` (или `clip` при наличии torch)
4) Запуск
- `bash run.sh`

Проверка: `curl -s http://localhost:8000/health` → `{ "ok": true }`

## Примеры запросов
- Диалог
  - `SESSION=$(curl -sX POST http://localhost:8000/dialog/start -H 'Content-Type: application/json' -d '{}' | jq -r .session_id)`
  - `curl -sX POST http://localhost:8000/dialog/message -H 'Content-Type: application/json' -d "$(jq -n --arg s "$SESSION" --arg m 'у меня списали деньги дважды' '{session_id:$s, message:$m}')"`

- RAG
  - `curl -sX POST http://localhost:8000/rag/ingest -H 'Content-Type: application/json' -d '{}'`
  - `curl -sX POST http://localhost:8000/rag/ask -H 'Content-Type: application/json' -d '{"question":"какая политика возврата?","top_k":4}'`

- Vision
  - `curl -s -F 'file=@/abs/path/image.png' http://localhost:8000/vision/classify`

- Самодиагностика
  - `curl -s http://localhost:8000/features | jq`

## Параметры
- `OPENAI_API_KEY` — ключ OpenAI
- `OPENAI_MODEL` — чат‑модель (дефолт `gpt-5-mini`)
- `OPENAI_USE_RESPONSES` — 1 для Responses (gpt‑5), 0 для Chat
- `REPLY_MAX_TOKENS` — лимит токенов ответа (дефолт 256)
- `CLASSIFY_MAX_TOKENS` — лимит для классификатора (дефолт 16)
- RAG: `ENABLE_RAG`, `RAG_USE_OPENAI_EMBEDDINGS`, `OPENAI_EMBED_MODEL`, `RAG_CHUNK_CHARS`, `RAG_CHUNK_OVERLAP`
- Vision: `ENABLE_VISION`, `VISION_BACKEND=openai|clip`, `OPENAI_VISION_MODEL`, `VISION_ALLOW_INSECURE_DOWNLOAD`

## Примечания
- gpt‑5 использует Responses API с reasoning `minimal` (fallback `low`). Для самой низкой задержки используйте быстрые Chat‑модели (например, `gpt-4o-mini`) и `OPENAI_USE_RESPONSES=0`.
- Для локального CLIP потребуется установить `torch`, `torchvision`, `open-clip-torch`; на первом запуске скачаются веса.

## Отладка
- Если POST /vision/classify молчит, убедитесь в корректности пути к файлу и используйте `-vS`.
- Состояние фич: `/features`.
