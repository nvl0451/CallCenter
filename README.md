# CallCenter LLM + RAG + Vision

## Быстрый старт
1. Python 3.10+
2. `python -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. Создайте `.env` из `.env.example` и пропишите `OPENAI_API_KEY` (и при необходимости `OPENAI_MODEL`)
5. `bash run.sh` (скрипт автоматически активирует `.venv`)

### Частичный запуск (только диалог/классификация)
- По умолчанию RAG и Vision отключены, чтобы не требовать heavy-зависимости.
- Включение фич:
  - `ENABLE_RAG=1` — включает ChromaDB + sentence-transformers
  - `ENABLE_VISION=1` — включает CLIP (torch + open-clip)

Пример `.env` для минимального запуска:
```
OPENAI_API_KEY=sk-your-openai-key
ENABLE_RAG=0
ENABLE_VISION=0
```

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
curl -sX POST http://localhost:8000/rag/ingest -H 'Content-Type: application/json' -d '{}' # требует ENABLE_RAG=1

curl -sX POST http://localhost:8000/rag/ask -H 'Content-Type: application/json' \
  -d '{"question":"какая политика возврата?"}'
```

### Vision
```bash
curl -s -F 'file=@screenshot.png' http://localhost:8000/vision/classify # требует ENABLE_VISION=1
```

Если видите ошибку про `python-multipart`, убедитесь, что пакет установлен в том же интерпретаторе, который запускает uvicorn. Скрипт `run.sh` использует `.venv/bin/uvicorn`, если виртуальное окружение существует.

## Заметки по стоимости/скорости
- Температура = 0.2; max_tokens = 512 — дешёвая, быстрая генерация.
- Классификация — короткий вызов LLM с `max_tokens=16`, `temperature=0.0`, возвращает JSON `{category, confidence}`.
- Можно включить агрессивное триммирование истории (см. `history[-8:]`).

## Метрики/логи
- В ответах диалога возвращается приблизительная оценка стоимости и latency.
- Для продакшена подключите Prometheus/OTel.
