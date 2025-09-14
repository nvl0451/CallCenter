# CallCenter — Admin-First + Tools v2 (Step 2)

Доработанная спецификация для расширенной кастомизации: сначала данные и админ‑CRUD (источники правды для категорий, промптов и RAG‑документов), затем инструменты и оркестрация агента поверх этой модели. Добавлены жизненный цикл RAG‑файлов и тестовая стратегия (латентность/стоимость/галлюцинации).

## 0. Цели и приоритеты

- Сначала CRUD и хранилище: категории, промпты, vision‑лейблы, RAG‑документы + индексация
- Затем агент с tool‑orchestration, читающий категории/промпты из БД
- Тестовая система: бюджет задержки/стоимости, анти‑галлюцинации, мок‑LLM/эмбеддинги
- Защитные лимиты (размеры, таймауты, счётчики) и телеметрия

Вне рамок: UI‑панель (может быть позже), RBAC/мульти‑тенантность (позже). Аутентификация — единый `ADMIN_TOKEN`.

## 1. Текущая база (as‑is)

- FastAPI: диалог/RAG/Vision; OpenAI SDK (Responses для gpt‑5); фич‑флаги
- RAG: ChromaDB + OpenAI/SBERT эмбеддинги; чанкинг
- Vision: OpenAI или локальный CLIP

## 2. Данные и хранилища (источники правды)

### 2.1. Таблицы (SQLite)

```
-- Текстовые категории (для классификатора и выбора промптов)
CREATE TABLE IF NOT EXISTS cls_categories (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,               -- "техподдержка", "продажи", "жалоба", ...
  synonyms_json TEXT NOT NULL DEFAULT '[]',-- ["поддержка","support",...]
  system_prompt TEXT NOT NULL DEFAULT '',  -- персональный системный промпт для этой категории
  priority INTEGER NOT NULL DEFAULT 0,
  active INTEGER NOT NULL DEFAULT 1,
  updated_at REAL NOT NULL DEFAULT (strftime('%s','now'))
);

-- Vision‑лейблы (синонимы и текстовые шаблоны для CLIP)
CREATE TABLE IF NOT EXISTS vision_labels (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  synonyms_json TEXT NOT NULL DEFAULT '[]',
  templates_json TEXT NOT NULL DEFAULT '[]',
  priority INTEGER NOT NULL DEFAULT 0,
  active INTEGER NOT NULL DEFAULT 1,
  updated_at REAL NOT NULL DEFAULT (strftime('%s','now'))
);

-- RAG‑документы (файлы или inline‑тексты)
CREATE TABLE IF NOT EXISTS rag_documents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT NOT NULL,
  kind TEXT NOT NULL CHECK(kind IN('file','inline')),
  rel_path TEXT,                 -- относительный путь в storage для kind='file'
  content_text TEXT,             -- для kind='inline'
  sha256 TEXT NOT NULL,
  bytes INTEGER NOT NULL,
  mime TEXT,
  source TEXT,                   -- например, "admin-upload", "admin-inline"
  active INTEGER NOT NULL DEFAULT 1,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  indexed_at REAL,               -- когда последний раз индексирован
  embed_model TEXT,              -- модель, которой индексировали (для инвалидации)
  chunks_count INTEGER NOT NULL DEFAULT 0,
  dirty INTEGER NOT NULL DEFAULT 1 -- требует (ре)индексации
);
```

Примечания:

- `system_prompt` делает категории источником правды для тональности/персоны — кодовые `PROMPTS_BY_TYPE` становятся fallback.
- `rag_documents` покрывает как загрузки, так и редактируемые inline‑тексты.

### 2.2. Файловое хранилище

- Путь: `./storage/rag/` (настраиваемо: `RAG_STORAGE_DIR`)
- Сохраняем оригиналы по безопасно‑санитезированному пути: `<id>_<slug>.<ext>`; в БД — `rel_path`
- Удаление по умолчанию мягкое (active=0). Физическое удаление — опция (`RAG_HARD_DELETE=0/1`).

### 2.3. Индексация и инвалидация

- Триггеры:
  - Создание/обновление активного документа → `dirty=1`
  - Изменение `OPENAI_EMBED_MODEL` → все активные `dirty=1`
  - `DELETE` (active=0) → удалить соответствующие id из Chroma
- Выполнение:
  - Малые документы (≤1 MB) индексировать синхронно в ответе (админ видит прогресс)
  - Крупные — ставить в очередь и индексировать фоново (упрощённо: выполнять в отдельном потоке)
  - `POST /admin/rag/reindex` — переиндексация всех активных `dirty=1` (батчами)
- Метаданные чанков содержат ссылку на `doc_id` и `embed_model`.

## 3. Админ‑CRUD и кэши

### 3.1. Аутентификация

- `Authorization: Bearer <ADMIN_TOKEN>`; переменная окружения `ADMIN_TOKEN` обязательна для `/admin/*`

### 3.2. Эндпоинты

- Категории (с привязкой промптов):
  - `GET /admin/classes`
  - `POST /admin/classes` — `{name, synonyms:[..], system_prompt, priority?, active?}`
  - `PATCH /admin/classes/{id}`
  - `DELETE /admin/classes/{id}` — `active=0`
- Vision‑лейблы:
  - `GET /admin/vision/labels`
  - `POST /admin/vision/labels` — `{name, synonyms:[..], templates:[..], priority?, active?}`
  - `PATCH /admin/vision/labels/{id}` / `DELETE`
- RAG‑документы:
  - `POST /admin/rag/files` — multipart upload; resp `{id, title}` (синхронная индексация для маленьких)
  - `POST /admin/rag/inline` — создать/обновить inline‑документ `{title, content_text}`
  - `GET /admin/rag/docs` — список активных с флагом `dirty`
  - `DELETE /admin/rag/docs/{id}` — `active=0` (и удалить из индекса)
  - `POST /admin/rag/reindex` — обработать все `dirty=1`

### 3.3. Кэши и использование в рантайме

- Кэш категорий: `{name -> {synonyms, system_prompt, priority}}`
- Кэш vision‑лейблов: `{name -> {synonyms, templates}}`
- Инвалидация после каждого успешного CRUD; загрузка при старте.
- Диалог/классификатор:
  - Если БД непуста → использовать категории и `system_prompt` из БД
  - Иначе fallback: текущие 3 категории + встроенные промпты
- Vision/CLIP использует лейблы и шаблоны из БД, иначе — дефолт.

## 4. Агент и инструменты (поверх CRUD)

### 4.1. Инструменты

- `classify_text(text) -> {category, confidence}` (основано на БД‑категориях)
- `rag_search(question, top_k=4) -> [{id, snippet, score}]` (`snippet` = первые 800 симв.)
- `vision_classify(image_b64) -> {category, confidence, labels}`

### 4.2. `/agent/message`

- Вход: `{session_id?, text?, image_b64?}`
- Алгоритм:
  - Создать/найти `session_id`
  - Системный prompt = общий стиль + (если есть category) соответствующий `system_prompt` из БД
  - Интеграция OpenAI: Responses tools (gpt‑5) или Chat functions
  - Лимиты: ≤3 tool‑вызова, tool‑таймаут 8с, общий ≤20с
  - Ответ: `{session_id, tools_used:[..], reply, sources?}`

## 5. Безопасность, лимиты, хранение

- `ADMIN_TOKEN` обязателен на `/admin/*`
- Ограничения:
  - `MAX_UPLOAD_MB` (деф. 8)
  - Расширения: `.txt .md` (PDF позже)
  - Тул‑лимиты: как выше
- Хранилище: `RAG_STORAGE_DIR=./storage/rag`; `RAG_HARD_DELETE=0|1`

## 6. Тестирование и качества (latency/$/hallucinations)

- Pytest‑пакет без тяжёлых зависимостей
- Моки/стабы:
  - `OPENAI_MOCK=1` → LLM/embeddings/vision заменяются на детерминированные стабы
  - Для Chroma — локальный временный каталог; для эмбеддингов — стабильный хэш‑вектор
- Бюджеты и ассёрты:
  - `BUDGET_MS_DIALOG`/`BUDGET_MS_RAG` — средняя/перцентильная задержка в мок‑режиме (например, <50ms)
  - `BUDGET_COST_USD_REQ` — в реальных e2e (skipped по умолчанию) проверять, что usage/cost ниже порога
  - Анти‑галлюцинации: для вопроса вне KB ответ должен содержать оговорку ("нет в документах")
- Снэпшоты/голдены для классификатора: JSON‑формат строгий, confidence ∈ [0,1]
- Скрипты:
  - `scripts/smoke_latency.py` — прогоны /dialog, /rag/ask с измерением задержек
  - `scripts/load_reindex.py` — загрузка 5–20 файлов, reindex, затем /rag/ask

## 7. Переменные окружения (новые)

- `ADMIN_TOKEN`
- `MAX_UPLOAD_MB=8`
- `RAG_STORAGE_DIR=./storage/rag`
- `RAG_HARD_DELETE=0`
- `OPENAI_MOCK=0` (включить тестовые стабы)
- `OPENCLIP_MODEL=ViT-B-32`, `OPENCLIP_CACHE_DIR=./.openclip_cache`

## 8. Телеметрия и /features

- Расширить `/features`:
  - `tools: {enabled, max_calls}`
  - размеры кэшей (classes/labels)
  - rag: число активных документов, dirty‑count
- Логи: структурные записи при tool‑вызовах `{tool, ms, ok}` и при индексации `{doc_id, chunks, ms, ok}`

## 9. План внедрения (итерации)

1. Миграции БД, кэши, CRUD endpoints, файловое хранилище
2. Жизненный цикл RAG: индексация/инвалидация/удаление; reindex батчи; расширение `/features`
3. Тестовый контур: pytest, стабы, smoke‑скрипты; базовые бюджеты
4. Агент и tools поверх БД‑категорий и RAG
5. Документация и (опционально) простая админ‑страница позже

## 10. API‑примеры

### 10.1. /admin/classes (POST)

```json
{
  "name": "техподдержка",
  "synonyms": ["поддержка", "support"],
  "system_prompt": "Ты специалист техподдержки...",
  "priority": 10
}
```

### 10.2. /admin/rag/inline (POST)

```json
{ "title": "Политика возврата", "content_text": "Возврат средств..." }
```

### 10.3. /agent/message (POST)

```json
{ "text": "у меня списали деньги дважды" }
```

Ответ:

```json
{
  "session_id": "...",
  "tools_used": ["classify_text", "rag_search"],
  "reply": "...",
  "sources": [{ "id": "...", "score": 0.81 }]
}
```

## 11. Риски/заметки

- Responses‑модели могут быть медленнее → токен‑кап и минимальный reasoning
- CLIP на CPU медленный; по умолчанию Vision через OpenAI
- SQLite подходит для одного процесса; в многопроцессе нужен внешний DB или единичный воркер

---
