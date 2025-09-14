# Agent Guide — CallCenter LLM + RAG + Vision

This repository is organized for AI coding agents and humans to work productively. It cleanly separates API routing, services, backends, data access, and startup/bootstrapping. Defaults are used only to seed the DB; runtime always reads from the DB via caches.

## Directory Map

- app/
  - main.py — thin entrypoint creating the FastAPI app
  - api/ — all FastAPI routers (no business logic)
    - health.py → GET /health
    - features.py → GET /features
    - dialog.py → POST /dialog/start, /dialog/message
    - rag.py → POST /rag/ingest, /rag/ask
    - vision.py → POST /vision/classify
    - agent.py → POST /agent/message (sales negotiator prototype)
    - admin/ (split by domain)
      - classes.py → /admin/classes CRUD; /admin/update_stems
      - vision.py → /admin/vision CRUD
      - rag_docs.py → /admin/rag/* (inline/file/docs/reindex/sync/reset)
      - settings.py → /admin/settings (system_base GET/PUT), /admin/bootstrap
  - services/ — stable service layer used by API
    - cache.py — DB-backed caches: classes, vision labels, system_base
    - classifier_service.py — SBERT-first text classifier + among-labels
    - rag_service.py — RAG facade (ingest/search/reset/delete)
    - vision_service.py — Vision facade (OpenAI/CLIP classify)
    - llm.py — re-exports OpenAI client
    - sales.py — re-exports negotiator state/slots
    - backends/ — implementation detail modules
      - llm_backend.py — OpenAI client
      - rag_backend.py — Chroma/embeddings RAG backend
      - vision_backend.py — Vision backend (reads labels from DB cache)
      - classifier_backend.py — SBERT classifier impl
      - sales_backend.py — session state + slot parsing + plan recommend
  - data/
    - db.py — connect() + run_migrations()
    - bootstrap.py — bootstrap_defaults(), ensure_default_system_base(), update_stems(), seed KB helpers
    - messages.py — dialog history table helpers
    - repos/ — DB repositories (pure SQL)
      - classes_repo.py, vision_labels_repo.py, rag_docs_repo.py, settings_repo.py
  - core/
    - startup.py — boot sequence: migrations → ensure defaults → bootstrap → cache warmup → include routers
    - config.py — re-exports settings (source: app/config.py)
    - constants.py — re-exports seed defaults (source: app/constants.py)
  - models/
    - schemas.py — all Pydantic models used by API
  - config.py — Settings source (env-driven). Prefer importing via app.core.config
  - constants.py — Seed defaults only (used by data/bootstrap). Never read at runtime.

## Runtime Principles

- No runtime fallbacks for prompts/labels. All prompts, stems, and vision labels live in SQLite and are editable via /admin/*.
- Seed defaults (system_base, class prompts/stems, vision labels) initialize the DB on first boot only.
- Caches (services/cache) are the single read path for categories, vision labels, and system_base.
- RAG/vision backends are separated from API via services/* facades.

## Adding or Changing Functionality

1) New endpoint
   - Create a router module under app/api (or app/api/admin for admin endpoints).
   - Import service methods (not backends) and Pydantic models from app/models/schemas.py.
   - Include route-level input validation; avoid business logic in routers.

2) Service logic
   - Add functions to services/* (or a new module) and call backends or repos.
   - Keep services pure and small; surface clear Python types and dicts.

3) DB work
   - Use a repo under app/data/repos; keep SQL here.
   - If schema changes are needed, extend app/data/db.py run_migrations().
   - For boot/seed flows, extend app/data/bootstrap.py.

4) Defaults
   - Only edit app/constants.py to change seed defaults.
   - If you need to reset DB values to defaults, prefer adding a dedicated admin utility endpoint.

## Negotiator (Agent)

- /agent/message implements a prototype sales funnel with two paths: decisive (confident plan) and guided (unsure → ask 2 short questions → recommend plan).
- State is in-memory per session (services/backends/sales_backend.py). Future: persist to DB.
- RAG sources are included in responses when ENABLE_RAG=1 and the vector index is available.

## Feature Flags & Settings

- See /features for a snapshot of flags and metrics.
- Key env vars (see README/.env.example):
  - OPENAI_API_KEY, OPENAI_MODEL, OPENAI_USE_RESPONSES, REPLY_MAX_TOKENS
  - ENABLE_RAG, RAG_USE_OPENAI_EMBEDDINGS, OPENAI_EMBED_MODEL, CHROMA_DIR, DOCS_DIR, RAG_STORAGE_DIR
  - ENABLE_VISION, VISION_BACKEND, OPENAI_VISION_MODEL
  - ADMIN_TOKEN, SQLITE_PATH
  - CLASSIFIER_BACKEND (responses|sbert), SBERT_MODEL

## Expectations for Agents

- Use services/* and repos/*; do not import backends directly from API modules.
- Keep router files small and focused on request/response glue.
- Prefer surgical changes; avoid mixing refactors with feature edits.
- When editing multiple files, keep changes coherent with this layout. If deviating, document in PR notes.

## Testing & Smokes

- Use the provided smoke scripts in scripts/ to validate RAG sync, admin CRUD, and agent flows.
- OPENAI key optional: some flows will gracefully return offline messages when not configured.

## Notes

- app/config.py and app/constants.py are the underlying sources for settings/defaults; new code should import via app/core/config.py and app/core/constants.py respectively.
- Compiled artifacts (__pycache__) and runtime storage are ignored via .gitignore.

