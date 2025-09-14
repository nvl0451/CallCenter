# Scope Freeze — CallCenter LLM + RAG + Vision

This document locks the deliverable scope for submission. The project ships as a solid server for LLM calls, RAG, and Vision with admin CRUD and a working Negotiator prototype (decisive + guided paths). Defer advanced sales tooling to future work.

## Objective
- Deliver a production‑ready FastAPI service that:
  - Performs LLM chat and strict JSON classification
  - Provides RAG (ingest/search) with resilient indexing from a docs folder
  - Provides Vision classification (OpenAI or local CLIP when enabled)
  - Exposes admin CRUD for classes, vision labels, and RAG documents
  - Includes a session‑based Negotiator prototype demonstrating decisive and guided purchase flows

## In Scope (Must Ship)
- Core APIs
  - `/dialog/*` basic chat (classification + reply)
  - `/rag/ingest`, `/rag/ask` (sources included)
  - `/vision/classify` (graceful errors when disabled)
  - `/admin/*` CRUD + RAG lifecycle:
    - Classes: `GET/POST/PUT/DELETE /admin/classes`
    - Vision: `GET/POST/PUT/DELETE /admin/vision`
    - RAG docs: `GET /admin/rag/docs`, `POST /admin/rag/inline`, `POST /admin/rag/file`, `PUT/DELETE /admin/rag/docs/{id}`
    - Index mgmt: `POST /admin/rag/reset_index`, `/admin/rag/mark_all_dirty`, `/admin/rag/reindex_dirty`, `/admin/rag/reindex/{id}`, `/admin/rag/sync_docs`, `/admin/rag/sync_seed_kb`
  - Negotiator: `/agent/message` with session state (intent/plan/slots/stage)
- RAG ingestion from environment `DOCS_DIR`; normalized metrics and sources surfaced
- Classifier: strict JSON via Responses or local SBERT (configurable backend)
- Vision: DB‑driven labels/synonyms/templates; dynamic logits
- Caches for classes and vision labels; invalidation on CRUD
- Feature flags and stability: service starts cleanly with any feature disabled (no crashes)
- Documentation: README updated; this SCOPE_FREEZE.md; NEGOTIATOR.md present

## Out of Scope (Documented Future Work)
- Sales tools: `plan_compare`, `discount_policy`, `compliance_info`, `schedule_call`
- Negotiator: full haggling/discount logic and comparisons
- Persisting sales state in DB (current: in‑memory per session)
- End‑to‑end pytest suite and deterministic mocks (OPENAI_MOCK is minimal)
- Admin UI / RBAC / multi‑tenant

## Definition of Done
- Runs via `run.sh`; Quickstart in README; `.env.example` provided
- `/features` accurately reports model, flags, tools budget, caches, RAG metrics
- Admin token enforced on all `/admin/*` endpoints
- RAG lifecycle verified: reset → sync (DOCS_DIR) → seed KB → mark dirty → reindex
- Negotiator prototype behavior
  - Decisive path (confident plan): short RU pitch, price line, benefits (when RAG available), CTA; stage lock
  - Guided path (unsure): exactly 2 short questions, plan recommendation, CTA
  - Quote only created when both billing and admin_email are present
  - Follow‑ups avoid repeated benefits; email‑only follow‑up does not trigger RAG
- Vision endpoint gracefully disabled when deps/flags missing; works with OpenAI key when enabled

## Validation (Smoke Scripts)
- `scripts/run_crud_strict.sh` — CRUD of classes/vision and basic checks
- `scripts/run_reindex_strict.sh` — reset/sync/reindex flows; label dynamics
- `scripts/run_agent_purchase.sh` — decisive and unsure agent paths; sources present when RAG enabled
- `scripts/run_negotiator_smoke.sh` — 3‑step funnel (discovery → plan → email/quote); no early quote; plan consistency

## Configuration & Ops
- Environment
  - `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_USE_RESPONSES`
  - `ENABLE_RAG`, `RAG_USE_OPENAI_EMBEDDINGS`, `OPENAI_EMBED_MODEL`, `CHROMA_DIR`, `DOCS_DIR`, `RAG_STORAGE_DIR`, `RAG_HARD_DELETE`
  - `ENABLE_VISION`, `VISION_BACKEND`, `OPENAI_VISION_MODEL`, `VISION_ALLOW_INSECURE_DOWNLOAD`
  - `ADMIN_TOKEN`, `SQLITE_PATH`, `REPLY_MAX_TOKENS`, `LLM_TEMPERATURE`
  - `CLASSIFIER_BACKEND` (responses|sbert), `SBERT_MODEL`, `SBERT_DEVICE`
- Behavior
  - No startup crashes if features disabled or keys missing
  - Tool budgets: ≤3 calls/turn, ~20s total; per‑tool 8s target (best‑effort)
  - RAG search returns sources with `{source, doc_id, score, snippet}`

## Submission Notes
- SPEC.md delivered (baseline) and SPEC_2.md implemented (admin‑first + tools)
- Negotiator is a prototype for demonstration, not a full sales engine
- Future work listed above; prompts can be moved to DB later if needed
