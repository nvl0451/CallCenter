#!/usr/bin/env bash
set -euo pipefail

# Agent purchase scenario smoke test
# - Requires: server running, ADMIN_TOKEN
# - Validates: plan detection + RAG reply, negotiation path when unsure

BASE_URL=${BASE_URL:-"http://localhost:8000"}
ADMIN_TOKEN=${ADMIN_TOKEN:-""}

# Load .env if present to pick up DOCS_DIR, ENABLE_RAG, etc.
if [ -f .env ]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs) || true
fi

need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing $1" >&2; exit 1; }; }
need curl; need jq
[ -n "${ADMIN_TOKEN:-}" ] || { echo "ADMIN_TOKEN required" >&2; exit 1; }

# Args: --reset | -r to drop and rebuild the vector index
DO_RESET=0
while [ $# -gt 0 ]; do
  case "$1" in
    -r|--reset)
      DO_RESET=1; shift ;;
    -h|--help)
      cat <<USAGE
Usage: ADMIN_TOKEN=... [DOCS_DIR=...] bash scripts/run_agent_purchase.sh [--reset]
  -r, --reset   Drop and recreate the Chroma collection before syncing docs
USAGE
      exit 0 ;;
    *)
      echo "[WARN] Unknown arg: $1 (ignored)" >&2; shift ;;
  esac
done
#[ -n "$ADMIN_TOKEN" ] already checked above

auth=( -H "Authorization: Bearer ${ADMIN_TOKEN}" )
json=( -H 'Content-Type: application/json' -H 'Accept: application/json' )

GET() { curl -sSf "${BASE_URL}$1" -H 'Accept: application/json'; }
POSTJ() { local p=$1; local body=$2; curl -sSf -X POST "${BASE_URL}${p}" "${json[@]}" -d "$body"; }
POSTA() { local p=$1; local body=$2; curl -sSf -X POST "${BASE_URL}${p}" "${auth[@]}" "${json[@]}" -d "$body"; }

echo "[STEP] health"
GET /health | jq -e '.ok == true' >/dev/null

echo "[STEP] features"
features=$(GET /features)
rag_enabled=$(echo "$features" | jq -r '.rag.enabled')
echo "$features" | jq '{rag:.rag,tools:.tools,classifier:.classifier}'

if [ "$DO_RESET" = "1" ]; then
  echo "[STEP] reindex (RESET index, sync docs, sync KB, reindex)"
else
  echo "[STEP] reindex (sync docs, sync KB, reindex)"
fi
echo "[INFO] DOCS_DIR=${DOCS_DIR:-}"
if [ "$DO_RESET" = "1" ]; then
  curl -sS -X POST "${BASE_URL}/admin/rag/reset_index" "${auth[@]}" -H 'Accept: application/json' >/dev/null || true
fi
INS_DOCS=0
if [ -n "${DOCS_DIR:-}" ] && [ -d "${DOCS_DIR:-}" ]; then
  sync_out=$(curl -sS -X POST "${BASE_URL}/admin/rag/sync_docs" "${auth[@]}" "${json[@]}" -d "$(jq -n --arg b "$DOCS_DIR" '{base:$b}')")
  echo "$sync_out" | jq . >/dev/null 2>&1 || { echo "sync_docs did not return JSON" >&2; exit 1; }
  INS_DOCS=$(echo "$sync_out" | jq -r '.synced.inserted + .synced.updated')
  if [ "$INS_DOCS" = "0" ]; then echo "[WARN] sync_docs found nothing at $DOCS_DIR" >&2; fi
else
  echo "[WARN] DOCS_DIR not set or not a directory; skipping sync_docs" >&2
fi
INS_SEED=0
if [ -n "${DOCS_DIR:-}" ] && ([[ "${DOCS_DIR}" = "./data/kb" ]] || [[ "${DOCS_DIR}" = "data/kb" ]]); then
  echo "[INFO] DOCS_DIR points to data/kb; skipping sync_seed_kb to avoid duplicates"
else
  seed_out=$(curl -sS -X POST "${BASE_URL}/admin/rag/sync_seed_kb" "${auth[@]}" -H 'Accept: application/json') || true
  if echo "$seed_out" | jq . >/dev/null 2>&1; then
    INS_SEED=$(echo "$seed_out" | jq -r '.updated + .inserted // 0' 2>/dev/null || echo 0)
  fi
fi
NEED_REINDEX=$DO_RESET
if [ "$INS_DOCS" != "0" ] || [ "$INS_SEED" != "0" ]; then NEED_REINDEX=1; fi
if [ "$NEED_REINDEX" = "1" ]; then
  curl -sS -X POST "${BASE_URL}/admin/rag/mark_all_dirty" "${auth[@]}" -H 'Accept: application/json' >/dev/null || true
  curl -sS -X POST "${BASE_URL}/admin/rag/reindex_dirty" "${auth[@]}" -H 'Content-Type: application/json' -d '{}' >/dev/null || true
else
  echo "[INFO] No doc changes and no reset; skipping reindex"
fi

echo "[STEP] agent: definitive Plus"
req1='{"text":"We want to purchase the Plus plan for our 8-person team."}'
echo "[USER] $(echo "$req1" | jq -r .text)"
resp1=$(POSTJ /agent/message "$req1")
echo "$resp1" | jq '{reply, tools_used, latencies, llm_fallback, sources_count: (.sources| length)}'
tools1=$(echo "$resp1" | jq -r '.tools_used | join(",")')
echo "$tools1" | grep -q 'classify_text' || { echo "expected classify_text tool" >&2; exit 1; }
if [ "$rag_enabled" = "true" ]; then
  echo "$resp1" | jq -e '.sources | length >= 1' >/dev/null || { echo "expected sources for confident plan" >&2; exit 1; }
  has_docs=$(echo "$resp1" | jq -r '.sources | map(select(.source=="docs")) | length')
  if [ "$has_docs" = "0" ]; then
    echo "expected at least one source from docs (metadata.source==\"docs\")" >&2; exit 1;
  fi
fi
echo "[LAT] rag_ms=$(echo "$resp1" | jq -r '.latencies.rag_ms') llm_ms=$(echo "$resp1" | jq -r '.latencies.llm_ms') fallback=$(echo "$resp1" | jq -r '.llm_fallback')"

echo "[STEP] agent: unsure negotiate"
req2='{"text":"We need multiple channels and some automation but not sure which plan fits."}'
echo "[USER] $(echo "$req2" | jq -r .text)"
resp2=$(POSTJ /agent/message "$req2")
echo "$resp2" | jq '{reply, tools_used, latencies, llm_fallback}'
tools2=$(echo "$resp2" | jq -r '.tools_used | join(",")')
echo "$tools2" | grep -q 'classify_text' || { echo "expected classify_text tool (unsure)" >&2; exit 1; }
# Negotiation reply likely includes a question mark or probing words
reply2=$(echo "$resp2" | jq -r '.reply')
echo "$reply2" | grep -Eiq '\?|сколько|users|пользовател' || { echo "expected probing questions in negotiation reply" >&2; exit 1; }
echo "[LAT] rag_ms=$(echo "$resp2" | jq -r '.latencies.rag_ms') llm_ms=$(echo "$resp2" | jq -r '.latencies.llm_ms') fallback=$(echo "$resp2" | jq -r '.llm_fallback')"

echo "[OK] agent purchase smoke passed"
