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
[ -n "$ADMIN_TOKEN" ] || { echo "ADMIN_TOKEN required" >&2; exit 1; }

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

echo "[STEP] reindex (reset index, sync docs, sync KB, reindex)"
echo "[INFO] DOCS_DIR=${DOCS_DIR:-}"
curl -sS -X POST "${BASE_URL}/admin/rag/reset_index" "${auth[@]}" -H 'Accept: application/json' >/dev/null || true
if [ -n "${DOCS_DIR:-}" ] && [ -d "${DOCS_DIR:-}" ]; then
  sync_out=$(curl -sS -X POST "${BASE_URL}/admin/rag/sync_docs" "${auth[@]}" "${json[@]}" -d "$(jq -n --arg b "$DOCS_DIR" '{base:$b}')")
  echo "$sync_out" | jq . >/dev/null 2>&1 || { echo "sync_docs did not return JSON" >&2; exit 1; }
  ins=$(echo "$sync_out" | jq -r '.synced.inserted + .synced.updated')
  if [ "$ins" = "0" ]; then echo "[WARN] sync_docs found nothing at $DOCS_DIR" >&2; fi
else
  echo "[WARN] DOCS_DIR not set or not a directory; skipping sync_docs" >&2
fi
if [ -n "${DOCS_DIR:-}" ] && ([[ "${DOCS_DIR}" = "./data/kb" ]] || [[ "${DOCS_DIR}" = "data/kb" ]]); then
  echo "[INFO] DOCS_DIR points to data/kb; skipping sync_seed_kb to avoid duplicates"
else
  curl -sS -X POST "${BASE_URL}/admin/rag/sync_seed_kb" "${auth[@]}" -H 'Accept: application/json' >/dev/null || true
fi
curl -sS -X POST "${BASE_URL}/admin/rag/mark_all_dirty" "${auth[@]}" -H 'Accept: application/json' >/dev/null || true
curl -sS -X POST "${BASE_URL}/admin/rag/reindex_dirty" "${auth[@]}" -H 'Content-Type: application/json' -d '{}' >/dev/null || true

echo "[STEP] agent: definitive Plus"
req1='{"text":"We want to purchase the Plus plan for our 8-person team."}'
echo "[USER] $(echo "$req1" | jq -r .text)"
resp1=$(POSTJ /agent/message "$req1")
echo "$resp1" | jq .
tools1=$(echo "$resp1" | jq -r '.tools_used | join(",")')
echo "$tools1" | grep -q 'classify_text' || { echo "expected classify_text tool" >&2; exit 1; }
if [ "$rag_enabled" = "true" ]; then
  echo "$resp1" | jq -e '.sources | length >= 1' >/dev/null || { echo "expected sources for confident plan" >&2; exit 1; }
  echo "[INFO] sources:"; echo "$resp1" | jq -r '.sources[] | "- src=" + ( .source // "?" ) + ", doc_id=" + ( .doc_id // "?" ) + ", score=" + ( .score|tostring ) + "\n  " + (.snippet|tostring)' || true
  has_docs=$(echo "$resp1" | jq -r '.sources | map(select(.source=="docs")) | length')
  if [ "$has_docs" = "0" ]; then
    echo "expected at least one source from docs (metadata.source==\"docs\")" >&2; exit 1;
  fi
fi

echo "[STEP] agent: unsure negotiate"
req2='{"text":"We need multiple channels and some automation but not sure which plan fits."}'
echo "[USER] $(echo "$req2" | jq -r .text)"
resp2=$(POSTJ /agent/message "$req2")
echo "$resp2" | jq .
tools2=$(echo "$resp2" | jq -r '.tools_used | join(",")')
echo "$tools2" | grep -q 'classify_text' || { echo "expected classify_text tool (unsure)" >&2; exit 1; }
# Negotiation reply likely includes a question mark or probing words
reply2=$(echo "$resp2" | jq -r '.reply')
echo "$reply2" | grep -Eiq '\?|сколько|users|пользовател' || { echo "expected probing questions in negotiation reply" >&2; exit 1; }

echo "[OK] agent purchase smoke passed"
