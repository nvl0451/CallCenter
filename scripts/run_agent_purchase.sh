#!/usr/bin/env bash
set -euo pipefail

# Agent purchase scenario smoke test
# - Requires: server running, ADMIN_TOKEN
# - Validates: plan detection + RAG reply, negotiation path when unsure

BASE_URL=${BASE_URL:-"http://localhost:8000"}
ADMIN_TOKEN=${ADMIN_TOKEN:-""}

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

echo "[STEP] reindex (best effort)"
curl -sS -X POST "${BASE_URL}/admin/rag/mark_all_dirty" "${auth[@]}" -H 'Accept: application/json' >/dev/null || true
curl -sS -X POST "${BASE_URL}/admin/rag/reindex_dirty" "${auth[@]}" -H 'Content-Type: application/json' -d '{}' >/dev/null || true

echo "[STEP] agent: definitive Plus"
req1='{"text":"We want to purchase the Plus plan for our 8-person team."}'
resp1=$(POSTJ /agent/message "$req1")
echo "$resp1" | jq .
tools1=$(echo "$resp1" | jq -r '.tools_used | join(",")')
echo "$tools1" | grep -q 'classify_text' || { echo "expected classify_text tool" >&2; exit 1; }
if [ "$rag_enabled" = "true" ]; then
  echo "$resp1" | jq -e '.sources | length >= 1' >/dev/null || { echo "expected sources for confident plan" >&2; exit 1; }
fi

echo "[STEP] agent: unsure negotiate"
req2='{"text":"We need multiple channels and some automation but not sure which plan fits."}'
resp2=$(POSTJ /agent/message "$req2")
echo "$resp2" | jq .
tools2=$(echo "$resp2" | jq -r '.tools_used | join(",")')
echo "$tools2" | grep -q 'classify_text' || { echo "expected classify_text tool (unsure)" >&2; exit 1; }
# Negotiation reply likely includes a question mark or probing words
reply2=$(echo "$resp2" | jq -r '.reply')
echo "$reply2" | grep -Eiq '\?|сколько|users|пользовател' || { echo "expected probing questions in negotiation reply" >&2; exit 1; }

echo "[OK] agent purchase smoke passed"

