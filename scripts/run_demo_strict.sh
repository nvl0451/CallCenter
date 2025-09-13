#!/usr/bin/env bash
set -euo pipefail

# This script exercises dialog classification (sales, tech support, complaint)
# and RAG ask, assuming GPT-only classification with provided labels from DB.
# It asserts expected categories and sanity-checks RAG scores are in [0,1].

API_BASE=${API_BASE:-http://localhost:8000}
ADMIN_TOKEN=${ADMIN_TOKEN:-}

curl_jq() {
  curl -sS "$@" | jq .
}

echo "[+] Checking /health"
curl -sS "$API_BASE/health" | jq . >/dev/null

if [[ -n "$ADMIN_TOKEN" ]]; then
  echo "[+] Bootstrapping admin defaults (classes, labels, KB)"
  curl -sS -X POST "$API_BASE/admin/bootstrap" \
    -H "Authorization: Bearer $ADMIN_TOKEN" | jq .
else
  echo "[i] ADMIN_TOKEN not set, skipping /admin/bootstrap"
fi

echo "[+] Features snapshot"
curl -sS "$API_BASE/features" | jq .

classify_case() {
  local label="$1"; shift
  local message="$1"; shift
  echo "[+] Start session for: $label"
  local sid
  sid=$(curl -sS -X POST "$API_BASE/dialog/start" -H 'Content-Type: application/json' -d '{}' | jq -r .session_id)
  if [[ -z "$sid" || "$sid" == "null" ]]; then
    echo "[!] Failed to start session" >&2; exit 1
  fi
  echo "[+] Send message: $message"
  local resp
  resp=$(curl -sS -X POST "$API_BASE/dialog/message" -H 'Content-Type: application/json' \
    -d "$(jq -n --arg s "$sid" --arg m "$message" '{session_id:$s, message:$m}')")
  echo "$resp" | jq .
  local typ
  typ=$(echo "$resp" | jq -r .type)
  if [[ "$typ" != "$label" ]]; then
    echo "[!] Expected type '$label' but got '$typ'" >&2; exit 2
  fi
  echo "[OK] Classified as $typ"
}

classify_case "продажи" "Хочу купить тариф Бизнес, подскажите цену и как оплатить?"
classify_case "техподдержка" "Приложение не запускается, появляется ошибка при старте. Что делать?"
classify_case "жалоба" "У меня жалоба: списали деньги дважды, верните оплату."

echo "[+] RAG ingest demo KB (idempotent)"
curl -sS -X POST "$API_BASE/rag/ingest" -H 'Content-Type: application/json' -d '{}' | jq .

echo "[+] RAG ask"
RAG_RESP=$(curl -sS -X POST "$API_BASE/rag/ask" -H 'Content-Type: application/json' \
  -d '{"question":"какая политика возврата?","top_k":4}')
echo "$RAG_RESP" | jq .

echo "[+] Verifying RAG source scores are within [0,1]"
bad=$(echo "$RAG_RESP" | jq '[.sources[].score | select(. < 0 or . > 1)] | length')
if [[ "$bad" != "0" ]]; then
  echo "[!] RAG contains scores outside [0,1]" >&2; exit 3
fi
echo "[OK] RAG scores look sane"

echo "[DONE] All checks passed"

