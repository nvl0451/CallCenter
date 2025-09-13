#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${BASE_URL:-"http://localhost:8000"}

need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing $1" >&2; exit 1; }; }
need curl; need jq

POST() { local p=$1; shift || true; local body=${1:-"{}"}; curl -sSf -X POST "${BASE_URL}${p}" -H 'Content-Type: application/json' -d "$body"; }

echo "[STEP] Unsure → discovery"
user1='мы присматриваемся, подскажите какой план подойдёт'
echo "[USER] $user1"
resp1=$(POST /agent/message "$(jq -n --arg t "$user1" '{text:$t}')")
echo "$resp1" | jq '{reply, tools_used, stage, slots, latencies}'
sid=$(echo "$resp1" | jq -r .session_id)
[ -n "$sid" ] || { echo "no session" >&2; exit 1; }

echo "[STEP] Provide details + email (quote)"
user2='8 пользователей; каналы: email и чат; оплата помесячно; admin@company.com'
echo "[USER] $user2"
resp2=$(POST /agent/message "$(jq -n --arg s "$sid" --arg t "$user2" '{session_id:$s, text:$t}')")
echo "$resp2" | jq '{reply, tools_used, stage, slots, quote, latencies}'
url=$(echo "$resp2" | jq -r '.quote.url // empty')
[ -n "$url" ] || { echo "expected quote url" >&2; exit 1; }
echo "[OK] Negotiator smoke passed"
