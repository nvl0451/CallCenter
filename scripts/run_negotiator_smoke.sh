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

echo "[STEP] Provide company details (no email yet)"
user2='8 пользователей; каналы: email и чат; оплата помесячно'
echo "[USER] $user2"
resp2=$(POST /agent/message "$(jq -n --arg s "$sid" --arg t "$user2" '{session_id:$s, text:$t}')")
echo "$resp2" | jq '{reply, tools_used, stage, slots, quote, latencies}'
url=$(echo "$resp2" | jq -r '.quote.url // empty')
if [ -n "$url" ]; then echo "[WARN] quote already created on step 2 (unexpected)" >&2; fi
plan2=$(echo "$resp2" | jq -r '.slots.plan_candidate // empty')

echo "[STEP] Provide admin email (quote)"
user3='admin@company.com'
echo "[USER] $user3"
resp3=$(POST /agent/message "$(jq -n --arg s "$sid" --arg t "$user3" '{session_id:$s, text:$t}')")
echo "$resp3" | jq '{reply, tools_used, stage, slots, quote, latencies}'
final_url=$(echo "$resp3" | jq -r '.quote.url // empty')
[ -n "$final_url" ] || { echo "expected quote url on step 3" >&2; exit 1; }
# Plan consistency checks
plan3=$(echo "$resp3" | jq -r '.slots.plan_candidate // empty')
qplan=$(echo "$resp3" | jq -r '.quote.plan // empty')
if [ -n "$plan2" ] && [ -n "$plan3" ] && [ "$plan2" != "$plan3" ]; then
  echo "[FAIL] plan changed between step2 ($plan2) and step3 ($plan3)" >&2; exit 1;
fi
if [ -n "$plan2" ] && [ -n "$qplan" ] && [ "$plan2" != "$qplan" ]; then
  echo "[FAIL] quote plan ($qplan) != recommended plan ($plan2)" >&2; exit 1;
fi
echo "[OK] Negotiator smoke passed"
