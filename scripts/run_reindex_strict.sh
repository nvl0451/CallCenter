#!/usr/bin/env bash
set -euo pipefail

# Reindex + Classifier + CLIP strict runner
# - Uses admin endpoints to exercise RAG reindex lifecycle
# - Validates SBERT classifier tweak (no LLM dependency)
# - Optionally exercises CLIP label changes if Vision CLIP is available
#
# Requirements: curl, jq. Server running. ADMIN_TOKEN set.

BASE_URL=${BASE_URL:-"http://localhost:8000"}
ADMIN_TOKEN=${ADMIN_TOKEN:-""}
RAG_STORAGE_DIR=${RAG_STORAGE_DIR:-"./storage/rag"}

need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing $1" >&2; exit 1; }; }
need curl; need jq
[ -n "$ADMIN_TOKEN" ] || { echo "ADMIN_TOKEN required" >&2; exit 1; }

auth=( -H "Authorization: Bearer ${ADMIN_TOKEN}" )
json_hdr=( -H 'Content-Type: application/json' -H 'Accept: application/json' )

GET() { curl -sSf "${BASE_URL}$1" "${auth[@]}" -H 'Accept: application/json'; }
POST() { local p=$1; shift || true; local data='{}'; [ $# -ge 1 ] && data=$1; curl -sSf -X POST "${BASE_URL}${p}" "${auth[@]}" "${json_hdr[@]}" -d "${data}"; }
PUT() { local p=$1; shift || true; local data='{}'; [ $# -ge 1 ] && data=$1; curl -sSf -X PUT  "${BASE_URL}${p}" "${auth[@]}" "${json_hdr[@]}" -d "${data}"; }
DEL() { curl -sSf -X DELETE "${BASE_URL}$1" "${auth[@]}" -H 'Accept: application/json'; }

log() { printf "\033[33m[STEP]\033[0m %s\n" "$*"; }
ok()  { printf "\033[32m[OK]\033[0m %s\n" "$*"; }
die() { printf "\033[31m[FAIL]\033[0m %s\n" "$*"; exit 1; }

log "Health and features"
GET /health | jq -e '.ok == true' >/dev/null || die "/health failed"
features=$(GET /features)
classifier_backend=$(echo "$features" | jq -r '.classifier.backend // "responses"')
vision_enabled=$(echo "$features" | jq -r '.vision.enabled')
vision_available=$(echo "$features" | jq -r '.vision.available')
vision_backend=$(echo "$features" | jq -r '.vision.backend // "openai"')
rag_enabled=$(echo "$features" | jq -r '.rag.enabled')
rag_active_before=$(echo "$features" | jq -r '.admin.rag_docs.active')
ok "features: classifier=$classifier_backend, rag_enabled=$rag_enabled, vision=$vision_backend/$vision_available"

log "Create inline RAG doc (dirty)"
doc_title="Reindex Autotest $(date +%H:%M:%S)"
doc_content="Первая версия текста. Возврат средств в течение 14 дней."
inline=$(POST /admin/rag/inline "$(jq -n --arg t "$doc_title" --arg c "$doc_content" '{title:$t, content_text:$c, index_now:false}')")
doc_id=$(echo "$inline" | jq -r '.id')
[ -n "$doc_id" ] || die "no doc id"
echo "$inline" | jq -e '.dirty == 1 and .kind == "inline"' >/dev/null || die "inline not dirty"
ok "created inline id=$doc_id dirty=1"

log "Reindex single doc"
re1=$(POST "/admin/rag/reindex/${doc_id}" '{}')
echo "$re1" | jq -e '.ok == true and .chunks >= 1' >/dev/null || die "reindex one failed"
post=$(GET /admin/rag/docs | jq -r --argjson id "$doc_id" 'map(select(.id==$id)) | .[0]')
echo "$post" | jq -e '.dirty == 0 and .indexed_at != null and .chunks_count >= 1' >/dev/null || die "doc not marked indexed"
ok "single reindex ok (chunks=$(echo "$re1" | jq -r '.chunks'))"

log "Update inline content (mark dirty)"
new_content="Вторая версия текста. Добавлен новый раздел. Возврат и условия поддержки."
upd=$(PUT "/admin/rag/docs/${doc_id}" "$(jq -n --arg c "$new_content" '{content_text:$c}')")
echo "$upd" | jq -e '.dirty == 1' >/dev/null || die "update did not mark dirty"
ok "updated and dirty"

log "Reindex again and verify counters"
re2=$(POST "/admin/rag/reindex/${doc_id}" '{}')
echo "$re2" | jq -e '.ok == true and .chunks >= 1' >/dev/null || die "reindex two failed"
post2=$(GET /admin/rag/docs | jq -r --argjson id "$doc_id" 'map(select(.id==$id)) | .[0]')
echo "$post2" | jq -e '.dirty == 0 and .indexed_at != null and .chunks_count >= 1' >/dev/null || die "second index flags wrong"
ok "second reindex ok"

log "Mark all dirty and reindex dirty"
POST /admin/rag/mark_all_dirty '{}'>/dev/null || die "mark_all_dirty failed"
redirty=$(GET /admin/rag/docs | jq -r --argjson id "$doc_id" 'map(select(.id==$id)) | .[0].dirty')
[ "$redirty" = "1" ] || die "doc not dirty after mark_all_dirty"
batch=$(POST /admin/rag/reindex_dirty '{}')
echo "$batch" | jq -e '.processed >= 1 and .errors == 0' >/dev/null || die "reindex_dirty failed"
ok "batch reindex ok"

log "Classifier tweak via stems (SBERT backend required)"
if [ "$classifier_backend" = "sbert" ]; then
  # find sales class id
  classes=$(GET /admin/classes)
  sales_id=$(echo "$classes" | jq -r 'map(select(.name=="продажи")) | .[0].id')
  if [ "$sales_id" = "null" ] || [ -z "$sales_id" ]; then die "продажи class not found"; fi
  stem_token="zz-autostem-$(date +%s)"
  # build new stems array appending stem_token
  cur_stems_json=$(echo "$classes" | jq -r --argjson id "$sales_id" 'map(select(.id==$id)) | .[0].stems_json')
  if [ -z "$cur_stems_json" ] || [ "$cur_stems_json" = "null" ]; then cur_stems_json='[]'; fi
  new_stems=$(jq -c --arg s "$stem_token" '. + [$s]' <<<"$cur_stems_json")
  body=$(jq -n --arg sjson "$new_stems" '{name:"продажи", synonyms:["sales"], stems:($sjson|fromjson), system_prompt:"", priority:0, active:1}')
  PUT "/admin/classes/${sales_id}" "$body" >/dev/null || die "failed to update продажи stems"
  # start dialog and send message containing stem
  sid=$(curl -sSf -X POST "${BASE_URL}/dialog/start" -H 'Content-Type: application/json' -d '{}' | jq -r '.session_id')
  [ -n "$sid" ] || die "no session id"
  msg=$(curl -sSf -X POST "${BASE_URL}/dialog/message" -H 'Content-Type: application/json' -d "$(jq -n --arg s "$sid" --arg m "$stem_token хочу тариф" '{session_id:$s, message:$m}')")
  cat_type=$(echo "$msg" | jq -r '.type')
  conf=$(echo "$msg" | jq -r '.confidence')
  [ "$cat_type" = "продажи" ] || die "expected продажи, got $cat_type"
  awk "BEGIN{if(!($conf>=0 && $conf<=1)) exit 1}" || die "confidence out of range: $conf"
  ok "SBERT classified as продажи with confidence=$conf"
else
  ok "Classifier backend=$classifier_backend — skipping stems test (requires SBERT)"
fi

log "CLIP label change (if Vision CLIP available)"
if [ "$vision_enabled" = "true" ] && [ "$vision_available" = "true" ] && [ "$vision_backend" = "clip" ]; then
  labels_before=$(GET /admin/vision | jq -r 'map(select(.active==1)) | length')
  new_label="auto-clip-$(date +%s)"
  v_created=$(POST /admin/vision "$(jq -n --arg n "$new_label" '{name:$n, synonyms:["auto"], templates:["скриншот: {s}"], priority:0, active:1}')")
  echo "$v_created" | jq -e '.id != null' >/dev/null || die "vision create failed"
  # classify sample image and check logits size increased
  if [ -f sample_img/inter.png ]; then
    resp=$(curl -sSf -X POST "${BASE_URL}/vision/classify" -H "Authorization: Bearer ${ADMIN_TOKEN}" -F "file=@sample_img/inter.png")
    logits_len=$(echo "$resp" | jq -r '.logits | length')
    expect=$((labels_before+1))
    [ "$logits_len" -eq "$expect" ] || die "expected logits=$expect, got $logits_len"
    ok "Vision logits reflect label count increase ($expect)"
  else
    ok "Sample image missing; vision classify skipped"
  fi
  # cleanup
  vid=$(echo "$v_created" | jq -r '.id')
  DEL "/admin/vision/${vid}" >/dev/null || true
else
  ok "Vision CLIP not available; skipping"
fi

ok "REINDEX + CLASSIFIER + CLIP checks completed"
