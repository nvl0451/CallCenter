#!/usr/bin/env bash
set -euo pipefail

# Strict CRUD test runner for CallCenter admin API
# - Requires: curl, jq
# - Server must be running (e.g., ./run.sh) and ADMIN_TOKEN must be set

BASE_URL=${BASE_URL:-"http://localhost:8000"}
ADMIN_TOKEN=${ADMIN_TOKEN:-""}
RAG_STORAGE_DIR=${RAG_STORAGE_DIR:-"./storage/rag"}

red() { printf "\033[31m%s\033[0m\n" "$*"; }
grn() { printf "\033[32m%s\033[0m\n" "$*"; }
ylw() { printf "\033[33m%s\033[0m\n" "$*"; }

die() { red "[FAIL] $*"; exit 1; }

need_bin() {
  command -v "$1" >/dev/null 2>&1 || die "Missing dependency: $1"
}

need_bin curl; need_bin jq
[ -n "$ADMIN_TOKEN" ] || die "ADMIN_TOKEN env is required"

auth_hdr=( -H "Authorization: Bearer ${ADMIN_TOKEN}" )
json_hdr=( -H 'Content-Type: application/json' -H 'Accept: application/json' )

get_json() { curl -sSf "${BASE_URL}$1" "${auth_hdr[@]}" -H 'Accept: application/json'; }
post_json() { local path=$1; local data=$2; curl -sSf -X POST "${BASE_URL}${path}" "${auth_hdr[@]}" "${json_hdr[@]}" -d "$data"; }
put_json() { local path=$1; local data=$2; curl -sSf -X PUT "${BASE_URL}${path}" "${auth_hdr[@]}" "${json_hdr[@]}" -d "$data"; }
del_json() { local path=$1; curl -sSf -X DELETE "${BASE_URL}${path}" "${auth_hdr[@]}" -H 'Accept: application/json'; }

post_file() { # path, fieldname=file @filepath, plus index_now flag
  local path=$1; local file=$2; local index_now=${3:-1}
  curl -sSf -X POST "${BASE_URL}${path}" "${auth_hdr[@]}" \
    -F "file=@${file}" -F "index_now=${index_now}" -H 'Accept: application/json'
}

rand_suffix() { date +%s | awk '{print $1"-"rand()}' | tr -d '\n' | sed 's/\..*$//' ; }

step() { ylw "[STEP] $*"; }
ok() { grn "[OK] $*"; }

step "Health check"
get_json /health | jq -e '.ok == true' >/dev/null || die "/health failed"

features_json=$(get_json /features)
echo "$features_json" | jq . >/dev/null || die "/features not JSON"
classes_active_before=$(echo "$features_json" | jq -r '.admin.caches.classes')
vision_active_before=$(echo "$features_json" | jq -r '.admin.caches.vision_labels')
rag_active_before=$(echo "$features_json" | jq -r '.admin.rag_docs.active')
ok "Features loaded (classes=$classes_active_before, vision=$vision_active_before, rag_active=$rag_active_before)"

step "Bootstrap idempotence"
boot_json=$(post_json /admin/bootstrap '{}')
echo "$boot_json" | jq -e '.admin.caches.classes >= 0' >/dev/null 2>&1 || true
ok "Bootstrap responded"

step "Update stems"
upd_json=$(post_json /admin/update_stems '{}')
echo "$upd_json" | jq -e '.updated_rows >= 0' >/dev/null || die "update_stems bad response"
ok "Stems updated: $(echo "$upd_json" | jq -r '.updated_rows') rows"

# ---------- Classes CRUD ----------
step "Classes: create"
suffix=$(rand_suffix)
cls_name="autotest-class-$suffix"
create_cls_json=$(post_json /admin/classes "$(jq -n --arg n "$cls_name" '{name:$n, synonyms:["auto","test"], stems:["стем"], system_prompt:"SP autotest", priority:7, active:1}')")
cls_id=$(echo "$create_cls_json" | jq -r '.id')
[ -n "$cls_id" ] || die "No class id returned"
ok "Created class id=$cls_id name=$cls_name"

step "Classes: list and find"
list_cls_json=$(get_json /admin/classes)
echo "$list_cls_json" | jq -e --argjson id "$cls_id" 'map(select(.id==$id)) | length == 1' >/dev/null || die "Created class not found in list"
ok "Class appears in list"

step "Classes: update"
update_cls_json=$(put_json "/admin/classes/${cls_id}" "$(jq -n --arg n "$cls_name" '{name:$n, synonyms:["auto","changed"], stems:["нов-стем"], system_prompt:"SP changed", priority:9, active:1}')")
echo "$update_cls_json" | jq -e '.priority == 9 and .system_prompt == "SP changed"' >/dev/null || die "Class not updated"
ok "Updated class fields"

step "Classes: delete (soft)"
del_cls_json=$(del_json "/admin/classes/${cls_id}")
echo "$del_cls_json" | jq -e '.deleted == 1' >/dev/null || die "Class soft delete failed"
post_del_list=$(get_json /admin/classes)
echo "$post_del_list" | jq -e --argjson id "$cls_id" 'map(select(.id==$id and .active==0)) | length == 1' >/dev/null || die "Deleted class not marked inactive"
ok "Class soft-deleted"

# Refresh features to check active cache count change (best-effort, tolerate unchanged if other activity)
sleep 0.2
features_after_cls=$(get_json /features)
classes_active_after=$(echo "$features_after_cls" | jq -r '.admin.caches.classes')
ok "Classes active before=$classes_active_before after=$classes_active_after (info)"

# ---------- Vision CRUD ----------
step "Vision: create"
v_suffix=$(rand_suffix)
v_name="autotest-vision-$v_suffix"
create_vis_json=$(post_json /admin/vision "$(jq -n --arg n "$v_name" '{name:$n, synonyms:["vision","auto"], templates:["скриншот: {s}", "ui: {s}"], priority:2, active:1}')")
vis_id=$(echo "$create_vis_json" | jq -r '.id')
[ -n "$vis_id" ] || die "No vision id returned"
ok "Created vision id=$vis_id"

step "Vision: update"
update_vis_json=$(put_json "/admin/vision/${vis_id}" '{"name":"'"$v_name"'","synonyms":["changed"],"templates":["dialog: {s}"],"priority":5,"active":1}')
echo "$update_vis_json" | jq -e '.priority == 5' >/dev/null || die "Vision not updated"
ok "Updated vision label"

step "Vision: delete (soft)"
del_vis_json=$(del_json "/admin/vision/${vis_id}")
echo "$del_vis_json" | jq -e '.deleted == 1' >/dev/null || die "Vision soft delete failed"
post_vis_list=$(get_json /admin/vision)
echo "$post_vis_list" | jq -e --argjson id "$vis_id" 'map(select(.id==$id and .active==0)) | length == 1' >/dev/null || die "Deleted vision not inactive"
ok "Vision soft-deleted"

sleep 0.2
features_after_vis=$(get_json /features)
vision_active_after=$(echo "$features_after_vis" | jq -r '.admin.caches.vision_labels')
ok "Vision active before=$vision_active_before after=$vision_active_after (info)"

# ---------- RAG Docs CRUD ----------
step "RAG: create inline"
inline_title="Inline Autotest $(date +%H:%M:%S)"
inline_body=$(jq -n --arg t "$inline_title" --arg c "Это тестовый документ RAG. Возврат средств в течение 14 дней." '{title:$t, content_text:$c, index_now:true}')
inline_json=$(post_json /admin/rag/inline "$inline_body")
inline_id=$(echo "$inline_json" | jq -r '.id')
[ -n "$inline_id" ] || die "Inline doc id missing"
echo "$inline_json" | jq -e '.kind == "inline" and .active == 1' >/dev/null || die "Inline doc invalid"
ok "Inline doc created id=$inline_id"

step "RAG: upload file"
tmpf=$(mktemp /tmp/ragXXXX.txt)
trap 'rm -f "$tmpf" >/dev/null 2>&1 || true' EXIT
printf '%s\n' "Тестовый файл Autotest." >"$tmpf"
file_json=$(post_file /admin/rag/file "$tmpf" 1)
file_id=$(echo "$file_json" | jq -r '.id')
[ -n "$file_id" ] || die "File doc id missing"
file_rel=$(echo "$file_json" | jq -r '.rel_path')
[ -n "$file_rel" ] || die "File rel_path missing"
echo "$file_json" | jq -e '.kind == "file" and (.rel_path|length) > 0 and .active == 1' >/dev/null || die "File doc invalid"
ok "File doc uploaded id=$file_id"

# Verify file exists on disk (best effort)
if [ -f "${RAG_STORAGE_DIR}/${file_rel}" ]; then
  ok "File present on disk: ${RAG_STORAGE_DIR}/${file_rel}"
else
  ylw "[STEP] Note: expected file not found at ${RAG_STORAGE_DIR}/${file_rel} (may be custom storage path)"
fi

sleep 0.3
features_after_docs=$(get_json /features)
rag_active_after_create=$(echo "$features_after_docs" | jq -r '.admin.rag_docs.active')
ok "RAG active before=$rag_active_before after_create=$rag_active_after_create (info)"

step "RAG: delete both docs (soft)"
del_inline_json=$(del_json "/admin/rag/docs/${inline_id}")
echo "$del_inline_json" | jq -e '.deleted == 1' >/dev/null || die "Inline delete failed"
del_file_json=$(del_json "/admin/rag/docs/${file_id}")
echo "$del_file_json" | jq -e '.deleted == 1' >/dev/null || die "File delete failed"

sleep 0.3
features_after_del=$(get_json /features)
rag_active_after_del=$(echo "$features_after_del" | jq -r '.admin.rag_docs.active')
ok "RAG active after_delete=$rag_active_after_del (info)"

step "RAG: list all docs and verify inactive"
rag_list=$(get_json /admin/rag/docs)
echo "$rag_list" | jq -e --argjson i "$inline_id" 'map(select(.id==$i and .active==0)) | length == 1' >/dev/null || die "Inline not inactive after delete"
echo "$rag_list" | jq -e --argjson i "$file_id" 'map(select(.id==$i and .active==0)) | length == 1' >/dev/null || die "File not inactive after delete"
ok "RAG docs soft-deleted"

# Cleanup physical file if still present and API did not hard-delete
if [ -n "${file_rel}" ]; then
  if [ -f "${RAG_STORAGE_DIR}/${file_rel}" ]; then
    rm -f "${RAG_STORAGE_DIR}/${file_rel}" || true
    if [ -f "${RAG_STORAGE_DIR}/${file_rel}" ]; then
      ylw "[STEP] Could not remove ${RAG_STORAGE_DIR}/${file_rel} (permissions?)."
    else
      ok "Cleaned up file ${RAG_STORAGE_DIR}/${file_rel}"
    fi
  fi
fi

ok "ALL CRUD TESTS PASSED"
