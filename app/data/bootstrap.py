from __future__ import annotations
import pathlib, json, time, hashlib
from typing import Dict, List
from .db import connect
from ..core.constants import (
    DEFAULT_CLASS_PROMPTS,
    DEFAULT_CLASS_SYNONYMS,
    DEFAULT_CLASS_STEMS,
    DEFAULT_SYSTEM_BASE,
    DEFAULT_VISION_CATEGORIES,
    DEFAULT_VISION_TEMPLATES,
    DEFAULT_VISION_SYNONYMS,
)

def _now() -> float:
    return time.time()

def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def bootstrap_defaults() -> Dict[str, int]:
    inserted = {"classes": 0, "vision_labels": 0, "rag_docs": 0}
    conn = connect(); cur = conn.cursor()
    cur.execute("SELECT COUNT(1) FROM cls_categories WHERE active=1")
    if int(cur.fetchone()[0]) == 0:
        for prio, name in enumerate(["техподдержка", "продажи", "жалоба"][::-1]):
            cur.execute(
                "INSERT INTO cls_categories(name, synonyms_json, stems_json, system_prompt, priority, active, updated_at) VALUES(?,?,?,?,?,1,?)",
                (
                    name,
                    json.dumps(DEFAULT_CLASS_SYNONYMS.get(name, []), ensure_ascii=False),
                    json.dumps(DEFAULT_CLASS_STEMS.get(name, []), ensure_ascii=False),
                    DEFAULT_CLASS_PROMPTS.get(name, ""),
                    prio, _now(),
                ),
            )
            inserted["classes"] += 1
    cur.execute("SELECT COUNT(1) FROM vision_labels WHERE active=1")
    if int(cur.fetchone()[0]) == 0:
        for prio, name in enumerate(list(DEFAULT_VISION_CATEGORIES)[::-1]):
            cur.execute(
                "INSERT INTO vision_labels(name, synonyms_json, templates_json, priority, active, updated_at) VALUES(?,?,?,?,1,?)",
                (
                    name,
                    json.dumps(DEFAULT_VISION_SYNONYMS.get(name, []), ensure_ascii=False),
                    json.dumps(DEFAULT_VISION_TEMPLATES, ensure_ascii=False),
                    prio, _now(),
                ),
            )
            inserted["vision_labels"] += 1
    cur.execute("SELECT COUNT(1) FROM rag_documents WHERE active=1")
    if int(cur.fetchone()[0]) == 0:
        base = pathlib.Path("data/kb")
        for fname in ["pricing.md", "refund_policy.md", "troubleshooting.md"]:
            p = base / fname
            if not p.exists():
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except Exception:
                continue
            cur.execute(
                """
                INSERT INTO rag_documents (title, kind, rel_path, content_text, sha256, bytes, mime, source, active, created_at, updated_at, indexed_at, embed_model, chunks_count, dirty)
                VALUES(?,?,?,?,?,?,?,?,1,?,?,?,?,?,1)
                """,
                (
                    fname,
                    "inline",
                    None,
                    text,
                    _sha256_text(text),
                    len(text.encode("utf-8")),
                    "text/markdown",
                    "seed-kb",
                    _now(),
                    _now(),
                    None,
                    None,
                    0,
                ),
            )
            inserted["rag_docs"] += 1
    conn.commit(); conn.close();
    return inserted

def update_stems_bulk(stems_by_name: Dict[str, list]) -> int:
    if not stems_by_name: return 0
    import json as _json
    conn = connect(); cur = conn.cursor(); updated = 0
    for name, stems in stems_by_name.items():
        try:
            val = _json.dumps(stems or [], ensure_ascii=False)
            cur.execute("UPDATE cls_categories SET stems_json=?, updated_at=strftime('%s','now') WHERE name=? AND active=1", (val, name))
            updated += cur.rowcount or 0
        except Exception:
            pass
    conn.commit(); conn.close(); return updated

def update_default_stems() -> int:
    return update_stems_bulk(DEFAULT_CLASS_STEMS)

def ensure_default_system_base() -> None:
    from .repos.settings_repo import get_setting, set_setting
    if get_setting("system_base") is None:
        try:
            set_setting("system_base", DEFAULT_SYSTEM_BASE)
        except Exception:
            pass

def load_active_inline_docs() -> List[str]:
    conn = connect(); cur = conn.cursor()
    cur.execute("SELECT content_text FROM rag_documents WHERE active=1 AND kind='inline'")
    rows = [r[0] for r in cur.fetchall() if r and r[0]]
    conn.close(); return rows

def mark_inline_indexed(embed_model: str) -> int:
    conn = connect(); cur = conn.cursor()
    cur.execute("UPDATE rag_documents SET indexed_at=strftime('%s','now'), embed_model=?, dirty=0 WHERE active=1 AND kind='inline'", (embed_model,))
    changed = cur.rowcount or 0
    conn.commit(); conn.close(); return changed

def sync_seed_kb_from_files(base_dir: str = "data/kb") -> Dict[str, int]:
    out = {"inserted": 0, "updated": 0}
    base = pathlib.Path(base_dir)
    if not base.exists(): return out
    conn = connect(); cur = conn.cursor()
    for p in base.glob("*.md"):
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        fname = p.name
        sha = _sha256_text(text); b = len(text.encode("utf-8"))
        cur.execute("SELECT id, sha256 FROM rag_documents WHERE title=? AND kind='inline'", (fname,))
        row = cur.fetchone(); now = _now()
        if row is None:
            cur.execute(
                """
                INSERT INTO rag_documents (title, kind, rel_path, content_text, sha256, bytes, mime, source, active, created_at, updated_at, indexed_at, embed_model, chunks_count, dirty)
                VALUES(?,?,?,?,?,?,?, ?,1, ?, ?, NULL, NULL, 0, 1)
                """,
                (fname, "inline", None, text, sha, b, "text/markdown", "seed-kb", now, now),
            )
            out["inserted"] += 1
        else:
            doc_id = int(row[0]); old_sha = str(row[1] or "")
            if old_sha != sha:
                cur.execute("UPDATE rag_documents SET content_text=?, sha256=?, bytes=?, dirty=1, updated_at=? WHERE id=?", (text, sha, b, now, doc_id))
                out["updated"] += 1
    conn.commit(); conn.close(); return out

