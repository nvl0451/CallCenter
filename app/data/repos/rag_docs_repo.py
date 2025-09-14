from __future__ import annotations
from typing import List, Dict
import pathlib, time, hashlib
from ..db import connect
from ...core.config import settings

def _now() -> float:
    return time.time()

def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def rag_doc_metrics() -> Dict[str, int]:
    conn = connect(); cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(1) FROM rag_documents WHERE active=1")
        active_count = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(1) FROM rag_documents WHERE active=1 AND dirty=1")
        dirty_count = int(cur.fetchone()[0])
    except Exception:
        active_count, dirty_count = 0, 0
    finally:
        conn.close()
    return {"active": active_count, "dirty": dirty_count}

def ensure_storage_dirs():
    pathlib.Path(settings.rag_storage_dir).mkdir(parents=True, exist_ok=True)

def list_rag_documents(active_only: bool = True) -> List[Dict]:
    conn = connect(); cur = conn.cursor()
    if active_only:
        cur.execute("SELECT id, title, kind, rel_path, bytes, mime, source, active, created_at, updated_at, indexed_at, embed_model, chunks_count, dirty FROM rag_documents WHERE active=1 ORDER BY id DESC")
    else:
        cur.execute("SELECT id, title, kind, rel_path, bytes, mime, source, active, created_at, updated_at, indexed_at, embed_model, chunks_count, dirty FROM rag_documents ORDER BY id DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close(); return rows

def insert_rag_file(title: str, rel_path: str, bytes_len: int, sha256: str, mime: str | None, source: str = "admin-upload") -> int:
    conn = connect(); cur = conn.cursor(); now = _now()
    cur.execute(
        """
        INSERT INTO rag_documents (title, kind, rel_path, content_text, sha256, bytes, mime, source, active, created_at, updated_at, indexed_at, embed_model, chunks_count, dirty)
        VALUES(?, 'file', ?, NULL, ?, ?, ?, ?, 1, ?, ?, NULL, NULL, 0, 1)
        """,
        (title, rel_path, sha256, int(bytes_len), mime, source, now, now),
    )
    new_id = int(cur.lastrowid)
    conn.commit(); conn.close(); return new_id

def insert_rag_inline(title: str, content_text: str, source: str = "admin-inline") -> int:
    conn = connect(); cur = conn.cursor(); now = _now()
    sha = _sha256_text(content_text); b = len(content_text.encode("utf-8"))
    cur.execute(
        """
        INSERT INTO rag_documents (title, kind, rel_path, content_text, sha256, bytes, mime, source, active, created_at, updated_at, indexed_at, embed_model, chunks_count, dirty)
        VALUES(?, 'inline', NULL, ?, ?, ?, 'text/markdown', ?, 1, ?, ?, NULL, NULL, 0, 1)
        """,
        (title, content_text, sha, b, source, now, now),
    )
    new_id = int(cur.lastrowid)
    conn.commit(); conn.close(); return new_id

def update_rag_doc(doc_id: int, **fields) -> int:
    allowed = {"title", "rel_path", "content_text", "sha256", "bytes", "mime", "source", "active", "updated_at", "indexed_at", "embed_model", "chunks_count", "dirty"}
    sets = []; params: List = []  # type: ignore
    for k, v in fields.items():
        if k not in allowed:
            continue
        sets.append(f"{k}=?"); params.append(v)
    if not sets:
        return 0
    params.extend([int(doc_id)])
    conn = connect(); cur = conn.cursor()
    cur.execute(f"UPDATE rag_documents SET {', '.join(sets)} WHERE id=?", params)
    changed = cur.rowcount or 0
    conn.commit(); conn.close(); return changed

def soft_delete_rag_doc(doc_id: int) -> int:
    conn = connect(); cur = conn.cursor()
    cur.execute("UPDATE rag_documents SET active=0, updated_at=strftime('%s','now') WHERE id=?", (int(doc_id),))
    changed = cur.rowcount or 0
    conn.commit(); conn.close(); return changed

def get_rag_doc(doc_id: int) -> Dict | None:
    conn = connect(); cur = conn.cursor()
    cur.execute("SELECT * FROM rag_documents WHERE id=?", (int(doc_id),))
    row = cur.fetchone(); conn.close()
    return None if row is None else dict(row)

def mark_all_dirty() -> int:
    conn = connect(); cur = conn.cursor()
    cur.execute("UPDATE rag_documents SET dirty=1 WHERE active=1")
    changed = cur.rowcount or 0
    conn.commit(); conn.close(); return changed

def list_dirty_docs(limit: int | None = None) -> List[Dict]:
    conn = connect(); cur = conn.cursor()
    if limit and int(limit) > 0:
        cur.execute("SELECT * FROM rag_documents WHERE active=1 AND dirty=1 ORDER BY id DESC LIMIT ?", (int(limit),))
    else:
        cur.execute("SELECT * FROM rag_documents WHERE active=1 AND dirty=1 ORDER BY id DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close(); return rows

def _sync_dir_inline(base_dir: str, source_tag: str = "docs") -> Dict[str, int]:
    import pathlib as _pl
    base = _pl.Path(base_dir)
    out = {"inserted": 0, "updated": 0, "seen": 0}
    if not base.exists():
        return out
    conn = connect(); cur = conn.cursor()
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".md", ".txt"):
            continue
        rel = str(p.relative_to(base))
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        out["seen"] += 1
        sha = _sha256_text(text)
        b = len(text.encode("utf-8"))
        cur.execute("SELECT id, sha256, source FROM rag_documents WHERE title=? AND kind='inline'", (rel,))
        row = cur.fetchone()
        now = _now()
        if row is None:
            cur.execute(
                """
                INSERT INTO rag_documents (title, kind, rel_path, content_text, sha256, bytes, mime, source, active, created_at, updated_at, indexed_at, embed_model, chunks_count, dirty)
                VALUES(?,?,?,?,?,?,?, ?,1, ?, ?, NULL, NULL, 0, 1)
                """,
                (rel, "inline", None, text, sha, b, "text/markdown" if p.suffix.lower()==".md" else "text/plain", source_tag, now, now),
            )
            out["inserted"] += 1
        else:
            doc_id = int(row[0]); old_sha = str(row[1] or ""); old_src = str(row[2] or "")
            if old_sha != sha:
                cur.execute("UPDATE rag_documents SET content_text=?, sha256=?, bytes=?, dirty=1, updated_at=?, source=? WHERE id=?", (text, sha, b, now, source_tag, doc_id))
                out["updated"] += 1
            elif old_src != source_tag:
                cur.execute("UPDATE rag_documents SET source=?, updated_at=? WHERE id=?", (source_tag, now, doc_id))
                out["updated"] += 1
    conn.commit(); conn.close(); return out

def sync_docs_dir(base_dir: str) -> Dict[str, int]:
    return _sync_dir_inline(base_dir, source_tag="docs")

