from __future__ import annotations
import os
import sqlite3
import pathlib
from typing import List, Dict, Tuple
import time, hashlib
import json
from .config import settings

DB_PATH = pathlib.Path(settings.sqlite_path)


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def run_migrations():
    conn = connect()
    cur = conn.cursor()
    # messages table already created in memory.init_db(); keep idempotent here too
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            ts REAL NOT NULL
        )
        """
    )
    # text categories with prompt
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS cls_categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            synonyms_json TEXT NOT NULL DEFAULT '[]',
            stems_json TEXT NOT NULL DEFAULT '[]',
            system_prompt TEXT NOT NULL DEFAULT '',
            priority INTEGER NOT NULL DEFAULT 0,
            active INTEGER NOT NULL DEFAULT 1,
            updated_at REAL NOT NULL DEFAULT (strftime('%s','now'))
        )
        """
    )
    # Backfill stems_json for older installations
    try:
        cur.execute("PRAGMA table_info(cls_categories)")
        cols = [r[1] for r in cur.fetchall()]
        if 'stems_json' not in cols:
            cur.execute("ALTER TABLE cls_categories ADD COLUMN stems_json TEXT NOT NULL DEFAULT '[]'")
    except Exception:
        pass
    # vision labels
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS vision_labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            synonyms_json TEXT NOT NULL DEFAULT '[]',
            templates_json TEXT NOT NULL DEFAULT '[]',
            priority INTEGER NOT NULL DEFAULT 0,
            active INTEGER NOT NULL DEFAULT 1,
            updated_at REAL NOT NULL DEFAULT (strftime('%s','now'))
        )
        """
    )
    # rag documents (file or inline)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            kind TEXT NOT NULL CHECK(kind IN('file','inline')),
            rel_path TEXT,
            content_text TEXT,
            sha256 TEXT NOT NULL,
            bytes INTEGER NOT NULL,
            mime TEXT,
            source TEXT,
            active INTEGER NOT NULL DEFAULT 1,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            indexed_at REAL,
            embed_model TEXT,
            chunks_count INTEGER NOT NULL DEFAULT 0,
            dirty INTEGER NOT NULL DEFAULT 1
        )
        """
    )
    conn.commit()
    conn.close()


def fetch_active_classes() -> List[Dict]:
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, synonyms_json, stems_json, system_prompt, priority FROM cls_categories WHERE active=1 ORDER BY priority DESC, id ASC"
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def fetch_classes(active_only: bool = True) -> List[Dict]:
    conn = connect()
    cur = conn.cursor()
    if active_only:
        cur.execute(
            "SELECT id, name, synonyms_json, stems_json, system_prompt, priority, active, updated_at FROM cls_categories WHERE active=1 ORDER BY priority DESC, id ASC"
        )
    else:
        cur.execute(
            "SELECT id, name, synonyms_json, stems_json, system_prompt, priority, active, updated_at FROM cls_categories ORDER BY priority DESC, id ASC"
        )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def insert_class(name: str, synonyms: List[str] | None, stems: List[str] | None, system_prompt: str, priority: int, active: int = 1) -> int:
    conn = connect(); cur = conn.cursor()
    cur.execute(
        "INSERT INTO cls_categories(name, synonyms_json, stems_json, system_prompt, priority, active, updated_at) VALUES(?,?,?,?,?,?,?)",
        (
            name.strip(),
            json.dumps(synonyms or [], ensure_ascii=False),
            json.dumps(stems or [], ensure_ascii=False),
            system_prompt or "",
            int(priority or 0),
            int(1 if active else 0),
            _now(),
        ),
    )
    new_id = int(cur.lastrowid)
    conn.commit(); conn.close()
    return new_id

def update_class(cls_id: int, **fields) -> int:
    allowed = {"name", "synonyms_json", "stems_json", "system_prompt", "priority", "active"}
    sets = []
    params: List = []
    for k, v in fields.items():
        if k in ("synonyms",):
            k = "synonyms_json"; v = json.dumps(v or [], ensure_ascii=False)
        if k in ("stems",):
            k = "stems_json"; v = json.dumps(v or [], ensure_ascii=False)
        if k not in allowed:
            continue
        sets.append(f"{k}=?")
        params.append(v)
    if not sets:
        return 0
    params.extend([_now(), int(cls_id)])
    conn = connect(); cur = conn.cursor()
    cur.execute(f"UPDATE cls_categories SET {', '.join(sets)}, updated_at=? WHERE id=?", params)
    changed = cur.rowcount or 0
    conn.commit(); conn.close()
    return changed

def soft_delete_class(cls_id: int) -> int:
    conn = connect(); cur = conn.cursor()
    cur.execute("UPDATE cls_categories SET active=0, updated_at=? WHERE id=?", (_now(), int(cls_id)))
    changed = cur.rowcount or 0
    conn.commit(); conn.close()
    return changed


def fetch_active_vision_labels() -> List[Dict]:
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, synonyms_json, templates_json, priority FROM vision_labels WHERE active=1 ORDER BY priority DESC, id ASC"
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def fetch_vision_labels(active_only: bool = True) -> List[Dict]:
    conn = connect(); cur = conn.cursor()
    if active_only:
        cur.execute(
            "SELECT id, name, synonyms_json, templates_json, priority, active, updated_at FROM vision_labels WHERE active=1 ORDER BY priority DESC, id ASC"
        )
    else:
        cur.execute(
            "SELECT id, name, synonyms_json, templates_json, priority, active, updated_at FROM vision_labels ORDER BY priority DESC, id ASC"
        )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close();
    return rows

def insert_vision_label(name: str, synonyms: List[str] | None, templates: List[str] | None, priority: int, active: int = 1) -> int:
    conn = connect(); cur = conn.cursor()
    cur.execute(
        "INSERT INTO vision_labels(name, synonyms_json, templates_json, priority, active, updated_at) VALUES(?,?,?,?,?,?)",
        (
            name.strip(),
            json.dumps(synonyms or [], ensure_ascii=False),
            json.dumps(templates or [], ensure_ascii=False),
            int(priority or 0),
            int(1 if active else 0),
            _now(),
        ),
    )
    new_id = int(cur.lastrowid)
    conn.commit(); conn.close()
    return new_id

def update_vision_label(lbl_id: int, **fields) -> int:
    allowed = {"name", "synonyms_json", "templates_json", "priority", "active"}
    sets = []; params: List = []
    for k, v in fields.items():
        if k in ("synonyms",):
            k = "synonyms_json"; v = json.dumps(v or [], ensure_ascii=False)
        if k in ("templates",):
            k = "templates_json"; v = json.dumps(v or [], ensure_ascii=False)
        if k not in allowed:
            continue
        sets.append(f"{k}=?"); params.append(v)
    if not sets:
        return 0
    params.extend([_now(), int(lbl_id)])
    conn = connect(); cur = conn.cursor()
    cur.execute(f"UPDATE vision_labels SET {', '.join(sets)}, updated_at=? WHERE id=?", params)
    changed = cur.rowcount or 0
    conn.commit(); conn.close()
    return changed

def soft_delete_vision_label(lbl_id: int) -> int:
    conn = connect(); cur = conn.cursor()
    cur.execute("UPDATE vision_labels SET active=0, updated_at=? WHERE id=?", (_now(), int(lbl_id)))
    changed = cur.rowcount or 0
    conn.commit(); conn.close()
    return changed


def rag_doc_metrics() -> Dict[str, int]:
    conn = connect()
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(1) FROM rag_documents WHERE active=1")
        active_count = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(1) FROM rag_documents WHERE active=1 AND dirty=1")
        dirty_count = int(cur.fetchone()[0])
    except sqlite3.Error:
        active_count, dirty_count = 0, 0
    finally:
        conn.close()
    return {"active": active_count, "dirty": dirty_count}


def ensure_storage_dirs():
    pathlib.Path(settings.rag_storage_dir).mkdir(parents=True, exist_ok=True)


def _now() -> float:
    return time.time()


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def bootstrap_defaults() -> Dict[str, int]:
    """Populate DB with previous hardcoded defaults if empty.
    - Classes + prompts (tech support, sales, complaint)
    - Vision labels + synonyms/templates
    - RAG demo docs from data/kb as inline
    Idempotent: inserts only when respective tables are empty.
    """
    inserted = {"classes": 0, "vision_labels": 0, "rag_docs": 0}
    conn = connect()
    cur = conn.cursor()
    # Classes
    cur.execute("SELECT COUNT(1) FROM cls_categories WHERE active=1")
    if int(cur.fetchone()[0]) == 0:
        # Built-in prompts
        prompts = {
            "техподдержка": (
                "Ты специалист техподдержки. Попроси у клиента минимально необходимые детали, "
                "дай 1-3 шага решения, при необходимости предложи эскалацию."
            ),
            "продажи": (
                "Ты менеджер по продажам. Уточни потребности, предложи подходящий тариф/пакет, "
                "сформулируй чёткий next step (демо, счёт, пробный период)."
            ),
            "жалоба": (
                "Ты сотрудник по работе с жалобами. Признай проблему, извинись, опиши, что сделаешь, "
                "предложи компенсацию при необходимости и сроки ответа."
            ),
        }
        syns_map = {
            "техподдержка": ["поддержка", "support"],
            "продажи": ["sales"],
            "жалоба": ["complaint"],
        }
        stems_map = {
            "техподдержка": ["ошибк", "не запуска", "проблем", "support"],
            "продажи": ["куп", "тариф", "цен", "оплат", "счёт", "подписк"],
            "жалоба": ["жалоб", "возврат", "refund", "дважд"],
        }
        for prio, name in enumerate(["техподдержка", "продажи", "жалоба" ][::-1]):
            cur.execute(
                "INSERT INTO cls_categories(name, synonyms_json, stems_json, system_prompt, priority, active, updated_at) VALUES(?,?,?,?,?,1,?)",
                (
                    name,
                    json.dumps(syns_map.get(name, []), ensure_ascii=False),
                    json.dumps(stems_map.get(name, []), ensure_ascii=False),
                    prompts.get(name, ""),
                    prio, _now(),
                ),
            )
            inserted["classes"] += 1
    # Vision labels
    cur.execute("SELECT COUNT(1) FROM vision_labels WHERE active=1")
    if int(cur.fetchone()[0]) == 0:
        categories = [
            "ошибка интерфейса",
            "проблема с оплатой",
            "технический сбой",
            "вопрос по продукту",
            "другое",
        ]
        clip_templates = [
            "скриншот: {s}",
            "интерфейс: {s}",
            "сообщение: {s}",
            "предупреждение: {s}",
            "ошибка: {s}",
            "a screenshot of {s}",
            "ui: {s}",
            "dialog: {s}",
            "notice: {s}",
        ]
        clip_synonyms = {
            "ошибка интерфейса": [
                "ошибка интерфейса",
                "сообщение об ошибке",
                "окно ошибки",
                "предупреждение интерфейса",
                "interface error",
                "ui error",
                "error dialog",
                "warning dialog",
            ],
            "проблема с оплатой": [
                "проблема с оплатой",
                "ошибка оплаты",
                "payment error",
                "billing problem",
                "declined card",
            ],
            "технический сбой": [
                "технический сбой",
                "server error",
                "internal error",
                "crash",
                "stack trace",
            ],
            "вопрос по продукту": [
                "вопрос по продукту",
                "product question",
                "how to use",
                "help screen",
            ],
            "другое": [
                "другое",
                "other",
                "misc",
            ],
        }
        for prio, name in enumerate(categories[::-1]):
            cur.execute(
                "INSERT INTO vision_labels(name, synonyms_json, templates_json, priority, active, updated_at) VALUES(?,?,?,?,1,?)",
                (
                    name,
                    json.dumps(clip_synonyms.get(name, []), ensure_ascii=False),
                    json.dumps(clip_templates, ensure_ascii=False),
                    prio, _now(),
                ),
            )
            inserted["vision_labels"] += 1
    # RAG demo docs from data/kb as inline
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
    conn.commit()
    conn.close()
    return inserted


def load_active_inline_docs() -> List[str]:
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT content_text FROM rag_documents WHERE active=1 AND kind='inline' AND content_text IS NOT NULL"
    )
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    return rows


def mark_inline_indexed(embed_model: str):
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        "UPDATE rag_documents SET indexed_at=?, embed_model=?, dirty=0 WHERE active=1 AND kind='inline'",
        (_now(), embed_model),
    )
    conn.commit()
    conn.close()

def list_rag_documents(active_only: bool = True) -> List[Dict]:
    conn = connect(); cur = conn.cursor()
    if active_only:
        cur.execute("SELECT * FROM rag_documents WHERE active=1 ORDER BY id DESC")
    else:
        cur.execute("SELECT * FROM rag_documents ORDER BY id DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close(); return rows

def insert_rag_inline(title: str, content_text: str, source: str = "admin-inline") -> int:
    now = _now()
    sha = _sha256_text(content_text or "")
    b = len((content_text or "").encode("utf-8"))
    conn = connect(); cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO rag_documents (title, kind, rel_path, content_text, sha256, bytes, mime, source, active, created_at, updated_at, indexed_at, embed_model, chunks_count, dirty)
        VALUES(?,?,?,?,?,?,?, ?,1, ?, ?, NULL, NULL, 0, 1)
        """,
        (title, "inline", None, content_text, sha, b, "text/markdown", source, now, now),
    )
    new_id = int(cur.lastrowid)
    conn.commit(); conn.close(); return new_id

def insert_rag_file(title: str, rel_path: str, bytes_len: int, sha256: str, mime: str | None, source: str = "admin-upload") -> int:
    now = _now()
    conn = connect(); cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO rag_documents (title, kind, rel_path, content_text, sha256, bytes, mime, source, active, created_at, updated_at, indexed_at, embed_model, chunks_count, dirty)
        VALUES(?,?,?,?,?,?,?, ?,1, ?, ?, NULL, NULL, 0, 1)
        """,
        (title, "file", rel_path, None, sha256, int(bytes_len), mime, source, now, now),
    )
    new_id = int(cur.lastrowid)
    conn.commit(); conn.close(); return new_id

def update_rag_doc(doc_id: int, **fields) -> int:
    allowed = {"title", "rel_path", "content_text", "sha256", "bytes", "mime", "source", "active", "dirty", "indexed_at", "embed_model", "chunks_count"}
    sets = []; params: List = []
    for k, v in fields.items():
        if k not in allowed: continue
        sets.append(f"{k}=?"); params.append(v)
    if not sets:
        return 0
    params.extend([_now(), int(doc_id)])
    conn = connect(); cur = conn.cursor()
    cur.execute(f"UPDATE rag_documents SET {', '.join(sets)}, updated_at=? WHERE id=?", params)
    changed = cur.rowcount or 0
    conn.commit(); conn.close(); return changed

def soft_delete_rag_doc(doc_id: int) -> int:
    conn = connect(); cur = conn.cursor()
    cur.execute("UPDATE rag_documents SET active=0, updated_at=? WHERE id=?", (_now(), int(doc_id)))
    changed = cur.rowcount or 0
    conn.commit(); conn.close(); return changed

def get_rag_doc(doc_id: int) -> Dict | None:
    conn = connect(); cur = conn.cursor()
    cur.execute("SELECT * FROM rag_documents WHERE id=?", (int(doc_id),))
    row = cur.fetchone()
    conn.close();
    return dict(row) if row else None

def mark_all_dirty() -> int:
    conn = connect(); cur = conn.cursor()
    cur.execute("UPDATE rag_documents SET dirty=1, updated_at=strftime('%s','now') WHERE active=1")
    changed = cur.rowcount or 0
    conn.commit(); conn.close(); return changed

def list_dirty_docs(limit: int | None = None) -> List[Dict]:
    conn = connect(); cur = conn.cursor()
    q = "SELECT * FROM rag_documents WHERE active=1 AND dirty=1 ORDER BY id ASC"
    if isinstance(limit, int) and limit > 0:
        q += f" LIMIT {int(limit)}"
    cur.execute(q)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close(); return rows

def set_doc_dirty(doc_id: int, dirty: int = 1) -> int:
    conn = connect(); cur = conn.cursor()
    cur.execute("UPDATE rag_documents SET dirty=?, updated_at=strftime('%s','now') WHERE id=?", (int(1 if dirty else 0), int(doc_id)))
    changed = cur.rowcount or 0
    conn.commit(); conn.close(); return changed


def update_stems_bulk(stems_by_name: Dict[str, list]) -> int:
    """Update stems_json for given category names. Returns rows updated."""
    if not stems_by_name:
        return 0
    import json as _json
    conn = connect()
    cur = conn.cursor()
    updated = 0
    for name, stems in stems_by_name.items():
        try:
            val = _json.dumps(stems or [], ensure_ascii=False)
            cur.execute(
                "UPDATE cls_categories SET stems_json=?, updated_at=strftime('%s','now') WHERE name=? AND active=1",
                (val, name),
            )
            updated += cur.rowcount or 0
        except Exception:
            pass
    conn.commit(); conn.close()
    return updated


def update_default_stems() -> int:
    """Set improved stems for default 3 classes if they exist."""
    stems = {
        "техподдержка": [
            "ошибк", "не запуска", "проблем", "что дела", "как исправ", "инструкц",
            "шаг", "перезагруз", "переустанов", "очистк кеш", "не удаётс войт", "код ошиб", "лог"
        ],
        "продажи": [
            "куп", "тариф", "цен", "оплат", "счёт", "подписк", "демо", "пробн"
        ],
        "жалоба": [
            "жалоб", "вернут ден", "возврат", "списал дважд", "недоволен", "обман", "мошенн", "пожалова", "компенсац"
        ],
    }
    return update_stems_bulk(stems)
