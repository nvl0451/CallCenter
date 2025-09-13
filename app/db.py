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


def fetch_active_vision_labels() -> List[Dict]:
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, synonyms_json, templates_json, priority FROM vision_labels WHERE active=1 ORDER BY priority DESC, id ASC"
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


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
