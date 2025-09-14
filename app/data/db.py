from __future__ import annotations
import sqlite3
import pathlib
from ..core.config import settings

DB_PATH = pathlib.Path(settings.sqlite_path)


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def run_migrations() -> None:
    conn = connect()
    cur = conn.cursor()
    # messages
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
    # app settings
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS app_settings (
            name TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at REAL NOT NULL DEFAULT (strftime('%s','now'))
        )
        """
    )
    # categories
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
    # ensure stems_json exists (older installs)
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
    # rag documents
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
    conn.commit(); conn.close()

