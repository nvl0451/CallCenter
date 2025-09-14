from __future__ import annotations
import sqlite3, pathlib, time
from typing import List, Tuple
from ..core.config import settings

DB_PATH = pathlib.Path(settings.sqlite_path)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
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
    conn.commit(); conn.close()

init_db()

def add_message(session_id: str, role: str, content: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO messages(session_id, role, content, ts) VALUES(?,?,?,?)",
        (session_id, role, content, time.time()),
    )
    conn.commit(); conn.close()

def get_session_messages(session_id: str, limit: int = 20) -> List[Tuple[str,str]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
        (session_id, limit),
    )
    rows = cur.fetchall()
    conn.close()
    return list(reversed(rows))

