from __future__ import annotations
from ..db import connect

def get_setting(name: str) -> str | None:
    conn = connect(); cur = conn.cursor()
    cur.execute("SELECT value FROM app_settings WHERE name=?", (name,))
    row = cur.fetchone(); conn.close()
    return None if row is None else str(row[0])

def set_setting(name: str, value: str) -> int:
    import time
    now = time.time()
    conn = connect(); cur = conn.cursor()
    cur.execute(
        "INSERT INTO app_settings(name,value,updated_at) VALUES(?,?,?) ON CONFLICT(name) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (name, value, now),
    )
    changed = cur.rowcount or 0
    conn.commit(); conn.close(); return changed

