from __future__ import annotations
from typing import List, Dict
import json, time
from ..db import connect

def _now() -> float:
    return time.time()

def fetch_active_vision_labels() -> List[Dict]:
    conn = connect(); cur = conn.cursor()
    cur.execute("SELECT id, name, synonyms_json, templates_json, priority FROM vision_labels WHERE active=1 ORDER BY priority DESC, id ASC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close(); return rows

def fetch_vision_labels(active_only: bool = True) -> List[Dict]:
    conn = connect(); cur = conn.cursor()
    if active_only:
        cur.execute("SELECT id, name, synonyms_json, templates_json, priority, active, updated_at FROM vision_labels WHERE active=1 ORDER BY priority DESC, id ASC")
    else:
        cur.execute("SELECT id, name, synonyms_json, templates_json, priority, active, updated_at FROM vision_labels ORDER BY priority DESC, id ASC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close(); return rows

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
    conn.commit(); conn.close(); return new_id

def update_vision_label(lbl_id: int, **fields) -> int:
    allowed = {"name", "synonyms_json", "templates_json", "priority", "active"}
    sets = []; params: List = []  # type: ignore
    import json as _json
    for k, v in fields.items():
        if k in ("synonyms",):
            k = "synonyms_json"; v = _json.dumps(v or [], ensure_ascii=False)
        if k in ("templates",):
            k = "templates_json"; v = _json.dumps(v or [], ensure_ascii=False)
        if k not in allowed:
            continue
        sets.append(f"{k}=?"); params.append(v)
    if not sets:
        return 0
    params.extend([_now(), int(lbl_id)])
    conn = connect(); cur = conn.cursor()
    cur.execute(f"UPDATE vision_labels SET {', '.join(sets)}, updated_at=? WHERE id=?", params)
    changed = cur.rowcount or 0
    conn.commit(); conn.close(); return changed

def soft_delete_vision_label(lbl_id: int) -> int:
    conn = connect(); cur = conn.cursor()
    cur.execute("UPDATE vision_labels SET active=0, updated_at=? WHERE id=?", (_now(), int(lbl_id)))
    changed = cur.rowcount or 0
    conn.commit(); conn.close(); return changed

