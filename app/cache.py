from __future__ import annotations
import json
from typing import Dict, List, Tuple
from . import db

# In-memory caches
_classes: List[Dict] = []
_vision_labels: List[Dict] = []


def reload_caches() -> Dict[str, int]:
    global _classes, _vision_labels
    _classes = db.fetch_active_classes()
    _vision_labels = db.fetch_active_vision_labels()
    return {"classes": len(_classes), "vision_labels": len(_vision_labels)}


def classes_cache() -> List[Dict]:
    return list(_classes)


def vision_labels_cache() -> List[Dict]:
    return list(_vision_labels)


def classes_normalization_map() -> Dict[str, str]:
    """Lowercased lookup of synonyms and names -> canonical name."""
    mp: Dict[str, str] = {}
    for row in _classes:
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        mp[name.lower()] = name
        try:
            syns = json.loads(row.get("synonyms_json") or "[]")
        except Exception:
            syns = []
        for s in syns or []:
            if isinstance(s, str) and s.strip():
                mp[s.strip().lower()] = name
    return mp


def classes_prompts_map() -> Dict[str, str]:
    """Canonical name -> system_prompt from DB (may be empty)."""
    mp: Dict[str, str] = {}
    for row in _classes:
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        mp[name] = str(row.get("system_prompt", ""))
    return mp


def classes_names() -> List[str]:
    return [str(r.get("name")) for r in _classes if str(r.get("name", "")).strip()]


def classes_stems_map() -> Dict[str, List[str]]:
    mp: Dict[str, List[str]] = {}
    import json as _json
    for row in _classes:
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        try:
            stems = _json.loads(row.get("stems_json") or "[]")
        except Exception:
            stems = []
        mp[name] = [s for s in stems if isinstance(s, str) and s.strip()]
    return mp
