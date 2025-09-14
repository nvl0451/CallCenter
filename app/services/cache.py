from __future__ import annotations
import json
from typing import Dict, List
from ..data.repos import classes_repo as _cls_repo
from ..data.repos import vision_labels_repo as _vis_repo
from ..data.repos import settings_repo as _set_repo
from ..core.constants import DEFAULT_SYSTEM_BASE

_classes: List[Dict] = []
_vision_labels: List[Dict] = []
_system_base: str = DEFAULT_SYSTEM_BASE


def reload_caches() -> Dict[str, int]:
    global _classes, _vision_labels, _system_base
    _classes = _cls_repo.fetch_active_classes()
    _vision_labels = _vis_repo.fetch_active_vision_labels()
    try:
        _sb = _set_repo.get_setting("system_base")
        if _sb:
            _system_base = _sb
    except Exception:
        _system_base = DEFAULT_SYSTEM_BASE
    return {"classes": len(_classes), "vision_labels": len(_vision_labels)}


def classes_cache() -> List[Dict]:
    return list(_classes)


def vision_labels_cache() -> List[Dict]:
    return list(_vision_labels)


def classes_normalization_map() -> Dict[str, str]:
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


def system_base() -> str:
    return _system_base
