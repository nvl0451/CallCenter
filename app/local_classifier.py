from __future__ import annotations
from typing import List, Dict, Tuple
import threading, time

from .config import settings
from . import cache as app_cache

_lock = threading.Lock()
_model = None
_label_sigs: List[str] = []
_label_names: List[str] = []
_label_embs = None  # type: ignore


def _ensure_model():
    global _model
    if _model is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        raise RuntimeError(f"sentence-transformers not installed: {e}")
    device = settings.sbert_device or None
    _model = SentenceTransformer(settings.sbert_model, device=device)


def _build_label_sig(row: Dict) -> str:
    import json as _json
    name = str(row.get("name", "")).strip()
    syns = []
    stems = []
    try:
        syns = _json.loads(row.get("synonyms_json") or "[]")
    except Exception:
        syns = []
    try:
        stems = _json.loads(row.get("stems_json") or "[]")
    except Exception:
        stems = []
    syns = [s for s in syns if isinstance(s, str) and s.strip()][:6]
    stems = [s for s in stems if isinstance(s, str) and s.strip()][:6]
    txt = name
    if stems:
        txt += ". Стемы: " + ", ".join(stems)
    if syns:
        txt += ". Синонимы: " + ", ".join(syns)
    return txt


def _refresh_labels_if_needed():
    global _label_sigs, _label_names, _label_embs
    rows = app_cache.classes_cache()
    if not rows:
        return False
    sigs = [_build_label_sig(r) for r in rows]
    names = [str(r.get("name", "")).strip() for r in rows]
    if sigs == _label_sigs and names == _label_names and _label_embs is not None:
        return False
    _ensure_model()
    embs = _model.encode(sigs, normalize_embeddings=True)
    _label_sigs = sigs
    _label_names = names
    _label_embs = embs
    return True


def classify(text: str) -> Tuple[str, float, Dict]:
    t0 = time.perf_counter()
    with _lock:
        _refresh_labels_if_needed()
        if _label_embs is None or not _label_names:
            raise RuntimeError("No labels available for local classifier")
        _ensure_model()
        q = _model.encode([text], normalize_embeddings=True)[0]
    import numpy as _np  # type: ignore
    # cosine similarity since embeddings are normalized → dot product
    sims = _label_embs @ q  # type: ignore
    idx = int(_np.argmax(sims))
    sim = float(sims[idx])
    # Map cosine [-1,1] to [0,1]
    conf = max(0.0, min(1.0, (sim + 1.0) / 2.0))
    name = _label_names[idx]
    wall_ms = int((time.perf_counter() - t0) * 1000)
    meta = {"backend": "sbert", "api_ms": wall_ms, "wall_ms": wall_ms}
    return name, conf, meta

