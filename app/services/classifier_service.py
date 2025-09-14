from __future__ import annotations
import time as _time
from typing import Tuple, Dict

from ..core.config import settings
from .backends.classifier_backend import classify as _sbert_classify
from .backends.classifier_backend import classify_among as _sbert_among


def _classify_heuristic(text: str) -> tuple[str, float]:
    t = (text or "").lower()
    if any(k in t for k in ["оплат", "карта", "счет", "счёт", "купить", "тариф", "подписк"]):
        return "продажи", 0.8
    if any(k in t for k in ["жалоб", "недоволен", "недовольна", "возврат", "refun", "обман"]):
        return "жалоба", 0.8
    return "техподдержка", 0.7


def classify_text(text: str) -> Tuple[str, float, Dict]:
    t0 = _time.perf_counter()
    try:
        cat, conf, meta = _sbert_classify(text)
        meta = {"backend": "sbert", **meta}
        return cat, conf, meta
    except Exception:
        cat, conf = _classify_heuristic(text)
        wall_ms = int((_time.perf_counter() - t0) * 1000)
        return cat, conf, {"backend": "heuristic", "api_ms": 0, "wall_ms": wall_ms}


def classify_among(text: str, labels: list[str]) -> Tuple[str, float]:
    return _sbert_among(text, labels)
