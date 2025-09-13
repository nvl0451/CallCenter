from __future__ import annotations
from typing import List, Tuple
from .config import settings
import importlib.util as _modutil

vision_available: bool = False
vision_unavailable_reason: str | None = None
_model = None
_preprocess = None
_tokenizer = None

# Backends: OpenAI (default) or local CLIP if installed
_openai_ok = bool(settings.enable_vision and settings.vision_backend == "openai" and settings.openai_api_key)

def _has_module(name: str) -> bool:
    try:
        return _modutil.find_spec(name) is not None
    except Exception:
        return False

_clip_ok = bool(
    settings.enable_vision and settings.vision_backend == "clip" and _has_module("torch") and _has_module("open_clip")
)

if settings.enable_vision and (_openai_ok or _clip_ok):
    vision_available = True
else:
    vision_available = False
    if not settings.enable_vision:
        vision_unavailable_reason = "Vision disabled via ENABLE_VISION=0"
    else:
        vision_unavailable_reason = "OpenAI or CLIP backend unavailable"

def _labels_from_cache():
    try:
        from . import cache as app_cache
        rows = app_cache.vision_labels_cache()
    except Exception:
        rows = []
    if rows:
        import json as _json
        names = []
        syn_map = {}
        tpl_map = {}
        for r in rows:
            name = str(r.get("name", "")).strip()
            if not name:
                continue
            names.append(name)
            try:
                syns = _json.loads(r.get("synonyms_json") or "[]")
            except Exception:
                syns = []
            try:
                tpls = _json.loads(r.get("templates_json") or "[]")
            except Exception:
                tpls = []
            syn_map[name] = [s for s in syns if isinstance(s, str) and s.strip()] or [name]
            tpl_map[name] = [t for t in tpls if isinstance(t, str) and t.strip()]
        if names:
            return names, syn_map, tpl_map
    # Fallback defaults
    names = [
        "ошибка интерфейса",
        "проблема с оплатой",
        "технический сбой",
        "вопрос по продукту",
        "другое",
    ]
    tpl = [
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
    syn = {
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
    return names, syn, {name: tpl for name in names}

def _load_clip():
    global _model, _preprocess, _tokenizer
    if _model is None:
        # Optional insecure download path to work around corporate MITM certs
        if getattr(settings, "vision_allow_insecure_download", False):
            try:
                import ssl as _ssl
                _ssl._create_default_https_context = _ssl._create_unverified_context  # type: ignore
            except Exception:
                pass
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-B-32", pretrained="openai"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        model.eval()
        _model, _preprocess, _tokenizer = model, preprocess, tokenizer

def _classify_openai(img: "Image.Image") -> Tuple[str, float, List[float], List[str]]:
    from io import BytesIO
    import base64
    from openai import OpenAI
    client = OpenAI(api_key=settings.openai_api_key)
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    categories, _, _ = _labels_from_cache()
    instruction = (
        "Классифицируй изображение в одну категорию из списка и верни JSON: "
        "{\"category\":\"<категория>\",\"confidence\":<0..1>} "
        f"Категории: {', '.join(categories)}"
    )
    try:
        resp = client.chat.completions.create(
            model=settings.openai_vision_model,
            messages=[
                {"role": "system", "content": "Классифицируй кратко. Верни строгий JSON."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                },
            ],
            max_tokens=32,
            temperature=0.0,
        )
        txt = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        txt = f"{{\"category\":\"другое\",\"confidence\":0.0,\"error\":\"{e}\"}}"
    import json, re
    s = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.IGNORECASE|re.MULTILINE).strip()
    try:
        data = json.loads(s)
        cat = str(data.get("category", "другое")).strip().lower()
        conf = float(data.get("confidence", 0.0))
    except Exception:
        cat, conf = "другое", 0.0
    # map to known categories
    cat_norm = None
    for c in categories:
        if c.lower() == cat:
            cat_norm = c
            break
    if not cat_norm:
        cat_norm = "другое"
    idx = categories.index(cat_norm)
    logits = [0.0 for _ in categories]
    logits[idx] = max(0.0, min(1.0, conf))
    return cat_norm, max(0.0, min(1.0, conf)), logits, categories


def classify_image(img: "Image.Image") -> Tuple[str, float, List[float], List[str]]:
    if settings.vision_backend == "openai" and _openai_ok:
        return _classify_openai(img)
    if settings.vision_backend == "clip" and _clip_ok:
        # Use local CLIP
        import torch  # type: ignore
        _load_clip()
        image = _preprocess(img).unsqueeze(0)
        # Build rich text embeddings per class (average over templates+synonyms)
        with torch.no_grad():
            image_features = _model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            labels, syn_map, tpl_map = _labels_from_cache()
            class_feats = []
            for label in labels:
                phrases: List[str] = []
                tpls = tpl_map.get(label) or ["скриншот: {s}"]
                for syn in syn_map.get(label, [label]):
                    for tpl in tpls:
                        phrases.append(tpl.format(s=syn, label=label))
                toks = _tokenizer(phrases)
                tf = _model.encode_text(toks)
                tf /= tf.norm(dim=-1, keepdim=True)
                mean_tf = tf.mean(dim=0, keepdim=True)
                mean_tf /= mean_tf.norm(dim=-1, keepdim=True)
                class_feats.append(mean_tf)
            text_features = torch.cat(class_feats, dim=0)
            logits = (100.0 * image_features @ text_features.T).softmax(dim=-1).squeeze(0)
        conf, idx = float(logits.max().item()), int(logits.argmax().item())
        return labels[idx], conf, [float(x) for x in logits.tolist()], labels
    raise RuntimeError(vision_unavailable_reason or "Vision unavailable")
