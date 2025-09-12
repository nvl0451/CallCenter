from __future__ import annotations
from typing import List, Tuple
from .config import settings

vision_available: bool = False
vision_unavailable_reason: str | None = None
_model = None
_preprocess = None
_tokenizer = None

# Only attempt to import heavy deps if the feature is enabled
if settings.enable_vision:
    try:
        import torch  # noqa: F401
        import open_clip  # noqa: F401
        vision_available = True
    except Exception as e:
        vision_available = False
        vision_unavailable_reason = f"Vision deps unavailable: {e}"
else:
    vision_available = False
    vision_unavailable_reason = "Vision disabled via ENABLE_VISION=0"

CATEGORIES = [
    "ошибка интерфейса",
    "проблема с оплатой",
    "технический сбой",
    "вопрос по продукту",
    "другое",
]

def _load():
    global _model, _preprocess, _tokenizer
    if _model is None:
        if not vision_available:
            raise RuntimeError(vision_unavailable_reason or "Vision unavailable")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-B-32", pretrained="openai"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        model.eval()
        _model, _preprocess, _tokenizer = model, preprocess, tokenizer

def classify_image(img: "Image.Image") -> Tuple[str, float, List[float], List[str]]:
    _load()
    # Use torch only if available; _load() guarantees deps or raises
    import torch  # type: ignore
    from PIL import Image  # lazy import to avoid startup crash if missing
    image = _preprocess(img).unsqueeze(0)
    texts = _tokenizer([f"фото: {c}" for c in CATEGORIES])
    with torch.no_grad():
        image_features = _model.encode_image(image)
        text_features = _model.encode_text(texts)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        logits = (100.0 * image_features @ text_features.T).softmax(dim=-1).squeeze(0)
    conf, idx = float(logits.max().item()), int(logits.argmax().item())
    return CATEGORIES[idx], conf, [float(x) for x in logits.tolist()], CATEGORIES
