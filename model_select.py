import os
import re

_MODEL_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)")

def _inferred_size(model_id):
    m = _MODEL_SIZE_RE.search(model_id.replace("_", "-"))
    return float(m.group(1)) if m else 999.0

def load_allowed_models():
    raw = os.environ.get("ALLOWED_MODELS", "")
    models = [m.strip() for m in raw.split(",") if m.strip()]
    if not models:
        raise RuntimeError("ALLOWED_MODELS empty or unset")
    return models

def select_models(models):
    ranked = sorted(models, key=_inferred_size)
    return {"strongest": ranked[-1], "cheapest": ranked[0], "all": ranked}
