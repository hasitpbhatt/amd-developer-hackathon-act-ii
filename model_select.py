import os
import re

_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?|\d+p\d+)\s*b\b", re.I)

def _inferred_size(model_id):
    match = _SIZE_RE.search(model_id.replace("_", "-"))
    if not match:
        return 50.0
    raw = match.group(1).lower().replace("p", ".")
    try:
        return float(raw)
    except ValueError:
        return 50.0

def load_allowed_models():
    raw = os.environ.get("ALLOWED_MODELS", "")
    models = [m.strip() for m in raw.split(",") if m.strip()]
    if not models:
        raise RuntimeError("ALLOWED_MODELS empty or unset")
    return models

def select_tiers(models):
    ranked = sorted(models, key=_inferred_size)
    default_cheap = ranked[0]
    default_strong = ranked[-1]

    cheap_override = os.environ.get("CHEAP_MODEL_OVERRIDE", "").strip()
    strong_override = os.environ.get("STRONG_MODEL_OVERRIDE", "").strip()
    retry_override = os.environ.get("RETRY_MODEL_OVERRIDE", "").strip()

    cheap = cheap_override if cheap_override in models else default_cheap
    strong = strong_override if strong_override in models else default_strong

    if retry_override in models:
        retry = retry_override
    else:
        others = [m for m in ranked if m != strong]
        retry = others[-1] if others else strong

    return {"cheap": cheap, "strong": strong, "retry": retry}

def resolve_model(tier, tiers):
    return tiers.get(tier, tiers["strong"])
