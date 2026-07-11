import os
import re

_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?|\d+p\d+)\s*b\b", re.I)

_CATEGORY_PARETO = {
    "sentiment_classification": 0,
    "named_entity_recognition": 0,
    "summarization": 0,
    "factual_knowledge": 1,
    "math_reasoning": 2,
    "logical_reasoning": 3,
    "code_debugging": 3,
    "code_generation": 3,
}

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
    default_strong = ranked[-1]

    strong_override = os.environ.get("STRONG_MODEL_OVERRIDE", "").strip()
    retry_override = os.environ.get("RETRY_MODEL_OVERRIDE", "").strip()

    if strong_override in models:
        strong_idx = ranked.index(strong_override)
    else:
        strong_idx = len(ranked) - 1

    if retry_override in models:
        retry_idx = ranked.index(retry_override)
    else:
        retry_idx = max(0, strong_idx - 1)

    return {"ranked": ranked, "strong_idx": strong_idx, "retry_idx": retry_idx}

def resolve_model(category, tiers):
    ranked = tiers["ranked"]
    pareto_idx = _CATEGORY_PARETO.get(category, 1)
    n = len(ranked)
    if n == 1:
        return ranked[0]
    if pareto_idx == 0:
        return ranked[0]
    if pareto_idx >= 3:
        return ranked[tiers["strong_idx"]]
    bucket = n // 3
    model_idx = min(bucket * pareto_idx, n - 1)
    return ranked[model_idx]

def resolve_retry(category, tiers, primary_model):
    ranked = tiers["ranked"]
    idx = tiers["retry_idx"]
    retry = ranked[idx]
    if retry == primary_model:
        others = [m for m in ranked if m != primary_model]
        retry = others[-1] if others else primary_model
    return retry
