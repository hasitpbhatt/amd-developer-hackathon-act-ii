import re

CATEGORIES = (
    "code_debugging",
    "code_generation",
    "logical_reasoning",
    "math_reasoning",
    "named_entity_recognition",
    "sentiment_classification",
    "summarization",
    "factual_knowledge",
)

_CODE_MARKERS = re.compile(r"```|\bdef \w+\(|\bfunction\s+\w+\(|\bclass\s+\w+[:\(]|;\s*$|=>|\bimport\s+\w+", re.MULTILINE)
_BUG_WORDS = re.compile(r"\b(bug|fix|error|broken|not working|traceback|exception|incorrect output|debug)\b", re.I)
_CODE_CONTEXT_WORDS = re.compile(r"\b(function|method|class|script|program|algorithm|variable|my code|the code)\b", re.I)
_GEN_WORDS = re.compile(r"\b(write (a |an )?(\w+\s+){0,2}function|implement (a |an )?function|write code|create (a |an )?function|write a program)\b", re.I)
_MATH_KEYWORDS = re.compile(r"\b(calculate|percent|percentage|total|sum|how many|how much|average|ratio|profit|discount|interest rate|projection|depreciat)\b", re.I)
_MATH_SYMBOLS = re.compile(r"\$\d|\d+\s*%")
_LOGIC_WORDS = re.compile(r"\b(either .* or|neither .* nor|if and only if|exactly one of|all of the following conditions|who is (the|shortest|tallest|oldest|youngest)|which one of|logic puzzle|satisfies (all|every) (the )?condition|cannot both|who is lying|tell the truth)\b", re.I)
_NER_WORDS = re.compile(r"\b(extract (the )?(named )?(entit(y|ies)|names)|named entit(y|ies)|identify (all )?(people|persons|organizations|locations|dates|names)|list the (people|organizations|locations|dates|names))\b", re.I)
_SENTIMENT_WORDS = re.compile(r"\b(sentiment|positive, negative|positive or negative|how does .* feel|classify the (tone|emotion))\b", re.I)
_SUMMARY_WORDS = re.compile(r"\b(summari[sz]e|summary|condense|tl;dr|in one sentence|in \d+ words|shorten)\b", re.I)

_REGEX_SCORING = [
    ("code_debugging", _BUG_WORDS, 3),
    ("code_debugging", _CODE_MARKERS, 2),
    ("code_debugging", _CODE_CONTEXT_WORDS, 1),
    ("code_generation", _GEN_WORDS, 5),
    ("code_generation", _CODE_MARKERS, 1),
    ("logical_reasoning", _LOGIC_WORDS, 5),
    ("math_reasoning", _MATH_KEYWORDS, 3),
    ("math_reasoning", _MATH_SYMBOLS, 2),
    ("named_entity_recognition", _NER_WORDS, 5),
    ("sentiment_classification", _SENTIMENT_WORDS, 5),
    ("summarization", _SUMMARY_WORDS, 5),
]

_REGEX_THRESHOLD = 4

_ONNX_MODEL = None
_CAT_EMBEDDINGS = None

def classify_onnx(text):
    import numpy as np
    from numpy.linalg import norm
    try:
        from fastembed import TextEmbedding
    except ImportError:
        return None
    global _ONNX_MODEL, _CAT_EMBEDDINGS
    if _ONNX_MODEL is None:
        _ONNX_MODEL = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        _CAT_EMBEDDINGS = {}
        for cat in CATEGORIES:
            emb = list(_ONNX_MODEL.embed([cat]))[0]
            _CAT_EMBEDDINGS[cat] = emb / (norm(emb) + 1e-10)
    try:
        emb = list(_ONNX_MODEL.embed([text]))[0]
        emb = emb / (norm(emb) + 1e-10)
        best_cat = "factual_knowledge"
        best_score = -1.0
        for cat, cat_emb in _CAT_EMBEDDINGS.items():
            score = float(np.dot(emb, cat_emb))
            if score > best_score:
                best_score = score
                best_cat = cat
        return best_cat
    except Exception:
        return None

def classify(prompt):
    text = prompt.strip()

    scores = dict.fromkeys(CATEGORIES, 0)
    for cat, pattern, weight in _REGEX_SCORING:
        if pattern.search(text):
            scores[cat] += weight

    best_cat = max(scores, key=scores.get)
    best_score = scores[best_cat]

    if best_score >= _REGEX_THRESHOLD:
        return best_cat

    onnx_cat = classify_onnx(text)
    if onnx_cat is not None:
        return onnx_cat

    return "factual_knowledge"
