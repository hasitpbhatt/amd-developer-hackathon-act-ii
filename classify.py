import re
from collections import Counter
from math import sqrt

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

_REGEX_THRESHOLD = 3

_CAT_SEED_WORDS = {
    "code_debugging": {"bug", "fix", "error", "debug", "broken", "exception", "traceback", "incorrect", "not working", "my code", "def ", "class ", "function", "import ", "=>", "```"},
    "code_generation": {"write", "implement", "create", "generate", "function", "program", "script", "code", "algorithm", "class", "method"},
    "logical_reasoning": {"if and only if", "either", "neither", "logic", "puzzle", "condition", "conclusion", "which one", "truth", "lying", "implies", "therefore", "deduce", "infer"},
    "math_reasoning": {"calculate", "sum", "difference", "product", "quotient", "equation", "solve", "percent", "percentage", "total", "average", "ratio", "how many", "how much", "profit", "discount", "plus", "minus", "times", "divided"},
    "named_entity_recognition": {"extract", "named entity", "entities", "names", "people", "persons", "organizations", "locations", "dates", "identify", "list"},
    "sentiment_classification": {"sentiment", "positive", "negative", "neutral", "emotion", "tone", "feel", "opinion", "review", "feedback"},
    "summarization": {"summarize", "summary", "condense", "tl;dr", "shorten", "in one sentence", "brief", "concise", "overview", "key points"},
    "factual_knowledge": {"what is", "who is", "where is", "when did", "define", "meaning", "explain", "describe", "capital", "population", "located", "famous", "history", "known for", "consists of", "composed of", "refers to"},
}

def _tokenize(text):
    return re.findall(r"[a-zA-Z][a-zA-Z0-9']*(?: [a-zA-Z][a-zA-Z0-9']*)?", text.lower())

def _build_vector(tokens, seed_set):
    tv = Counter()
    for tok in tokens:
        if tok in seed_set:
            tv[tok] += 1
    return tv

def _cosine_sim(a, b):
    intersection = set(a) & set(b)
    num = sum(a[k] * b[k] for k in intersection)
    den = sqrt(sum(v * v for v in a.values())) * sqrt(sum(v * v for v in b.values()))
    return num / den if den > 0 else 0.0

_FALLBACK_VECTORS = None

def classify_fallback(text):
    global _FALLBACK_VECTORS
    if _FALLBACK_VECTORS is None:
        _FALLBACK_VECTORS = {}
        for cat, seeds in _CAT_SEED_WORDS.items():
            _FALLBACK_VECTORS[cat] = Counter({s: 1 for s in seeds})
    tokens = _tokenize(text)
    if not tokens:
        return None
    qv = _build_vector(tokens, set(w for seeds in _CAT_SEED_WORDS.values() for w in seeds))
    if not qv:
        return None
    best_cat = "factual_knowledge"
    best_score = -1.0
    for cat, fv in _FALLBACK_VECTORS.items():
        score = _cosine_sim(qv, fv)
        if score > best_score:
            best_score = score
            best_cat = cat
    return best_cat

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

    fallback_cat = classify_fallback(text)
    if fallback_cat is not None:
        return fallback_cat

    return "factual_knowledge"
