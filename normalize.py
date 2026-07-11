import re

_SENTIMENT_RE = re.compile(r"\b(positive|negative|neutral)\b", re.I)
_MATH_ANSWER_RE = re.compile(r"(?:\banswer:\s*)?(-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?)", re.I)
_CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\s*\n(.*?)```", re.DOTALL)
_ENTITY_LINE_RE = re.compile(r"^(Person|Org|Organization|Location|Date)\b", re.I)
_NER_SINGLE_RE = re.compile(r"\b(Person|Org|Organization|Location|Date|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")

def normalize_sentiment(text):
    m = _SENTIMENT_RE.search(text)
    return m.group(1).lower() if m else text.strip()

def normalize_math(text):
    m = _MATH_ANSWER_RE.search(text)
    if m:
        raw = m.group(1)
        if "/" in raw:
            parts = raw.split("/")
            try:
                val = float(parts[0]) / float(parts[1])
                return str(int(val) if val == int(val) else val)
            except (ValueError, ZeroDivisionError):
                pass
        return raw.strip()
    return text.strip()

def normalize_code(text):
    m = _CODE_BLOCK_RE.search(text)
    return m.group(1).strip() if m else text.strip()

def normalize_ner(text):
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    valid = []
    for line in lines:
        if _ENTITY_LINE_RE.match(line):
            valid.append(line)
    if valid:
        return "\n".join(valid)
    m = _NER_SINGLE_RE.search(text)
    return m.group(0).strip() if m else text.strip()

def normalize_summary(text, max_sentences=2):
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sents[:max_sentences]) if len(sents) > max_sentences else text.strip()

def normalize_logical(text):
    m = re.search(r"answer:\s*(.+)", text, re.I)
    return m.group(1).strip() if m else text.strip()

def normalize_factual(text):
    return " ".join(text.strip().split()[:50])

NORMALIZERS = {
    "sentiment_classification": normalize_sentiment,
    "math_reasoning": normalize_math,
    "named_entity_recognition": normalize_ner,
    "summarization": normalize_summary,
    "code_debugging": normalize_code,
    "code_generation": normalize_code,
    "logical_reasoning": normalize_logical,
    "factual_knowledge": normalize_factual,
}

def normalize(category, answer):
    fn = NORMALIZERS.get(category)
    if fn is None:
        return answer.strip()
    return fn(answer) or answer.strip()
