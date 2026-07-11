import re

_SENTIMENT_RE = re.compile(r"\b(positive|negative|neutral)\b", re.I)
_MATH_VALUE_RE = re.compile(r"\banswer\s*(?::|is)\s*(-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?)\s*$", re.I | re.MULTILINE)
_NUM_ONLY_RE = re.compile(r"^(-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?)\s*$", re.MULTILINE)
_CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\s*\n(.*?)```", re.DOTALL)
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_ENTITY_RE = re.compile(r"^\s*(Person|Org|Organization|Location|Date)\s*:", re.I)

def normalize_answer(prompt, raw_answer):
    text = raw_answer.strip()
    if not text:
        return ""

    m = _ANSWER_TAG_RE.search(text)
    if m:
        text = m.group(1).strip()

    m = _CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()

    m = _SENTIMENT_RE.search(text)
    if m:
        return m.group(1).lower()

    if _ENTITY_RE.match(text):
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        return "\n".join(lines)

    m = _MATH_VALUE_RE.search(text)
    if not m:
        m = _NUM_ONLY_RE.match(text)
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

    return " ".join(text.split()[:50])
