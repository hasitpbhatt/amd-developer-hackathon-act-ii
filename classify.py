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
_MATH_NUMS = re.compile(r"\d+(\.\d+)?")
_LOGIC_WORDS = re.compile(r"\b(either .* or|neither .* nor|if and only if|exactly one of|all of the following conditions|who is (the|shortest|tallest|oldest|youngest)|which one of|logic puzzle|satisfies (all|every) (the )?condition|cannot both|who is lying|tell the truth)\b", re.I)
_NER_WORDS = re.compile(r"\b(extract (the )?(named )?(entit(y|ies)|names)|named entit(y|ies)|identify (all )?(people|persons|organizations|locations|dates|names)|list the (people|organizations|locations|dates|names))\b", re.I)
_SENTIMENT_WORDS = re.compile(r"\b(sentiment|positive, negative|positive or negative|how does .* feel|classify the (tone|emotion))\b", re.I)
_SUMMARY_WORDS = re.compile(r"\b(summari[sz]e|summary|condense|tl;dr|in one sentence|in \d+ words|shorten)\b", re.I)

def classify(prompt):
    text = prompt.strip()

    if _BUG_WORDS.search(text) and (_CODE_MARKERS.search(text) or _CODE_CONTEXT_WORDS.search(text)):
        return "code_debugging"
    if _GEN_WORDS.search(text):
        return "code_generation"

    if _LOGIC_WORDS.search(text):
        return "logical_reasoning"

    if (_MATH_KEYWORDS.search(text) or _MATH_SYMBOLS.search(text)) and len(_MATH_NUMS.findall(text)) >= 1:
        return "math_reasoning"

    if _NER_WORDS.search(text):
        return "named_entity_recognition"

    if _SENTIMENT_WORDS.search(text):
        return "sentiment_classification"

    if _SUMMARY_WORDS.search(text):
        return "summarization"

    return "factual_knowledge"
