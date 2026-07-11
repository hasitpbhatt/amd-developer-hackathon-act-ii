SYSTEM_PROMPT = (
    "Answer concisely. For sentiment: one word (positive|negative|neutral). "
    "For NER: Person|Org|Location|Date per line. For code: output code only. "
    "For math/logic: Answer: <value>. For factual: brief answer. "
    "For summarization: one sentence."
)
MAX_TOKENS = 512
TEMPERATURE = 0.0

SYSTEM_PROMPTS = {"universal": SYSTEM_PROMPT}
MAX_TOKENS_BY_CATEGORY = {"universal": MAX_TOKENS}
TEMPERATURE_BY_CATEGORY = {"universal": TEMPERATURE}
REASONING_EFFORT_BY_CATEGORY = {}
