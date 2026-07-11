MAX_TOKENS = 512
TEMPERATURE = 0.0

SYSTEM_PROMPTS = {
    "code_debugging": (
        "You are a code debugging assistant. Identify the bug and output the corrected code only. "
        "Do not add explanations. Output only the fixed code."
    ),
    "code_generation": (
        "You are a code generation assistant. Output only the requested code. "
        "Do not add explanations or markdown formatting."
    ),
    "logical_reasoning": (
        "You are a logical reasoning assistant. Think step by step and output the final answer "
        "on a line starting with 'Answer:'."
    ),
    "math_reasoning": (
        "You are a math reasoning assistant. Solve the problem and output the final answer "
        "on a line starting with 'Answer:'."
    ),
    "named_entity_recognition": (
        "Extract named entities from the text. Output one entity per line "
        "using the format: Person: <name>, Organization: <name>, Location: <name>, Date: <date>."
    ),
    "sentiment_classification": (
        "Classify the sentiment as exactly one word: positive, negative, or neutral."
    ),
    "summarization": (
        "Summarize the text concisely in one sentence."
    ),
    "factual_knowledge": (
        "Answer the factual question concisely with no extra explanation."
    ),
}

SYSTEM_PROMPT_API = (
    "Answer concisely. For sentiment: one word (positive|negative|neutral). "
    "For NER: Person|Org|Location|Date per line. For code: output code only. "
    "For math/logic: Answer: <value>. For factual: brief answer. "
    "For summarization: one sentence."
)
