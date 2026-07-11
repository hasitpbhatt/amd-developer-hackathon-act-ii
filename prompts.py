TIER_BY_CATEGORY = {
    "sentiment_classification": "cheap",
    "named_entity_recognition": "cheap",
    "summarization": "cheap",
    "factual_knowledge": "cheap",
    "math_reasoning": "strong",
    "logical_reasoning": "strong",
    "code_debugging": "strong",
    "code_generation": "strong",
}

SYSTEM_PROMPTS = {
    "factual_knowledge": (
        "Answer first line. 2-4 sentences."
    ),
    "math_reasoning": (
        "Answer: <value> first. One line working."
    ),
    "sentiment_classification": (
        "positive|negative|neutral. First word."
    ),
    "summarization": (
        "Summarize first sentence. Follow length limit."
    ),
    "named_entity_recognition": (
        "Person|Org|Location|Date. One per line."
    ),
    "code_debugging": (
        "Fixed code block. One comment per fix."
    ),
    "logical_reasoning": (
        "Answer: <conclusion> first. 1-2 line reasoning."
    ),
    "code_generation": (
        "Code block. One-line docstring."
    ),
}

REASONING_EFFORT_BY_CATEGORY = {
    "sentiment_classification": "none",
    "named_entity_recognition": "none",
    "summarization": "none",
    "factual_knowledge": "none",
}

TEMPERATURE_BY_CATEGORY = {
    "factual_knowledge": 0.0,
    "math_reasoning": 0.0,
    "sentiment_classification": 0.0,
    "named_entity_recognition": 0.0,
    "summarization": 0.2,
    "code_debugging": 0.1,
    "logical_reasoning": 0.1,
    "code_generation": 0.2,
}

MAX_TOKENS_BY_CATEGORY = {
    "factual_knowledge": 300,
    "math_reasoning": 400,
    "sentiment_classification": 50,
    "summarization": 300,
    "named_entity_recognition": 200,
    "code_debugging": 800,
    "logical_reasoning": 800,
    "code_generation": 800,
}
