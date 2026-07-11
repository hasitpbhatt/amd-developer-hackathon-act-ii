SYSTEM_PROMPTS = {
    "factual_knowledge": "Brief answer.",
    "math_reasoning": "Answer: <value>",
    "sentiment_classification": "positive|negative|neutral",
    "summarization": "First sentence.",
    "named_entity_recognition": "Person|Org|Location|Date",
    "code_debugging": "Fixed code.",
    "logical_reasoning": "Answer: <conclusion>",
    "code_generation": "Code block.",
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
