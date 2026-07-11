"""Generate eval tasks.json and expected.json.

Output format matches competition:
  tasks.json: [{"task_id": str, "prompt": str}, ...]
  expected.json: {"task_id": str: "answer": str}

Run: python eval/generate.py
"""

import json
import os

TASKS = [
    # ── sentiment_classification (hits solve_sentiment deterministic) ──
    {"task_id": "sent_01", "prompt": "Classify the sentiment: I absolutely love this product!", "answer": "positive"},
    {"task_id": "sent_02", "prompt": "Classify the sentiment: This is the worst experience ever.", "answer": "negative"},
    {"task_id": "sent_03", "prompt": "Classify the sentiment: The sky is blue.", "answer": "neutral"},
    {"task_id": "sent_04", "prompt": "Classify the sentiment: The movie was not good at all.", "answer": "negative"},
    {"task_id": "sent_05", "prompt": "What is the sentiment? 'This restaurant is fantastic.'", "answer": "positive"},
    # ── math_reasoning (hits solve_math / solve_percentage deterministic) ──
    {"task_id": "math_01", "prompt": "What is 24 + 18?", "answer": "42"},
    {"task_id": "math_02", "prompt": "What is 15 * 3?", "answer": "45"},
    {"task_id": "math_03", "prompt": "What is 100 - 37?", "answer": "63"},
    {"task_id": "math_04", "prompt": "What is 144 / 12?", "answer": "12"},
    {"task_id": "math_05", "prompt": "Calculate 25 percent of 200.", "answer": "50"},
    # ── logical_reasoning (hits solve_logical_basic deterministic) ──
    {"task_id": "logic_01", "prompt": "Evaluate: True and False", "answer": "False"},
    {"task_id": "logic_02", "prompt": "Evaluate: True or False", "answer": "True"},
    {"task_id": "logic_03", "prompt": "Evaluate: not False", "answer": "True"},
    {"task_id": "logic_04", "prompt": "Evaluate: True and (False or True)", "answer": "True"},
    {"task_id": "logic_05", "prompt": "Evaluate: not (True and False)", "answer": "True"},
    # ── factual_knowledge (hits extractive QA deterministic) ──
    {"task_id": "fact_01", "prompt": "The capital of France is Paris. What is the capital of France?", "answer": "Paris"},
    {"task_id": "fact_02", "prompt": "Water has the chemical formula H2O. What is the chemical formula for water?", "answer": "H2O"},
    {"task_id": "fact_03", "prompt": "Romeo and Juliet was written by William Shakespeare. Who wrote Romeo and Juliet?", "answer": "William Shakespeare"},
    {"task_id": "fact_04", "prompt": "The largest planet in our solar system is Jupiter. What is the largest planet?", "answer": "Jupiter"},
    {"task_id": "fact_05", "prompt": "There are 7 continents. How many continents are there?", "answer": "7"},
    # ── counting / string / extract (hits deterministic solvers) ──
    {"task_id": "count_01", "prompt": "Count the number of words in 'hello world foo bar'", "answer": "4"},
    {"task_id": "count_02", "prompt": "Count the number of vowels in 'hello'", "answer": "2"},
    {"task_id": "string_01", "prompt": "Uppercase of 'hello'", "answer": "HELLO"},
    {"task_id": "string_02", "prompt": "Reverse of 'abc'", "answer": "cba"},
    {"task_id": "extract_01", "prompt": "Extract the email address from: Contact me at user@example.com", "answer": "user@example.com"},
    {"task_id": "extract_02", "prompt": "Extract the URL from: Visit https://example.com for more info", "answer": "https://example.com"},
    # ── named_entity_recognition (hits LLM path — needs API) ──
    {"task_id": "ner_01", "prompt": "Extract named entities from: Barack Obama was born in Hawaii.", "answer": "Person: Barack Obama\nLocation: Hawaii", "needs_api": True},
    {"task_id": "ner_02", "prompt": "Extract named entities from: Microsoft is based in Redmond.", "answer": "Organization: Microsoft\nLocation: Redmond", "needs_api": True},
    {"task_id": "ner_03", "prompt": "Extract named entities from: Steve Jobs founded Apple Inc.", "answer": "Person: Steve Jobs\nOrganization: Apple Inc.", "needs_api": True},
    {"task_id": "ner_04", "prompt": "Extract named entities from: The Eiffel Tower is in Paris.", "answer": "Location: Paris", "needs_api": True},
    {"task_id": "ner_05", "prompt": "Extract named entities from: John and Mary went to London.", "answer": "Person: John\nPerson: Mary\nLocation: London", "needs_api": True},
    # ── code_generation (hits LLM path) ──
    {"task_id": "code_gen_01", "prompt": "Write a Python function called `add` that returns the sum of a and b. Output only the function.", "answer": "def add(a, b):\n    return a + b", "needs_api": True},
    {"task_id": "code_gen_02", "prompt": "Write a Python function called `greet` that returns 'hello world'. Output only the function.", "answer": "def greet():\n    return 'hello world'", "needs_api": True},
    {"task_id": "code_gen_03", "prompt": "Write a Python function called `square` that returns n * n. Output only the function.", "answer": "def square(n):\n    return n * n", "needs_api": True},
    # ── code_debugging (hits LLM path) ──
    {"task_id": "code_dbg_01", "prompt": "Fix this function so it returns a + b:\ndef add(a, b):\n    return a - b", "answer": "def add(a, b):\n    return a + b", "needs_api": True},
    {"task_id": "code_dbg_02", "prompt": "Fix this function to return the sum:\ndef total(nums):\n    result = 0\n    for n in nums:\n        result -= n\n    return result", "needs_api": True},
    # ── summarization (hits LLM path) ──
    {"task_id": "summ_01", "prompt": "Summarize in one word: The movie was absolutely fantastic, amazing, and wonderful.", "answer": "fantastic", "needs_api": True},
    {"task_id": "summ_02", "prompt": "Summarize in one word: This is the worst meal I have ever had in my entire life.", "answer": "worst", "needs_api": True},
    {"task_id": "summ_03", "prompt": "Summarize: Revenue grew 20% year-over-year driven by cloud growth. Margins improved. EPS beat estimates.", "needs_api": True},
]

def main():
    tasks_dir = os.path.dirname(__file__)
    tasks_json = [{"task_id": t["task_id"], "prompt": t["prompt"]} for t in TASKS]
    expected_json = {t["task_id"]: t["answer"] for t in TASKS if not t.get("needs_api")}

    with open(os.path.join(tasks_dir, "tasks.json"), "w") as f:
        json.dump(tasks_json, f, indent=2)
    with open(os.path.join(tasks_dir, "expected.json"), "w") as f:
        json.dump(expected_json, f, indent=2)

    det_count = len(expected_json)
    api_count = sum(1 for t in TASKS if t.get("needs_api"))
    print(f"Generated {len(tasks_json)} tasks ({det_count} deterministic, {api_count} API-only)")

if __name__ == "__main__":
    main()
