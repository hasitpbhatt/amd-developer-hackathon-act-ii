import os
from collections import Counter

_LLM = None

def _init():
    global _LLM
    if _LLM is not None:
        return True
    try:
        from llama_cpp import Llama
    except ImportError:
        return False
    model_path = os.environ.get("LOCAL_MODEL_PATH", "/app/models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
    if not os.path.exists(model_path):
        return False
    try:
        _LLM = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=os.cpu_count() or 4,
            verbose=False,
        )
        return True
    except Exception:
        return False

def _normalize(text):
    return text.strip().lower().rstrip(".,;!?")

def solve_local_ensemble(prompt, system_prompt=None):
    if not _init():
        return None, 0.0
    try:
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        temps = [0.0, 0.3, 0.7]
        answers = []
        for t in temps:
            out = _LLM.create_chat_completion(
                messages, temperature=t, max_tokens=512
            )
            text = out["choices"][0]["message"]["content"].strip()
            if text:
                answers.append(_normalize(text))
        if not answers:
            return None, 0.0
        winner, count = Counter(answers).most_common(1)[0]
        confidence = count / len(answers)
        if confidence >= 0.67:
            return winner, confidence
        return None, 0.0
    except Exception:
        return None, 0.0

def solve_local(prompt, system_prompt=None, temperature=0.1):
    if not _init():
        return None
    try:
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        out = _LLM.create_chat_completion(
            messages, temperature=temperature, max_tokens=512
        )
        return out["choices"][0]["message"]["content"].strip()
    except Exception:
        return None
