import os
from collections import Counter

_LOCAL_AVAILABLE = None
_LOCAL_LLM = None

def _has_gpu():
    import torch
    return torch.cuda.is_available()

def _init_local():
    global _LOCAL_AVAILABLE, _LOCAL_LLM
    if _LOCAL_AVAILABLE is not None:
        return _LOCAL_AVAILABLE
    try:
        if not _has_gpu():
            _LOCAL_AVAILABLE = False
            return False
        from vllm import LLM, SamplingParams
        model_id = os.getenv("LOCAL_MODEL", "Qwen/Qwen2.5-3B-Instruct")
        _LOCAL_LLM = LLM(model=model_id, dtype="half", max_model_len=4096, gpu_memory_utilization=0.25, seed=42)
        _LOCAL_AVAILABLE = True
        return True
    except Exception:
        _LOCAL_AVAILABLE = False
        return False

def _normalize(text):
    return text.strip().lower().rstrip(".,;!?")

def solve_local_ensemble(prompt, system_prompt=None):
    if not _init_local():
        return None, 0.0
    try:
        from vllm import SamplingParams
        tokenizer = _LOCAL_LLM.get_tokenizer()
        messages = [
            {"role": "system", "content": system_prompt or "You are a precise AI assistant."},
            {"role": "user", "content": prompt},
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        temps = [0.0, 0.3, 0.7]
        params = [SamplingParams(temperature=t, max_tokens=512) for t in temps]
        outputs = _LOCAL_LLM.generate([formatted] * 3, params)
        answers = []
        for out in outputs:
            text = out.outputs[0].text.strip()
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

def solve_local(prompt, max_tokens=512, temperature=0.1, system_prompt=None):
    if not _init_local():
        return None
    try:
        from vllm import SamplingParams
        tokenizer = _LOCAL_LLM.get_tokenizer()
        messages = [
            {"role": "system", "content": system_prompt or "You are a precise AI assistant."},
            {"role": "user", "content": prompt},
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        outputs = _LOCAL_LLM.generate([formatted], params)
        return outputs[0].outputs[0].text.strip()
    except Exception:
        return None
