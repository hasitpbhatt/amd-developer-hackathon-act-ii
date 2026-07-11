import asyncio
import json
import os
import sys
import time

from deterministic import try_deterministic
from normalize import normalize_answer
from prompts import SYSTEM_PROMPT, MAX_TOKENS, TEMPERATURE
from model_select import load_allowed_models, select_models
from clients import FireworksClient


INPUT_PATH = os.environ.get("TASKS_INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("TASKS_OUTPUT_PATH", "/output/results.json")
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "6"))
MAX_RUNTIME_SECONDS = 9 * 60
SLOW_CALL_WARNING_MS = 20000
FORCE_LLM = os.environ.get("FORCE_LLM", "").strip().lower() in ("1", "true", "yes")

def log(msg):
    print(msg, file=sys.stderr, flush=True)

async def solve_task(client, models, sem, task):
    task_id = task["task_id"]
    prompt = task["prompt"]

    if not FORCE_LLM:
        det_answer = try_deterministic(prompt)
        if det_answer is not None:
            answer = normalize_answer(prompt, det_answer)
            log(f"[det] {task_id} tokens=0")
            return {"task_id": task_id, "answer": answer}

    model = models["strongest"]
    async with sem:
        try:
            raw_answer, tokens, latency_ms = await client.complete(
                model, SYSTEM_PROMPT, prompt, MAX_TOKENS,
                temperature=TEMPERATURE
            )
            answer = normalize_answer(prompt, raw_answer) if raw_answer else ""
            if answer:
                warn = " [SLOW]" if latency_ms >= SLOW_CALL_WARNING_MS else ""
                log(f"[ok] {task_id} model={model} tokens={tokens} latency={latency_ms}ms{warn}")
                return {"task_id": task_id, "answer": answer}
        except Exception as exc:
            log(f"[fail] {task_id} model={model} error={exc}")

        return {"task_id": task_id, "answer": ""}

def _load_tasks(path):
    with open(path) as f:
        raw_tasks = json.load(f)
    tasks = []
    for i, t in enumerate(raw_tasks):
        task_id = t.get("task_id") if isinstance(t, dict) else None
        if not task_id:
            log(f"[skip] malformed task at index {i}, no task_id: {t!r}")
            continue
        tasks.append({"task_id": task_id, "prompt": t.get("prompt") or ""})
    if len(tasks) != len(raw_tasks):
        log(f"Loaded {len(tasks)} usable tasks from {path} ({len(raw_tasks) - len(tasks)} skipped)")
    else:
        log(f"Loaded {len(tasks)} tasks from {path}")
    return tasks

async def run():
    start = time.monotonic()

    tasks = _load_tasks(INPUT_PATH)

    models = select_models(load_allowed_models())
    log(f"Models: cheapest={models['cheapest']} strongest={models['strongest']}")

    client = FireworksClient()
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    results = []
    if tasks:
        running = [asyncio.ensure_future(solve_task(client, models, sem, t)) for t in tasks]
        done, pending = await asyncio.wait(running, timeout=MAX_RUNTIME_SECONDS)

        for coro, t in zip(running, tasks):
            if coro in pending:
                coro.cancel()
                log(f"[timeout] {t['task_id']} still running at deadline, marking failed")
                results.append({"task_id": t["task_id"], "answer": ""})
            else:
                results.append(coro.result())

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.monotonic() - start
    log(f"Wrote {len(results)} results to {OUTPUT_PATH} in {elapsed:.1f}s")

if __name__ == "__main__":
    asyncio.run(run())
