import asyncio
import json
import os
import sys
import time

from classify import classify
from deterministic import try_deterministic
from prompts import (
    SYSTEM_PROMPTS,
    TIER_BY_CATEGORY,
    MAX_TOKENS_BY_CATEGORY,
    REASONING_EFFORT_BY_CATEGORY,
    TEMPERATURE_BY_CATEGORY,
)
from model_select import load_allowed_models, select_tiers, resolve_model
from clients import FireworksClient


INPUT_PATH = os.environ.get("TASKS_INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("TASKS_OUTPUT_PATH", "/output/results.json")
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "6"))
MAX_RUNTIME_SECONDS = 9 * 60
SLOW_CALL_WARNING_MS = 20000

def log(msg):
    print(msg, file=sys.stderr, flush=True)

async def solve_task(client, tiers, sem, task):
    task_id = task["task_id"]
    prompt = task["prompt"]

    det_answer = try_deterministic(prompt)
    if det_answer is not None:
        log(f"[det] {task_id} tokens=0")
        return {"task_id": task_id, "answer": det_answer}

    category = classify(prompt)
    system_prompt = SYSTEM_PROMPTS.get(category, SYSTEM_PROMPTS["factual_knowledge"])
    temperature = TEMPERATURE_BY_CATEGORY.get(category, 0.0)

    tier = TIER_BY_CATEGORY.get(category, "strong")
    model = resolve_model(tier, tiers)
    max_tokens = MAX_TOKENS_BY_CATEGORY.get(category, 300)
    reasoning_effort = REASONING_EFFORT_BY_CATEGORY.get(category)
    fallback = tiers["retry"] if tiers["retry"] != model else tiers["strong"]

    async with sem:
        for attempt, use_model in enumerate([model, fallback]):
            try:
                answer, tokens, latency_ms = await client.complete(
                    use_model, system_prompt, prompt, max_tokens,
                    reasoning_effort=reasoning_effort, temperature=temperature
                )
                if answer:
                    warn = " ⚠ SLOW" if latency_ms >= SLOW_CALL_WARNING_MS else ""
                    log(f"[ok] {task_id} category={category} tier={tier} model={use_model} tokens={tokens} latency={latency_ms}ms{warn}")
                    return {"task_id": task_id, "answer": answer}
            except Exception as exc:
                log(f"[retry] {task_id} attempt={attempt} model={use_model} error={exc}")
                await asyncio.sleep(1)

        log(f"[fail] {task_id} giving up after retries")
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

    models = load_allowed_models()
    tiers = select_tiers(models)
    log(f"Model tiers -> cheap: {tiers['cheap']} | strong: {tiers['strong']} | retry: {tiers['retry']}")

    client = FireworksClient()
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    results = []
    if tasks:
        running = [asyncio.ensure_future(solve_task(client, tiers, sem, t)) for t in tasks]
        _done, pending = await asyncio.wait(running, timeout=MAX_RUNTIME_SECONDS)

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
