"""Run pipeline against eval tasks, compare to expected, report accuracy.

Usage:
    python eval/run_eval.py                     # deterministic-only (fast, no API)
    python eval/run_eval.py --with-api           # full pipeline (requires FIREWORKS_API_KEY)
    python eval/run_eval.py --diffs              # show mismatches only
"""

import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from deterministic import try_deterministic
from normalize import normalize_answer

HERE = os.path.dirname(__file__)


def load_dataset():
    with open(os.path.join(HERE, "tasks.json")) as f:
        tasks = json.load(f)
    with open(os.path.join(HERE, "expected.json")) as f:
        expected = json.load(f)
    return tasks, expected


def compare(a, b):
    return a.strip().lower() == b.strip().lower()


def run_deterministic(tasks, expected):
    results = []
    for task in tasks:
        tid = task["task_id"]
        prompt = task["prompt"]
        exp = expected.get(tid)
        if exp is None:
            continue

        raw = try_deterministic(prompt)
        answer = normalize_answer(prompt, raw) if raw else None

        correct = compare(answer, exp) if answer else False
        results.append({
            "task_id": tid, "answer": answer, "expected": exp,
            "correct": correct, "source": "det" if answer else "missed"
        })
    return results


async def run_pipeline(tasks, expected):
    from clients import FireworksClient
    from model_select import load_allowed_models, select_models
    from prompts import SYSTEM_PROMPT, MAX_TOKENS, TEMPERATURE
    from normalize import normalize_answer

    models = select_models(load_allowed_models())
    client = FireworksClient()
    sem = asyncio.Semaphore(6)

    results = []
    for task in tasks:
        tid = task["task_id"]
        prompt = task["prompt"]
        exp = expected.get(tid)
        if exp is None:
            continue

        async with sem:
            try:
                raw, tokens, ms = await client.complete(
                    models["strongest"], SYSTEM_PROMPT, prompt, MAX_TOKENS,
                    temperature=TEMPERATURE
                )
                answer = normalize_answer(prompt, raw) if raw else ""
                results.append({
                    "task_id": tid, "answer": answer, "expected": exp,
                    "correct": compare(answer, exp), "source": "api",
                    "tokens": tokens, "latency_ms": ms, "model": models["strongest"]
                })
            except Exception as e:
                results.append({
                    "task_id": tid, "answer": f"<error: {e}>", "expected": exp,
                    "correct": False, "source": "error"
                })
    return results


def report(results):
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    sources = {}
    for r in results:
        s = r.get("source", "?")
        sources.setdefault(s, {"total": 0, "correct": 0, "tokens": 0})
        sources[s]["total"] += 1
        if r["correct"]:
            sources[s]["correct"] += 1
        sources[s]["tokens"] += r.get("tokens", 0)

    sep = "-" * 50
    print()
    print(sep)
    print(f"  Accuracy: {correct}/{total} = {correct/total*100:.1f}%")
    print(sep)
    for src, counts in sorted(sources.items()):
        pct = counts["correct"] / counts["total"] * 100
        tok = counts["tokens"]
        print(f"  {src}: {counts['correct']}/{counts['total']} = {pct:.1f}%  [{tok} tokens]")
    print(sep)

    for r in results:
        if not r["correct"]:
            exp = (r["expected"] or "")[:80]
            ans = (r["answer"] or "(None)")[:80]
            print(f"  X {r['task_id']}: expected={exp!r}")
            print(f"    got={ans!r}")


def report_summary(results):
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    print(f"Accuracy: {correct}/{total} = {correct/total*100:.1f}%")
    mismatches = [r for r in results if not r["correct"]]
    if mismatches:
        print(f"  {len(mismatches)} mismatches (use --diffs to see details)")


async def main():
    tasks, expected = load_dataset()
    use_api = "--with-api" in sys.argv
    show_diffs = "--diffs" in sys.argv

    start = time.monotonic()

    if use_api:
        results = await run_pipeline(tasks, expected)
    else:
        results = run_deterministic(tasks, expected)

    elapsed = time.monotonic() - start

    if show_diffs:
        report(results)
    else:
        report_summary(results)

    print(f"  [{elapsed:.2f}s]")


if __name__ == "__main__":
    asyncio.run(main())
