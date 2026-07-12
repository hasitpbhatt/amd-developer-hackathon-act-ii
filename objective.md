# Objective

**Minimize the number of tasks sent to the Fireworks API.**

Every task solved locally on the CPU costs zero tokens. Every task sent to Fireworks costs tokens and counts against our score.

## Constraints

- CPU-only inference, 2 GB memory
- No GPU, no vLLM, no large models
- Small models only (sub-1B parameters), fine-tuned for low latency on CPU
- Fireworks API available but penalized in scoring

## Strategy

1. **CPU layer first** — use deterministic solvers + small local models for zero-token answers
2. **Fireworks only as fallback** — when CPU cannot produce a confident answer
3. **Token efficiency** — the fewer API calls, the better the score

## Goal

Pass the LLM-Judge accuracy gate with the fewest possible Fireworks API calls.
