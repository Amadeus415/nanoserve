# AGENTS.md

## What this is

nanoserve — a minimal, hackable LLM serving stack for learning inference/serving
engineering on Apple Silicon (MacBook Pro M4, 48GB unified memory). Inspired by
karpathy's nanoGPT: every file short, readable, no magic. **Simplicity and
understandability outrank features, performance, and robustness.**

When making any change, ask: "could someone learning serving read this and
understand it?" If a change adds cleverness without adding insight, don't make it.

## Stack

- Python >= 3.10, no heavy frameworks. Deps: fastapi, uvicorn, huggingface-hub.
- `mlx-lm` only for the Apple Silicon engine (optional extra `[mac]`).
- Tests run anywhere with the MockEngine — no MLX or model weights required.

## Layout

```
nanoserve/
├── config.py     settings + env overrides (NANOSERVE_* vars)
├── engine.py     Engine interface; MockEngine + MLXEngine implementations
├── server.py     OpenAI-compatible chat API with SSE streaming + /metrics
├── bench.py      sweep prompt/max_tokens -> JSONL metric rows
├── plot.py       JSONL -> PNG charts
└── memory.py     weights + KV-cache memory math
scripts/download.py   pull weights from HuggingFace
tests/                pytest, mock-only, fast
results/              benchmark outputs (gitignored except .gitkeep)
```

## Conventions

- One file = one concept. Don't grow files past a few hundred lines; split by
  concept instead.
- Prefer plain functions and small classes over abstractions, registries, or
  plugin systems. The `Engine` interface in engine.py is the one allowed
  abstraction.
- Flat dict JSONL rows for benchmark output; no schemas or ORMs.
- No comments explaining *what* — code should show that. Comments only for
  *why* (e.g., a non-obvious inference/serving insight worth teaching).
- No dependencies beyond what's already declared unless truly necessary;
  ask first.
- Env config via `NANOSERVE_*` variables through config.py — no config files.

## Commands

```bash
pip install -e ".[dev]"            # setup (add ,mac,viz on the M4)
python -m pytest -q                # tests (must pass before any commit)
python -m nanoserve.bench --engine mock --prompt-tokens 16 128 --max-tokens 64 --out results/mock.jsonl
python -m nanoserve.plot results/mock.jsonl --out results
NANOSERVE_ENGINE=mock python -m nanoserve.server   # serve on :8000
```

## Definition of done

1. `python -m pytest -q` passes.
2. The feature works end-to-end with `--engine mock` / `NANOSERVE_ENGINE=mock`
   (no weights needed).
3. A newcomer could explain the changed code back after reading it once.

## Things to avoid

- Frameworks, DI containers, async complexity beyond what FastAPI requires.
- Backward compatibility shims — this is a learning repo, just change things.
- Optimizations that obscure the algorithm. Correctness and clarity first;
  speed comes from understanding, not tricks.
