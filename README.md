# nanoserve

A minimal, hackable LLM serving stack — think nanoGPT, but for *serving*.
Built as a foundation for learning inference/serving engineering on an
Apple Silicon MacBook (target: M4 Max/Pro, 48GB unified memory) with
[Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B).

Every file is short and meant to be read. No frameworks doing magic:
one engine interface, one HTTP server, one benchmark loop, one plotting script.

New here? Start with the visual, end-to-end
[getting-started guide](education/GETTING_STARTED.md), then use this README as
the compact command reference.

```
nanoserve/
├── education/          guides, experiments, and interactive visualizations
│   ├── GETTING_STARTED.md
│   ├── LEARNING.md
│   ├── experiments.md
│   └── visualizations/index.html
├── nanoserve/
│   ├── config.py     settings + env overrides
│   ├── engine.py     Engine interface; MockEngine (any laptop), MLXEngine (Apple Silicon)
│   ├── server.py     OpenAI-compatible chat API with SSE streaming + /metrics
│   ├── bench.py      sweep prompt lengths / max_tokens -> JSONL rows of metrics
│   ├── lab.py        run a declared experiment: prediction + sweep + machine state
│   ├── report.py     score prediction vs measurement, charts, markdown writeup
│   ├── plot.py       ad-hoc JSONL -> PNG charts
│   └── memory.py     weights + KV-cache memory math (fits-in-RAM calculator)
├── experiments/          one TOML per experiment + STATUS.md log
├── scripts/download.py   pull weights from HuggingFace
├── tests/                runs anywhere, no MLX or weights required
└── results/<exp-id>/     runs.jsonl, env.json, report.md, charts
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate

# any laptop (no Apple Silicon needed):
pip install -e ".[dev]"

# on the M4 (adds mlx-lm):
pip install -e ".[mac,viz,dev]"
```

Run the whole pipeline with **zero downloads** using the mock engine:

```bash
python -m pytest -q                                        # tests pass anywhere

python -m nanoserve.bench --engine mock \
    --prompt-tokens 16 128 512 2048 --max-tokens 64 \
    --out results/mock.jsonl                               # fake "inference"

python -m nanoserve.plot results/mock.jsonl --out results  # charts from it

NANOSERVE_ENGINE=mock python -m nanoserve.server           # serve the mock

curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":32,"stream":true}'
```

## On the M4: serving real Qwen3-14B

Qwen3-14B is the baseline model. Its official BF16 checkpoint fits in 48GB,
while a 4-bit build leaves much more room for context and experiments. Check any
config before downloading with the memory calculator:

```bash
python -m nanoserve.memory --params 14.8 --bits 4 --layers 40 --kv-heads 8 --head-dim 128
# weights: 6.9 GiB | ctx 32k: +5 GiB KV cache -> ~13 GiB total. Fits.
```

Get weights (a community MLX 4-bit quant, or convert your own). Nanoserve
automatically uses this local copy when it exists:

```bash
python scripts/download.py mlx-community/Qwen3-14B-4bit
# or build your own 4-bit from official weights:
python -m mlx_lm convert --hf-path Qwen/Qwen3-14B -q
```

Serve it. On Apple Silicon, this now needs no model flags: `auto` finds MLX,
and the default model is the 4-bit Qwen3-14B build (rather than the ~30 GB BF16
checkpoint):

```bash
python -m nanoserve.server

# optional: show Qwen3's reasoning before its answer
NANOSERVE_THINKING=true python -m nanoserve.server
```

Benchmark it — the experiment that matters first is **decode speed vs context
length** (KV cache grows, so tokens/sec usually falls):

```bash
python -m nanoserve.bench --engine mlx \
    --prompt-tokens 128 1024 4096 16384 32768 \
    --max-tokens 256 --repeats 3 --label 4bit --out results/qwen14b.jsonl

python -m nanoserve.plot results/qwen14b.jsonl --out results
```

## The lab: predict, measure, explain

`bench.py` gives you numbers. `lab.py` makes them mean something, by forcing a
written numeric prediction *before* the run and scoring it afterwards.

```bash
python -m nanoserve.lab list                  # what exists, what's been run
python -m nanoserve.lab run 000-smoke         # mock engine, ~15s, no weights
python -m nanoserve.report 000-smoke          # scorecard + charts + writeup
```

Each experiment is one TOML file in `experiments/`:

```bash
python -m nanoserve.lab new 002-output-length --title "Output length vs latency"
# edit experiments/002-output-length.toml — especially [prediction] — then run it
```

The spec holds the question, the hypothesis, the **prediction with its
arithmetic**, and the sweep grid. Running it captures machine state (chip, RAM,
AC power, thermal pressure before/after, git sha, mlx version) alongside the
measurements, and discards `warmup` runs so cold weights don't skew the median.

`report.py` then writes `results/<id>/report.md` containing:

- a **scorecard**: predicted vs measured per point, with % error
- median results with min/max spread (noisy runs are visible, not hidden)
- measured-vs-predicted chart, TTFT, peak RAM, and a **thermal drift chart**
  (decode tok/s by run order — a downward slope means the machine got hot, not
  that your variable did anything)
- a blank *What I learned* section you fill in yourself

and appends a row to `experiments/STATUS.md`.

`001-context-decode.toml` is ready to run on the M4 with predictions already
derived from KV-cache arithmetic — start there:

```bash
caffeinate -i python -m nanoserve.lab run 001-context-decode
python -m nanoserve.report 001-context-decode
```

See [`education/LEARNING.md`](education/LEARNING.md) for the reading list and
the reasoning behind this loop.

## What to measure (the learning curriculum)

Each metric below maps to a classic serving-engineering topic:

| Experiment | Command knob(s) | What you learn |
|---|---|---|
| Decode speed vs context length | `--prompt-tokens` | KV cache cost, why long context slows generation |
| TTFT vs prompt length | `--prompt-tokens` | prefill is compute-bound, decode is memory-bandwidth-bound |
| Quantization trade-offs | run same sweep at 4-bit vs 8-bit repos | bandwidth vs quality, memory headroom |
| Sampling params don't change speed... verify it | `--temperature` | decode loop is sampling-agnostic |
| Streaming vs batch latency | server `stream: true/false` | per-token latency vs throughput |
| Mock engine as simulator | `MockEngine(words_per_second=N)` | build intuition without burning battery |

The JSONL rows are flat dicts — load them into pandas/polars for your own
analysis beyond the built-in plots.

## Concepts cheat-sheet

- **Prefill**: process all prompt tokens at once (parallel, compute-bound).
  Measured by `ttft_s` / `prefill_tps`.
- **Decode**: emit one token per forward pass (sequential, bound by memory
  bandwidth — each step re-reads all weights + the whole KV cache).
  Measured by `decode_tps`.
- **KV cache**: per-token memory =
  `2 × n_layers × n_kv_heads × head_dim × dtype_bytes`. GQA (`n_kv_heads` <
  `n_heads`) is why modern models can afford long context.
- **Quantization**: fewer bits = less bytes to move per token = higher decode
  speed on Macs, plus smaller footprint. This is why 4-bit often *doubles*
  tokens/sec vs bf16 on Apple Silicon.

## Roadmap ideas (in rough order of value)

1. Prompt-cache reuse across requests (same system prefix → skip re-prefill)
2. Concurrency: measure how N parallel streams share the engine
3. Speculative decoding with a tiny draft model
4. Batch-size sweeps once batching exists (throughput vs latency curves)
5. Swap in llama.cpp backend behind the same Engine interface, compare

## Notes

- The mock engine's numbers are meaningless as performance data — they exist
  to validate plumbing and to let you develop the eval tooling anywhere.
- `results/` artifacts are gitignored except `.gitkeep`; charts you want to
  keep should be copied elsewhere or force-added.
