# nanoserve

A minimal, hackable LLM serving stack — think nanoGPT, but for *serving*.
Built as a foundation for learning inference/serving engineering on an
Apple Silicon MacBook (target: M4 Max/Pro, 48GB unified memory) with
[Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B).

Every file is short and meant to be read. No frameworks doing magic:
one engine interface, one HTTP server, one benchmark loop, one plotting script.

```
nanoserve/
├── nanoserve/
│   ├── config.py     settings + env overrides
│   ├── engine.py     Engine interface; MockEngine (any laptop), MLXEngine (Apple Silicon)
│   ├── server.py     OpenAI-compatible chat API with SSE streaming + /metrics
│   ├── bench.py      sweep prompt lengths / max_tokens -> JSONL rows of metrics
│   ├── plot.py       JSONL -> PNG charts (decode tok/s, TTFT vs context)
│   └── memory.py     weights + KV-cache memory math (fits-in-RAM calculator)
├── scripts/download.py   pull weights from HuggingFace
├── tests/                runs anywhere, no MLX or weights required
└── results/              benchmark outputs (jsonl + png)
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

Get weights (a community MLX 4-bit quant, or convert your own):

```bash
python scripts/download.py mlx-community/Qwen3-14B-4bit
# or build your own 4-bit from official weights:
python -m mlx_lm convert --hf-path Qwen/Qwen3-14B -q
```

Serve it:

```bash
NANOSERVE_ENGINE=mlx NANOSERVE_MODEL=<path-or-repo-id> python -m nanoserve.server
```

Benchmark it — the experiment that matters first is **decode speed vs context
length** (KV cache grows, so tokens/sec usually falls):

```bash
python -m nanoserve.bench --engine mlx --model <model-path> \
    --prompt-tokens 128 1024 4096 16384 32768 \
    --max-tokens 256 --repeats 3 --label 4bit --out results/qwen14b.jsonl

python -m nanoserve.plot results/qwen14b.jsonl --out results
```

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
